from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch
from gymnasium import spaces

from tools import build_alphazuma_55_macro_policy_bootstrap_v1 as bootstrap
from tools import distill_alphazuma_55_park_settle_v1 as v1
from tools import distill_alphazuma_55_park_settle_v3 as trainer_v3
from zuma_rl.observation_stack_wrapper import stacked_box


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STACK_FRAMES = 3
SOURCE_SEED = 1_543_003_100
TARGET_SEED = 99_081_679


def _tiny_stacked_per_tick_model(tmp_path: Path):
    """Small stacked per-tick BC-shaped student, saved like distill v3's."""

    observation_space, action_space, policy_kwargs = (
        v1.tiny_entity_polar_interface(features_dim=24)
    )
    stacked_kwargs = trainer_v3.stack_policy_kwargs(
        policy_kwargs, frames=STACK_FRAMES
    )
    model = v1.build_student_model(
        observation_space=stacked_box(observation_space, STACK_FRAMES),
        action_space=action_space,
        policy_kwargs=stacked_kwargs,
        model_config=v1.ParkSettleModelConfig(
            learned_features_dim=24, model_seed=SOURCE_SEED
        ),
        train_config=v1.ParkSettleTrainConfig(batch_size=8, device="cpu"),
    )
    source_path = tmp_path / "source_per_tick.zip"
    model.save(str(source_path))
    return model, source_path


@pytest.fixture(scope="module")
def transplant(tmp_path_factory: pytest.TempPathFactory):
    tmp_path = tmp_path_factory.mktemp("bootstrap")
    source_model, source_path = _tiny_stacked_per_tick_model(tmp_path)
    output_dir = tmp_path / "out"
    receipt = bootstrap.run(
        source_path=source_path,
        output_dir=output_dir,
        device="cpu",
        model_seed=TARGET_SEED,
    )
    return source_model, source_path, output_dir, receipt


# ---------------------------------------------------------------------------
# Shape and weight-equality assertions, layer by layer.
# ---------------------------------------------------------------------------


def test_transplant_interfaces(transplant) -> None:
    from sb3_contrib import MaskablePPO

    source_model, _, output_dir, receipt = transplant
    target = MaskablePPO.load(
        str(output_dir / bootstrap.OUTPUT_MODEL_NAME), device="cpu"
    )
    assert target.action_space.nvec.tolist() == [3, 180]
    assert source_model.action_space.nvec.tolist() == [4, 180]
    assert target.observation_space == source_model.observation_space
    # The stacked extractor class resolves from the distill-v3 module.
    assert type(target.policy.features_extractor) is (
        trainer_v3.StackedRevengeEntityFeatureExtractor
    )
    assert receipt["action_nvec"] == {
        "source": [4, 180],
        "target": [3, 180],
    }


def test_layer_by_layer_weight_equality(transplant) -> None:
    from sb3_contrib import MaskablePPO

    source_model, _, output_dir, _ = transplant
    target = MaskablePPO.load(
        str(output_dir / bootstrap.OUTPUT_MODEL_NAME), device="cpu"
    )
    source_state = source_model.policy.state_dict()
    target_state = target.policy.state_dict()
    checked_actor = checked_head = checked_value = 0
    for key, value in target_state.items():
        if key.startswith(bootstrap._ACTOR_PREFIXES):
            assert torch.equal(value, source_state[key]), key
            checked_actor += 1
        elif key in ("action_net.weight", "action_net.bias"):
            # Macro verb rows 0..2 come from per-tick rows 0..2 (hop row
            # 3 dropped); aim rows are the per-tick rows 4.. verbatim.
            assert value.shape[0] == 183
            assert torch.equal(value[:3], source_state[key][:3]), key
            assert torch.equal(value[3:], source_state[key][4:]), key
            checked_head += 1
        elif key.startswith(bootstrap._VALUE_PREFIXES):
            checked_value += 1
        else:
            raise AssertionError(f"unclassified policy parameter: {key}")
    assert checked_actor >= 1
    assert checked_head == 2
    # net_arch {"pi": [], "vf": [32]}: hidden value layer plus final head.
    assert checked_value == 4


def test_value_pathway_is_fresh_not_copied(transplant) -> None:
    from sb3_contrib import MaskablePPO

    source_model, _, output_dir, receipt = transplant
    target = MaskablePPO.load(
        str(output_dir / bootstrap.OUTPUT_MODEL_NAME), device="cpu"
    )
    source_state = source_model.policy.state_dict()
    target_state = target.policy.state_dict()
    fresh_weight_keys = [
        key
        for key in target_state
        if key.startswith(bootstrap._VALUE_PREFIXES)
        and key.endswith("weight")
    ]
    assert fresh_weight_keys
    for key in fresh_weight_keys:
        assert not torch.equal(target_state[key], source_state[key]), (
            f"value parameter was copied from the source: {key}"
        )
    assert "fresh" in receipt["value_head_init"]
    assert receipt["model_seed"] == TARGET_SEED


def test_logits_equivalence_on_random_stacked_observations(
    transplant,
) -> None:
    from sb3_contrib import MaskablePPO

    source_model, _, output_dir, receipt = transplant
    target = MaskablePPO.load(
        str(output_dir / bootstrap.OUTPUT_MODEL_NAME), device="cpu"
    )
    rng = np.random.default_rng(7)
    observations = rng.uniform(
        -1.0, 1.0, size=(5, int(target.observation_space.shape[0]))
    ).astype(np.float32)
    source_model.policy.set_training_mode(False)
    target.policy.set_training_mode(False)
    with torch.no_grad():
        source_tensor, _ = source_model.policy.obs_to_tensor(observations)
        target_tensor, _ = target.policy.obs_to_tensor(observations)
        source_verbs, source_aims = v1._policy_logits(
            source_model, source_tensor
        )
        target_verbs, target_aims = v1._policy_logits(target, target_tensor)
    assert torch.equal(source_verbs[:, :3], target_verbs)
    assert torch.equal(source_aims, target_aims)
    equivalence = receipt["logits_equivalence"]
    assert equivalence["verb_logits_bitwise_equal_rows_0_1_2"] is True
    assert equivalence["aim_logits_bitwise_equal"] is True


def test_parameter_count_check(transplant) -> None:
    source_model, _, _, receipt = transplant
    counts = receipt["parameter_counts"]
    features_in = int(source_model.policy.action_net.in_features)
    assert counts["difference"] == features_in + 1
    assert counts["expected_difference"] == features_in + 1
    assert counts["difference_ok"] is True
    assert counts["source_total"] - counts["target_total"] == (
        features_in + 1
    )


# ---------------------------------------------------------------------------
# MaskablePPO.load round-trip + masked predict on a synthetic stacked obs.
# ---------------------------------------------------------------------------


def test_round_trip_masked_predict_respects_the_macro_mask(
    transplant,
) -> None:
    from sb3_contrib import MaskablePPO

    _, _, output_dir, receipt = transplant
    model = MaskablePPO.load(
        str(output_dir / bootstrap.OUTPUT_MODEL_NAME), device="cpu"
    )
    assert receipt["round_trip_bitwise_equal"] is True
    observation = np.zeros(
        (1, int(model.observation_space.shape[0])), dtype=np.float32
    )
    for verb in range(3):
        mask = np.zeros(183, dtype=np.bool_)
        mask[verb] = True
        mask[3:] = True
        action, _ = model.predict(
            observation, deterministic=True, action_masks=mask
        )
        values = np.asarray(action).reshape(-1)
        assert int(values[0]) == verb
        assert 0 <= int(values[1]) < 180
    # An aim-restricted mask is honored too.
    mask = np.zeros(183, dtype=np.bool_)
    mask[:3] = True
    mask[3 + 77] = True
    action, _ = model.predict(
        observation, deterministic=True, action_masks=mask
    )
    assert int(np.asarray(action).reshape(-1)[1]) == 77


def test_zip_loads_without_the_trainer_or_builder_modules(
    transplant,
) -> None:
    """The saved zip resolves its pickled module refs on a bare load.

    A fresh interpreter with only the repository on sys.path (neither
    the transplant builder nor any trainer imported by hand) must load
    the zip and run a masked predict: unpickling policy_kwargs imports
    tools.distill_alphazuma_55_park_settle_v3 by itself.
    """

    _, _, output_dir, _ = transplant
    model_path = output_dir / bootstrap.OUTPUT_MODEL_NAME
    script = """
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
sys.path.insert(0, str(root))
sys.path.insert(0, str(root / "src"))

import numpy as np
from sb3_contrib import MaskablePPO

model = MaskablePPO.load(sys.argv[2], device="cpu")
assert model.action_space.nvec.tolist() == [3, 180]
assert "tools.build_alphazuma_55_macro_policy_bootstrap_v1" not in sys.modules
assert "tools.train_overnight_multilevel_v2" not in sys.modules
assert "tools.distill_alphazuma_55_park_settle_v3" in sys.modules

observation = np.zeros(
    (1, int(model.observation_space.shape[0])), dtype=np.float32
)
mask = np.zeros(183, dtype=bool)
mask[1] = True
mask[3:] = True
action, _ = model.predict(observation, deterministic=True, action_masks=mask)
values = np.asarray(action).reshape(-1)
assert int(values[0]) == 1
print(json.dumps({"status": "LOADED", "aim": int(values[1])}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script, str(PROJECT_ROOT), str(model_path)],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout.strip().splitlines()[-1])
    assert payload["status"] == "LOADED"
    assert 0 <= payload["aim"] < 180


# ---------------------------------------------------------------------------
# Receipt contract.
# ---------------------------------------------------------------------------


def test_receipt_contract(transplant) -> None:
    _, source_path, output_dir, receipt = transplant
    on_disk = json.loads(
        (output_dir / bootstrap.RECEIPT_NAME).read_text(encoding="utf-8")
    )
    assert on_disk["schema"] == bootstrap.RECEIPT_SCHEMA
    assert on_disk["source"]["path"] == str(source_path.resolve())
    assert on_disk["source"]["sha256"].startswith("sha256:")
    assert on_disk["output"]["sha256"].startswith("sha256:")
    assert on_disk["output"]["bytes"] > 0
    assert on_disk["formal_seed_consumption"] is False
    assert on_disk["training_authority"] is False
    assert on_disk["verb_row_map"]["dropped"] == "per-tick hop (row 3)"
    modes = {entry["mode"] for entry in on_disk["transplant_map"]}
    assert modes == {
        "verbatim",
        "verb_rows_0_1_2_then_aim_rows_verbatim",
        "fresh_target_init",
    }
    # Every policy parameter is accounted for exactly once.
    from sb3_contrib import MaskablePPO

    target = MaskablePPO.load(
        str(output_dir / bootstrap.OUTPUT_MODEL_NAME), device="cpu"
    )
    mapped = {entry["target"] for entry in on_disk["transplant_map"]}
    assert mapped == set(target.policy.state_dict().keys())
    total_elements = sum(
        entry["elements"] for entry in on_disk["transplant_map"]
    )
    # The state dict aliases the SHARED feature extractor under three
    # prefixes, so the map's element total counts shared tensors once per
    # alias; parameters() deduplicates, hence target_total is smaller.
    state_elements = sum(
        value.numel() for value in target.policy.state_dict().values()
    )
    assert total_elements == state_elements
    assert on_disk["parameter_counts"]["target_total"] <= total_elements


def test_source_interface_is_validated() -> None:
    from sb3_contrib import MaskablePPO

    observation_space, _, policy_kwargs = v1.tiny_entity_polar_interface(
        features_dim=8
    )
    wrong_space = spaces.MultiDiscrete(np.array((3, 180), dtype=np.int64))
    model = MaskablePPO(
        "MlpPolicy",
        v1._SpaceOnlyMaskableEnv(observation_space, wrong_space),
        policy_kwargs=policy_kwargs,
        n_steps=8,
        batch_size=8,
        device="cpu",
        verbose=0,
    )
    with pytest.raises(ValueError, match="per-tick"):
        bootstrap.build_macro_model_from_per_tick(model, model_seed=1)


def test_output_dir_is_never_reused(tmp_path: Path) -> None:
    _, source_path = _tiny_stacked_per_tick_model(tmp_path)
    output_dir = tmp_path / "occupied"
    output_dir.mkdir()
    with pytest.raises(FileExistsError):
        bootstrap.run(source_path=source_path, output_dir=output_dir)
