"""Safety-gate tests for the diagnostic PPO entry point."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from zuma_rl import train
from zuma_rl.fidelity_gate import FidelityGateReport, FidelityGateStatus
from zuma_rl.original_data import OriginalDataError
from zuma_rl.teacher_anchor import masked_distribution_forward_kl
from zuma_rl.training_gate import TrainingGateReport, TrainingGateStatus


def _args(*extra: str):
    return train.parse_args(["--acknowledge-fidelity-gate", *extra])


def test_training_is_disabled_until_closed_gate_is_acknowledged() -> None:
    args = train.parse_args([])
    assert args.environment == "revenge"
    assert args.acknowledge_fidelity_gate is False

    with pytest.raises(SystemExit, match=r"gate is CLOSED"):
        train._validate_args(args)


def test_transfer_training_requires_suite_and_rejects_closed_ack() -> None:
    with pytest.raises(SystemExit, match="requires --fidelity-suite"):
        train._validate_args(
            train.parse_args(["--transfer-training"])
        )

    with pytest.raises(SystemExit, match="requires --training-suite"):
        train._validate_args(
            train.parse_args(
                [
                    "--transfer-training",
                    "--fidelity-suite",
                    "fidelity.json",
                ]
            )
        )

    with pytest.raises(SystemExit, match="cannot be combined"):
        train._validate_args(
            train.parse_args(
                [
                    "--transfer-training",
                    "--fidelity-suite",
                    "suite.json",
                    "--acknowledge-fidelity-gate",
                ]
            )
        )


def test_open_gate_mode_lifts_only_closed_gate_step_cap() -> None:
    args = train.parse_args(
        [
            "--transfer-training",
            "--fidelity-suite",
            "suite.json",
            "--training-suite",
            "training.json",
            "--total-steps",
            "1000000",
        ]
    )

    train._validate_args(args)
    assert train._planned_effective_steps(args) >= 1_000_000


def test_transfer_gate_refuses_closed_suite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = train.parse_args(
        [
            "--transfer-training",
            "--fidelity-suite",
            str(tmp_path / "suite.json"),
            "--level",
            "Jungle2",
        ]
    )
    monkeypatch.setattr(
        train,
        "verify_fidelity_suite",
        lambda *args, **kwargs: FidelityGateReport(
            status=FidelityGateStatus.CLOSED,
            policy="original-transfer-jungle2-v1",
            reasons=("missing certifying evidence: match3",),
        ),
    )

    with pytest.raises(SystemExit, match=r"suite is CLOSED.*match3"):
        train._verify_transfer_gate(args, original_root=tmp_path)


def test_open_suite_is_bound_to_exact_training_scope(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = FidelityGateReport(
        status=FidelityGateStatus.OPEN,
        policy="original-transfer-jungle2-v4",
        summary={"gate_open": True},
    )
    monkeypatch.setattr(
        train,
        "verify_fidelity_suite",
        lambda *args, **kwargs: report,
    )
    valid = train.parse_args(
        [
            "--transfer-training",
            "--fidelity-suite",
            str(tmp_path / "suite.json"),
            "--level",
            "Jungle2",
        ]
    )
    assert (
        train._verify_transfer_gate(valid, original_root=tmp_path)
        is report
    )

    invalid = train.parse_args(
        [
            "--transfer-training",
            "--fidelity-suite",
            str(tmp_path / "suite.json"),
            "--level",
            "Jungle1",
        ]
    )
    with pytest.raises(SystemExit, match="outside policy"):
        train._verify_transfer_gate(invalid, original_root=tmp_path)


def test_legacy_open_fidelity_policy_cannot_authorize_transfer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = FidelityGateReport(
        status=FidelityGateStatus.OPEN,
        policy="original-transfer-jungle2-v1",
        summary={"gate_open": True},
    )
    monkeypatch.setattr(
        train,
        "verify_fidelity_suite",
        lambda *args, **kwargs: report,
    )
    args = train.parse_args(
        [
            "--transfer-training",
            "--fidelity-suite",
            str(tmp_path / "suite.json"),
            "--level",
            "Jungle2",
        ]
    )

    with pytest.raises(SystemExit, match="historical evidence only"):
        train._verify_transfer_gate(args, original_root=tmp_path)


def test_training_gate_is_independent_and_binds_the_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fidelity_report = FidelityGateReport(
        status=FidelityGateStatus.OPEN,
        policy="original-transfer-jungle2-v4",
    )
    fidelity_suite = tmp_path / "fidelity.json"
    fidelity_suite.write_text("{}", encoding="utf-8")
    args = train.parse_args(
        [
            "--transfer-training",
            "--fidelity-suite",
            str(fidelity_suite),
            "--training-suite",
            str(tmp_path / "training.json"),
        ]
    )
    closed = TrainingGateReport(
        status=TrainingGateStatus.CLOSED,
        policy="jungle2-teacher-anchor-v1",
        stage="state-policy-calibration-491520-v1",
        reasons=("prerequisite failed",),
    )
    monkeypatch.setattr(
        train,
        "verify_training_suite",
        lambda *args, **kwargs: closed,
    )
    with pytest.raises(SystemExit, match=r"training suite is CLOSED"):
        train._verify_training_gate(
            args,
            fidelity_report=fidelity_report,
            original_root=tmp_path,
        )

    opened = TrainingGateReport(
        status=TrainingGateStatus.OPEN,
        policy=train.TRANSFER_POLICY_ID,
        stage=train.TRANSFER_STAGE_ID,
        summary={
            "required_fidelity_policy": "original-transfer-jungle2-v4",
            "transfer_authorizing_policy": True,
            "required_fidelity_suite_sha256": (
                "sha256:"
                + hashlib.sha256(fidelity_suite.read_bytes()).hexdigest()
            ),
        },
    )
    monkeypatch.setattr(
        train,
        "verify_training_suite",
        lambda *args, **kwargs: opened,
    )
    with pytest.raises(SystemExit, match="differs from the fixed"):
        train._verify_training_gate(
            args,
            fidelity_report=fidelity_report,
            original_root=tmp_path,
        )


def test_transfer_run_directory_must_be_new(tmp_path: Path) -> None:
    args = train.parse_args(
        [
            "--transfer-training",
            "--fidelity-suite",
            "fidelity.json",
            "--training-suite",
            "training.json",
            "--run-dir",
            str(tmp_path),
        ]
    )

    with pytest.raises(SystemExit, match="must not already exist"):
        train._verify_new_transfer_run_directory(args)


def test_open_gate_run_metadata_is_content_addressed(
    tmp_path: Path,
) -> None:
    suite = tmp_path / "suite.json"
    suite.write_text('{"suite":"test"}', encoding="utf-8")
    training_suite = tmp_path / "training-suite.json"
    training_suite.write_text('{"suite":"training"}', encoding="utf-8")
    args = train.parse_args(
        [
            "--transfer-training",
            "--fidelity-suite",
            str(suite),
            "--training-suite",
            str(training_suite),
        ]
    )
    report = FidelityGateReport(
        status=FidelityGateStatus.OPEN,
        policy="original-transfer-jungle2-v2",
        summary={"gate_open": True},
    )
    training_report = TrainingGateReport(
        status=TrainingGateStatus.OPEN,
        policy="jungle2-teacher-anchor-v1",
        stage="state-policy-calibration-491520-v1",
        summary={"training_gate_open": True},
    )

    metadata = train._run_config(
        args,
        {"id": "ZumaRevenge-v0", "role": "fidelity_first_transfer"},
        tmp_path / "run",
        gate_report=report,
        training_gate_report=training_report,
    )

    assert metadata["diagnostic_non_transferable"] is False
    assert metadata["fidelity_gate"] == "open"
    assert metadata["closed_gate_max_effective_steps"] is None
    assert metadata["fidelity_suite"]["report"]["status"] == "OPEN"
    assert metadata["fidelity_suite"]["sha256"].startswith("sha256:")
    assert metadata["training_gate"] == "open"
    assert metadata["training_suite"]["report"]["status"] == "OPEN"
    assert metadata["training_suite"]["sha256"].startswith("sha256:")


def test_default_acknowledged_run_is_revenge_diagnostic() -> None:
    args = _args()
    train._validate_args(args)
    assert args.environment == "revenge"
    assert args.level == "Jungle1"
    assert args.frame_skip == 1
    assert args.training_time_limit_steps == 0
    assert args.max_balls == 192
    assert args.total_steps == 10_000
    assert args.eval_episodes == 20
    assert args.boundary_eval_episodes == 0
    assert train._planned_effective_steps(args) == 16_384


def test_closed_gate_hard_caps_complete_ppo_rollouts() -> None:
    with pytest.raises(SystemExit, match=r"capped at 100,000"):
        train._validate_args(_args("--total-steps", "100001"))

    with pytest.raises(SystemExit, match=r"complete rollouts"):
        train._validate_args(
            _args(
                "--total-steps",
                "10000",
                "--num-envs",
                "256",
            )
        )


def test_simple_reference_requires_a_second_explicit_opt_in() -> None:
    args = _args("--environment", "simple")
    with pytest.raises(SystemExit, match="reference-only"):
        train._validate_args(args)


def test_rejected_action_mask_is_revenge_only() -> None:
    args = _args(
        "--environment",
        "simple",
        "--allow-reference-smoke",
        "--mask-rejected-actions",
    )
    with pytest.raises(SystemExit, match="only applies to the revenge"):
        train._validate_args(args)


def test_initial_model_requires_factorized_maskable_revenge(
    tmp_path: Path,
) -> None:
    model = tmp_path / "model.zip"
    valid = _args(
        "--initial-model",
        str(model),
        "--mask-rejected-actions",
    )
    train._validate_args(valid)

    with pytest.raises(SystemExit, match="requires --mask-rejected-actions"):
        train._validate_args(_args("--initial-model", str(model)))
    with pytest.raises(SystemExit, match="requires factorized actions"):
        train._validate_args(
            _args(
                "--initial-model",
                str(model),
                "--mask-rejected-actions",
                "--flat-actions",
            )
        )
    transfer = train.parse_args(
        [
            "--transfer-training",
            "--fidelity-suite",
            "suite.json",
            "--training-suite",
            "training.json",
            "--initial-model",
            str(model),
            "--mask-rejected-actions",
        ]
    )
    train._validate_args(transfer)


def test_freeze_aim_path_requires_initialized_model(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="requires --initial-model"):
        train._validate_args(_args("--freeze-aim-path"))

    args = _args(
        "--initial-model",
        str(tmp_path / "model.zip"),
        "--mask-rejected-actions",
        "--freeze-aim-path",
    )
    train._validate_args(args)

    with pytest.raises(SystemExit, match="requires --initial-model"):
        train._validate_args(_args("--reset-optimizer-state"))
    reset_args = _args(
        "--initial-model",
        str(tmp_path / "model.zip"),
        "--mask-rejected-actions",
        "--reset-optimizer-state",
    )
    train._validate_args(reset_args)


def test_teacher_anchor_requires_a_reset_maskable_initial_model(
    tmp_path: Path,
) -> None:
    model = tmp_path / "model.zip"
    with pytest.raises(SystemExit, match="requires --initial-model"):
        train._validate_args(_args("--teacher-kl-coef", "1"))

    with pytest.raises(SystemExit, match="requires --reset-optimizer-state"):
        train._validate_args(
            _args(
                "--initial-model",
                str(model),
                "--mask-rejected-actions",
                "--teacher-kl-coef",
                "1",
            )
        )

    args = _args(
        "--initial-model",
        str(model),
        "--mask-rejected-actions",
        "--reset-optimizer-state",
        "--teacher-kl-coef",
        "1",
    )
    train._validate_args(args)

    with pytest.raises(SystemExit, match="cannot be combined"):
        train._validate_args(
            _args(
                "--initial-model",
                str(model),
                "--mask-rejected-actions",
                "--reset-optimizer-state",
                "--freeze-aim-path",
                "--teacher-kl-coef",
                "1",
            )
        )


def test_initial_model_run_metadata_is_content_addressed(
    tmp_path: Path,
) -> None:
    model = tmp_path / "model.zip"
    model.write_bytes(b"frozen-model")
    args = _args(
        "--initial-model",
        str(model),
        "--mask-rejected-actions",
    )

    metadata = train._run_config(
        args,
        {"id": "ZumaRevenge-v0"},
        tmp_path / "run",
    )

    assert metadata["initial_model"]["path"] == str(model.resolve())
    assert metadata["initial_model"]["sha256"].startswith("sha256:")
    assert metadata["training"]["initial_model"] == str(model.resolve())

    anchored_args = _args(
        "--initial-model",
        str(model),
        "--mask-rejected-actions",
        "--reset-optimizer-state",
        "--teacher-kl-coef",
        "2.0",
    )
    anchored_metadata = train._run_config(
        anchored_args,
        {"id": "ZumaRevenge-v0"},
        tmp_path / "anchored-run",
    )
    assert anchored_metadata["algorithm"] == (
        "teacher_anchored_maskable_ppo"
    )
    assert anchored_metadata["training"]["teacher_kl_coef"] == 2.0


def test_training_time_limit_is_diagnostic_revenge_only() -> None:
    with pytest.raises(SystemExit, match="only applies to the revenge"):
        train._validate_args(
            _args(
                "--environment",
                "simple",
                "--allow-reference-smoke",
                "--training-time-limit-steps",
                "128",
            )
        )

    with pytest.raises(SystemExit, match="cannot be used for transfer"):
        train._validate_args(
            train.parse_args(
                [
                    "--transfer-training",
                    "--fidelity-suite",
                    "suite.json",
                    "--training-suite",
                    "training.json",
                    "--training-time-limit-steps",
                    "128",
                ]
            )
        )


def test_training_time_limit_wrapper_preserves_full_eval_horizon() -> None:
    args = _args(
        "--max-ticks",
        "12000",
        "--training-time-limit-steps",
        "2048",
    )
    train._validate_args(args)
    wrapper = train._training_wrapper_class(args)

    assert wrapper is not None
    assert wrapper.func is train.TimeLimit
    assert wrapper.keywords == {"max_episode_steps": 2048}

    disabled = _args("--max-ticks", "12000")
    train._validate_args(disabled)
    assert train._training_wrapper_class(disabled) is None


def test_parallel_boundary_evaluation_arguments_are_consistent() -> None:
    with pytest.raises(SystemExit, match="cannot exceed eval-episodes"):
        train._validate_args(_args("--eval-num-envs", "21"))
    with pytest.raises(SystemExit, match="cannot exceed boundary-eval-episodes"):
        train._validate_args(
            _args(
                "--eval-num-envs",
                "4",
                "--boundary-eval-episodes",
                "2",
            )
        )
    with pytest.raises(SystemExit, match="requires a positive"):
        train._validate_args(
            _args("--also-evaluate-stochastic-boundaries")
        )

    args = _args(
        "--environment",
        "simple",
        "--allow-reference-smoke",
    )
    train._validate_args(args)


def test_reference_opt_in_is_rejected_for_revenge() -> None:
    args = _args("--allow-reference-smoke")
    with pytest.raises(SystemExit, match="only valid"):
        train._validate_args(args)


def test_numeric_validation_happens_before_sb3_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    imported = False

    def forbidden_import() -> dict[str, object]:
        nonlocal imported
        imported = True
        raise AssertionError("SB3 must not be imported")

    monkeypatch.setattr(train, "_load_training_dependencies", forbidden_import)
    with pytest.raises(SystemExit, match="total-steps must be positive"):
        train.main(
            [
                "--acknowledge-fidelity-gate",
                "--total-steps",
                "0",
            ]
        )
    assert imported is False


@pytest.mark.parametrize(
    ("flags", "message"),
    [
        (("--num-envs", "0"), "num-envs"),
        (("--rollout-steps", "1"), "rollout-steps"),
        (("--ppo-epochs", "0"), "ppo-epochs"),
        (("--teacher-kl-coef", "-1"), "teacher-kl-coef"),
        (("--batch-size", "1"), "batch-size"),
        (("--learning-rate", "0"), "learning-rate"),
        (("--reward-scale", "0"), "reward-scale"),
        (("--entropy-coef", "-0.1"), "entropy-coef"),
        (("--max-balls", "159"), "max-balls"),
        (("--eval-episodes", "0"), "eval-episodes"),
        (
            ("--boundary-eval-episodes", "-1"),
            "boundary-eval-episodes",
        ),
        (("--frame-skip", "0"), "frame-skip"),
        (("--max-ticks", "0"), "max-ticks"),
        (
            ("--training-time-limit-steps", "-1"),
            "training-time-limit-steps",
        ),
        (("--curve-index", "-1"), "curve-index"),
    ],
)
def test_invalid_numeric_arguments_are_rejected(
    flags: tuple[str, str],
    message: str,
) -> None:
    with pytest.raises(SystemExit, match=message):
        train._validate_args(_args(*flags))


def test_simple_run_metadata_is_permanently_marked_non_transferable(
    tmp_path: Path,
) -> None:
    args = _args(
        "--environment",
        "simple",
        "--allow-reference-smoke",
        "--original-root",
        "unused-relative-path",
    )
    env_class, env_kwargs, env_spec = train._environment_spec(args)
    metadata = train._run_config(args, env_spec, tmp_path.resolve())

    assert env_class.__name__ == "ZumaEnv"
    assert env_kwargs["config"].aim_bins == 180
    assert metadata["diagnostic_non_transferable"] is True
    assert metadata["fidelity_gate"] == "closed"
    assert metadata["fidelity_gate_acknowledged"] is True
    assert metadata["closed_gate_max_effective_steps"] == 100_000
    assert metadata["planned_effective_steps"] == 16_384
    assert metadata["environment"]["role"] == "reference_smoke_only"
    assert isinstance(metadata["training"]["original_root"], str)


def test_explicit_original_root_is_strict_and_clear(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match=r"ZumasRevenge\.exe.*main\.pak"):
        train._resolve_original_root(tmp_path)


def test_missing_original_installation_has_actionable_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing() -> Path:
        raise OriginalDataError("not found")

    monkeypatch.setattr(train, "find_original_installation", missing)
    with pytest.raises(
        SystemExit,
        match=r"--original-root PATH or set ZUMA_REVENGE_ROOT",
    ):
        train._resolve_original_root(None)


def test_main_uses_matching_train_and_eval_vector_types(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vector_calls: list[dict[str, object]] = []
    closed: list[object] = []
    learned: dict[str, object] = {}

    class DummyVecEnv:
        def close(self) -> None:
            closed.append(self)

    class SubprocVecEnv:
        def close(self) -> None:
            closed.append(self)

    def make_vec_env(
        env_class: object,
        *,
        n_envs: int,
        seed: int,
        env_kwargs: dict[str, object],
        vec_env_cls: type[DummyVecEnv] | type[SubprocVecEnv],
    ) -> DummyVecEnv | SubprocVecEnv:
        vector_calls.append(
            {
                "env_class": env_class,
                "n_envs": n_envs,
                "seed": seed,
                "env_kwargs": env_kwargs,
                "vec_env_cls": vec_env_cls,
            }
        )
        return vec_env_cls()

    class Callback:
        def __init__(
            self,
            *args: object,
            **kwargs: object,
        ) -> None:
            self.args = args
            self.kwargs = kwargs

    class PPO:
        def __init__(self, policy: str, env: object, **kwargs: object) -> None:
            self.num_timesteps = 0
            learned["policy"] = policy
            learned["env"] = env
            learned["constructor"] = kwargs

        def learn(self, **kwargs: object) -> None:
            learned["learn"] = kwargs
            self.num_timesteps = int(kwargs["total_timesteps"])

        def save(self, path: Path) -> None:
            learned["save"] = path
            path.with_suffix(".zip").write_bytes(b"fake-model")

    monkeypatch.setattr(
        train,
        "_load_training_dependencies",
        lambda: {
            "PPO": PPO,
            "CheckpointCallback": Callback,
            "EvalCallback": Callback,
            "make_vec_env": make_vec_env,
            "DummyVecEnv": DummyVecEnv,
            "SubprocVecEnv": SubprocVecEnv,
        },
    )
    monkeypatch.setattr(
        train,
        "_training_runtime",
        lambda: {"test_runtime": True},
    )
    run_dir = tmp_path / "run"

    train.main(
        [
            "--acknowledge-fidelity-gate",
            "--environment",
            "simple",
            "--allow-reference-smoke",
            "--total-steps",
            "4",
            "--num-envs",
            "2",
            "--rollout-steps",
            "2",
            "--batch-size",
            "2",
            "--run-dir",
            str(run_dir),
        ]
    )

    assert len(vector_calls) == 2
    assert vector_calls[0]["vec_env_cls"] is SubprocVecEnv
    assert vector_calls[1]["vec_env_cls"] is SubprocVecEnv
    assert vector_calls[0]["n_envs"] == 2
    assert vector_calls[1]["n_envs"] == 1
    assert len(closed) == 2
    learn_args = learned["learn"]
    assert isinstance(learn_args, dict)
    assert learn_args["total_timesteps"] == 4
    assert learn_args["progress_bar"] is True
    assert len(learn_args["callback"]) == 2
    constructor = learned["constructor"]
    assert isinstance(constructor, dict)
    assert constructor["n_epochs"] == 10
    assert learned["save"] == run_dir / "final_model"
    assert (run_dir / "config.json").is_file()
    completion = json.loads(
        (run_dir / "completion.json").read_text(encoding="utf-8")
    )
    assert completion["status"] == "COMPLETE"
    assert completion["effective_steps"] == 4
    config = json.loads(
        (run_dir / "config.json").read_text(encoding="utf-8")
    )
    assert config["runtime"] == {"test_runtime": True}


def test_boundary_evaluation_reseeds_and_writes_paired_metrics(
    tmp_path: Path,
) -> None:
    seeded: list[int] = []
    calls: list[dict[str, object]] = []

    class EvalEnv:
        def seed(self, seed: int) -> None:
            seeded.append(seed)

    class Model:
        num_timesteps = 123

    def evaluate_policy(
        model: object,
        env: object,
        **kwargs: object,
    ) -> tuple[list[float], list[int]]:
        calls.append({"model": model, "env": env, **kwargs})
        callback = kwargs["callback"]
        assert callable(callback)
        for episode_ordinal in range(2):
            terminal_ticks = [10, 14][episode_ordinal]
            callback(
                {
                    "done": True,
                    "info": {
                        "outcome": "win",
                        "native_outcome": "win",
                        "score": 2000,
                        "ticks": terminal_ticks,
                    },
                    "is_monitor_wrapped": False,
                    "i": 0,
                    "episode_counts": [episode_ordinal],
                },
                {},
            )
        return [1.0, 3.0], [10, 14]

    output = tmp_path / "pretrain_evaluation.json"
    payload = train._run_boundary_evaluation(
        phase="pretrain",
        output_path=output,
        model=Model(),
        eval_env=EvalEnv(),
        evaluate_policy=evaluate_policy,
        episode_count=2,
        seed=10_042,
    )

    assert seeded == [10_042]
    assert calls[0]["deterministic"] is True
    assert calls[0]["return_episode_rewards"] is True
    assert payload["model_num_timesteps"] == 123
    assert payload["mean_reward"] == 2.0
    assert payload["reward_std"] == 1.0
    assert payload["mean_episode_length"] == 12.0
    assert payload["episode_identities"] == [
        {
            "pairing_key": "env-0:episode-0",
            "vector_env_index": 0,
            "episode_ordinal_within_env": 0,
            "initial_env_seed": 10_042,
            "terminal_outcome": "win",
            "native_outcome": "win",
            "terminal_score": 2000,
            "terminal_ticks": 10,
            "time_limit_truncated": False,
        },
        {
            "pairing_key": "env-0:episode-1",
            "vector_env_index": 0,
            "episode_ordinal_within_env": 1,
            "initial_env_seed": 10_042,
            "terminal_outcome": "win",
            "native_outcome": "win",
            "terminal_score": 2000,
            "terminal_ticks": 14,
            "time_limit_truncated": False,
        },
    ]
    assert json.loads(output.read_text(encoding="utf-8")) == payload


def test_stochastic_boundary_evaluation_is_reproducible_and_rng_isolated(
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")
    sampled: list[float] = []

    class EvalEnv:
        num_envs = 4

        def seed(self, seed: int) -> None:
            self.last_seed = seed

    class Model:
        num_timesteps = 456

    def evaluate_policy(
        model: object,
        env: object,
        **kwargs: object,
    ) -> tuple[list[float], list[int]]:
        value = float(torch.rand(()))
        sampled.append(value)
        callback = kwargs["callback"]
        assert callable(callback)
        for vector_env_index in (2, 3):
            callback(
                {
                    "done": True,
                    "info": {
                        "outcome": "win",
                        "native_outcome": "win",
                        "score": 2000,
                        "ticks": 20,
                    },
                    "is_monitor_wrapped": False,
                    "i": vector_env_index,
                    "episode_counts": [0, 0, 0, 0],
                },
                {},
            )
        return [value, value], [20, 20]

    torch.manual_seed(1234)
    before = torch.get_rng_state().clone()
    payloads = []
    for index in range(2):
        payloads.append(
            train._run_boundary_evaluation(
                phase="pretrain",
                output_path=tmp_path / f"stochastic-{index}.json",
                model=Model(),
                eval_env=EvalEnv(),
                evaluate_policy=evaluate_policy,
                episode_count=2,
                seed=10_042,
                deterministic=False,
                action_sampling_seed=20_042,
            )
        )
        assert torch.equal(torch.get_rng_state(), before)

    assert sampled[0] == sampled[1]
    assert payloads[0]["deterministic"] is False
    assert payloads[0]["action_sampling_rng_isolated"] is True
    assert payloads[0]["vector_env_count"] == 4


def test_boundary_evaluation_preserves_identity_across_completion_order(
    tmp_path: Path,
) -> None:
    class EvalEnv:
        num_envs = 2

        def seed(self, seed: int) -> None:
            self.last_seed = seed

    class Model:
        num_timesteps = 789

    completion_order = [(1, 0), (0, 0), (1, 1), (0, 1)]

    def evaluate_policy(
        _model: object,
        _env: object,
        **kwargs: object,
    ) -> tuple[list[float], list[int]]:
        callback = kwargs["callback"]
        assert callable(callback)
        for result_index, (vector_env_index, episode_ordinal) in enumerate(
            completion_order
        ):
            counts = [0, 0]
            counts[vector_env_index] = episode_ordinal
            callback(
                {
                    "done": True,
                    "info": {
                        "episode": {},
                        "outcome": "win",
                        "native_outcome": "win",
                        "score": 2000 + result_index,
                        "ticks": [101, 202, 303, 404][result_index],
                    },
                    "is_monitor_wrapped": True,
                    "i": vector_env_index,
                    "episode_counts": counts,
                },
                {},
            )
        return [11.0, 22.0, 33.0, 44.0], [101, 202, 303, 404]

    payload = train._run_boundary_evaluation(
        phase="posttrain",
        output_path=tmp_path / "reordered.json",
        model=Model(),
        eval_env=EvalEnv(),
        evaluate_policy=evaluate_policy,
        episode_count=4,
        seed=50_000,
    )

    assert payload["version"] == 4
    assert payload["episode_rewards"] == [11.0, 22.0, 33.0, 44.0]
    assert [
        identity["pairing_key"]
        for identity in payload["episode_identities"]
    ] == [
        "env-1:episode-0",
        "env-0:episode-0",
        "env-1:episode-1",
        "env-0:episode-1",
    ]
    assert [
        identity["initial_env_seed"]
        for identity in payload["episode_identities"]
    ] == [50_001, 50_000, 50_001, 50_000]


def test_aim_path_freeze_is_bitwise_and_clears_loaded_momentum(
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")

    class Policy(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.features_extractor = torch.nn.Linear(4, 4)
            self.action_net = torch.nn.Linear(4, 5)
            self.optimizer = torch.optim.Adam(self.parameters(), lr=0.01)

    class ActionSpace:
        nvec = [2, 3]

    class Model:
        def __init__(self) -> None:
            self.policy = Policy()
            self.action_space = ActionSpace()

    model = Model()
    inputs = torch.ones((3, 4))
    warm_loss = model.policy.action_net(
        model.policy.features_extractor(inputs)
    ).sum()
    warm_loss.backward()
    model.policy.optimizer.step()
    model.policy.optimizer.zero_grad(set_to_none=True)

    verb_weight_before = model.policy.action_net.weight[:2].detach().clone()
    aim_weight_before = model.policy.action_net.weight[2:].detach().clone()
    aim_bias_before = model.policy.action_net.bias[2:].detach().clone()
    feature_before = {
        name: parameter.detach().clone()
        for name, parameter in (
            model.policy.features_extractor.named_parameters()
        )
    }

    freeze_state = train._install_aim_path_freeze(
        model=model,
        output_path=tmp_path / "aim_path_freeze.json",
    )

    assert all(
        not parameter.requires_grad
        for parameter in model.policy.features_extractor.parameters()
    )
    for parameter in (
        model.policy.action_net.weight,
        model.policy.action_net.bias,
    ):
        state = model.policy.optimizer.state[parameter]
        assert torch.count_nonzero(state["exp_avg"][2:]) == 0
        assert torch.count_nonzero(state["exp_avg_sq"][2:]) == 0

    loss = model.policy.action_net(
        model.policy.features_extractor(inputs)
    ).sum()
    loss.backward()
    assert torch.count_nonzero(
        model.policy.action_net.weight.grad[2:]
    ) == 0
    assert torch.count_nonzero(model.policy.action_net.bias.grad[2:]) == 0
    model.policy.optimizer.step()

    assert not torch.equal(
        verb_weight_before, model.policy.action_net.weight[:2]
    )
    assert torch.equal(aim_weight_before, model.policy.action_net.weight[2:])
    assert torch.equal(aim_bias_before, model.policy.action_net.bias[2:])
    for name, parameter in model.policy.features_extractor.named_parameters():
        assert torch.equal(feature_before[name], parameter)

    verification = train._verify_aim_path_freeze(
        model=model,
        freeze_state=freeze_state,
        output_path=tmp_path / "aim_path_freeze_verification.json",
    )
    assert verification["status"] == "PASS"
    assert verification["bitwise_unchanged"] is True


def test_optimizer_state_reset_preserves_every_parameter(
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")

    class Policy(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layer = torch.nn.Linear(3, 2)
            self.optimizer = torch.optim.Adam(self.parameters(), lr=0.01)

    class Model:
        def __init__(self) -> None:
            self.policy = Policy()

    model = Model()
    loss = model.policy.layer(torch.ones((2, 3))).sum()
    loss.backward()
    model.policy.optimizer.step()
    model.policy.optimizer.zero_grad(set_to_none=True)
    before = {
        name: parameter.detach().clone()
        for name, parameter in model.policy.named_parameters()
    }
    assert len(model.policy.optimizer.state) == 2

    receipt = train._reset_loaded_optimizer_state(
        model=model,
        output_path=tmp_path / "optimizer_state_reset.json",
    )

    assert receipt["status"] == "PASS"
    assert receipt["state_before"]["parameter_count"] == 2
    assert receipt["state_before"]["max_step"] == 1.0
    assert receipt["state_after"]["parameter_count"] == 0
    assert receipt["parameters"]["bitwise_unchanged"] is True
    assert len(model.policy.optimizer.state) == 0
    for name, parameter in model.policy.named_parameters():
        assert torch.equal(before[name], parameter)


def test_masked_distribution_forward_kl_sums_action_factors() -> None:
    torch = pytest.importorskip("torch")

    class MultiDistribution:
        def __init__(self, probabilities: list[list[float]]) -> None:
            self.distributions = [
                torch.distributions.Categorical(
                    probs=torch.tensor([probability], dtype=torch.float64)
                )
                for probability in probabilities
            ]

    teacher = MultiDistribution([[0.75, 0.25], [1.0, 0.0, 0.0]])
    current = MultiDistribution([[0.5, 0.5], [1.0, 0.0, 0.0]])

    actual = masked_distribution_forward_kl(teacher, current)
    expected = torch.distributions.kl_divergence(
        teacher.distributions[0],
        current.distributions[0],
    )

    assert actual.shape == (1,)
    assert torch.isfinite(actual).all()
    assert torch.allclose(actual, expected)


def test_teacher_policy_anchor_is_immutable_and_exercised(
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")

    class Policy(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layer = torch.nn.Linear(3, 2)
            self.optimizer = torch.optim.Adam(self.parameters(), lr=0.01)

        def set_training_mode(self, mode: bool) -> None:
            self.train(mode)

    class Model:
        _supports_teacher_policy_anchor = True

        def __init__(self) -> None:
            self.policy = Policy()

    model = Model()
    source = tmp_path / "teacher.zip"
    source.write_bytes(b"teacher-model")
    state = train._install_teacher_policy_anchor(
        model=model,
        coefficient=2.0,
        source_model_path=source,
        output_path=tmp_path / "teacher_policy_anchor.json",
    )
    installed = json.loads(
        (tmp_path / "teacher_policy_anchor.json").read_text(
            encoding="utf-8"
        )
    )
    assert installed["status"] == "INSTALLED"
    assert installed["initial_policy"]["bitwise_equal"] is True
    assert model._teacher_policy is not model.policy
    assert all(
        not parameter.requires_grad
        for parameter in model._teacher_policy.parameters()
    )

    with torch.no_grad():
        model.policy.layer.weight.add_(1.0)
    model._teacher_anchor_sample_count = 8
    model._teacher_anchor_kl_sum = 0.4
    model._teacher_anchor_kl_max = 0.2
    model._teacher_anchor_minibatch_count = 2
    model._teacher_anchor_train_calls = 1

    verified = train._verify_teacher_policy_anchor(
        model=model,
        anchor_state=state,
        output_path=(
            tmp_path / "teacher_policy_anchor_verification.json"
        ),
    )

    assert verified["status"] == "PASS"
    assert verified["teacher_policy"]["bitwise_unchanged"] is True
    assert verified["current_policy"]["bitwise_changed"] is True
    assert verified["training"]["mean_forward_kl"] == pytest.approx(0.05)
    assert model._teacher_policy is None


def test_force_close_vector_env_is_bounded_after_worker_failure() -> None:
    calls: list[str] = []

    class Remote:
        def close(self) -> None:
            calls.append("remote.close")

    class Process:
        alive = True

        def is_alive(self) -> bool:
            return self.alive

        def terminate(self) -> None:
            calls.append("process.terminate")

        def join(self, *, timeout: float) -> None:
            calls.append(f"process.join:{timeout}")

        def kill(self) -> None:
            calls.append("process.kill")
            self.alive = False

    class VecEnv:
        remotes = (Remote(),)
        processes = (Process(),)
        waiting = True
        closed = False

    env = VecEnv()
    train._force_close_vector_env(env)

    assert calls == [
        "remote.close",
        "process.terminate",
        "process.join:5.0",
        "process.kill",
        "process.join:2.0",
    ]
    assert env.waiting is False
    assert env.closed is True


def test_training_failure_receipt_is_machine_readable(tmp_path: Path) -> None:
    output = tmp_path / "failure.json"
    payload = train._write_training_failure_receipt(
        output_path=output,
        error=EOFError("worker pipe closed"),
        model_num_timesteps=40_960,
        emergency_model_saved=True,
    )

    assert payload["status"] == "ABORTED"
    assert payload["exception_type"] == "EOFError"
    assert payload["model_num_timesteps"] == 40_960
    assert payload["emergency_model_saved"] is True
    assert json.loads(output.read_text(encoding="utf-8")) == payload


def test_training_completion_receipt_is_canonical_and_fail_closed(
    tmp_path: Path,
) -> None:
    (tmp_path / "config.json").write_text("{}\n", encoding="utf-8")
    (tmp_path / "final_model.zip").write_bytes(b"model")
    output = tmp_path / "completion.json"

    payload = train._write_training_completion_receipt(
        output_path=output,
        run_dir=tmp_path,
        model_num_timesteps=4,
        expected_effective_steps=4,
        algorithm="ppo",
        boundary_evaluation_enabled=False,
        stochastic_boundary_evaluation_enabled=False,
    )

    assert payload["status"] == "COMPLETE"
    assert payload["process_contract"]["vector_env_close_status"] == "PASS"
    expected = (
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")
    assert output.read_bytes() == expected
    with pytest.raises(RuntimeError, match="already exists"):
        train._write_training_completion_receipt(
            output_path=output,
            run_dir=tmp_path,
            model_num_timesteps=4,
            expected_effective_steps=4,
            algorithm="ppo",
            boundary_evaluation_enabled=False,
            stochastic_boundary_evaluation_enabled=False,
        )

    with pytest.raises(RuntimeError, match="timestep mismatch"):
        train._write_training_completion_receipt(
            output_path=tmp_path / "mismatch.json",
            run_dir=tmp_path,
            model_num_timesteps=3,
            expected_effective_steps=4,
            algorithm="ppo",
            boundary_evaluation_enabled=False,
            stochastic_boundary_evaluation_enabled=False,
        )


def test_training_completion_requires_teacher_anchor_receipts(
    tmp_path: Path,
) -> None:
    (tmp_path / "config.json").write_text("{}\n", encoding="utf-8")
    (tmp_path / "final_model.zip").write_bytes(b"model")
    (tmp_path / "teacher_policy_anchor.json").write_text(
        "{}\n",
        encoding="utf-8",
    )
    (tmp_path / "teacher_policy_anchor_verification.json").write_text(
        "{}\n",
        encoding="utf-8",
    )

    payload = train._write_training_completion_receipt(
        output_path=tmp_path / "completion.json",
        run_dir=tmp_path,
        model_num_timesteps=4,
        expected_effective_steps=4,
        algorithm="teacher_anchored_maskable_ppo",
        boundary_evaluation_enabled=False,
        stochastic_boundary_evaluation_enabled=False,
        teacher_policy_anchor_enabled=True,
    )

    assert "teacher_policy_anchor" in payload["artifacts"]
    assert "teacher_policy_anchor_verification" in payload["artifacts"]
