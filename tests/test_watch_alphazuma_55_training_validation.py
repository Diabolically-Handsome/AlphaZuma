from __future__ import annotations

from tools import watch_alphazuma_55_training_validation as subject


def test_process_command_returns_none_for_absent_pid() -> None:
    assert subject._process_command(2_147_483_647) is None


def test_sha256_uses_prefixed_lowercase_digest(tmp_path) -> None:
    path = tmp_path / "artifact.bin"
    path.write_bytes(b"alphazuma")

    digest = subject._sha256(path)

    assert digest == "sha256:3f3b0ac8b8b09622bbc8c222d46b977a68e9d51f214a5ff556835a480ab20300"


def test_expected_matrix_attempts_multiplies_by_model_count() -> None:
    preregistration = {"total_attempts": 55}
    manifest = {"models": [{"id": f"model-{index}"} for index in range(8)]}

    assert subject._expected_matrix_attempts(preregistration, manifest) == 440
