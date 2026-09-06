"""Render one level's fastest verified paired blind attempt as MP4.

The recorder is bound to a completed paired multi-model aggregate and its
frozen model manifest.  It selects the fastest winning attempt for one level,
re-executes it once without rendering, records it a second time with
presentation-only rendering, and refuses to publish unless both trajectories
match the formal blind receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Mapping

if __package__ in {None, ""}:
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_PROJECT_ROOT))
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

import av
import numpy as np
from PIL import Image, ImageDraw

from tools.evaluate_zero_shot_multilevel import (
    _build_configs,
    _load_preregistration,
)
from tools.record_human_speedrun_blind_candidate import (
    IDENTITY_FIELDS,
    _assert_identity,
    _encode_frame,
    _font,
    _make_env,
    _run_episode,
    _verb_name,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _model_specs(manifest: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    if (
        manifest.get("schema") != "zuma-rl.zero-shot-models-manifest"
        or manifest.get("status") != "FROZEN"
    ):
        raise ValueError("unexpected or unfrozen models manifest")
    models = manifest.get("models")
    if not isinstance(models, list) or not models:
        raise ValueError("models manifest is empty")
    result: dict[str, dict[str, Any]] = {}
    for raw in models:
        if not isinstance(raw, dict):
            raise ValueError("models manifest contains a non-object entry")
        model = dict(raw)
        model_id = str(model.get("id", ""))
        if not model_id or model_id in result:
            raise ValueError(f"invalid or duplicate model id: {model_id!r}")
        result[model_id] = model
    return result


def _select_attempt(
    aggregate: Mapping[str, Any],
    *,
    level_id: str,
    model_id: str | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    matches: list[tuple[dict[str, Any], dict[str, Any]]] = []
    summaries = aggregate.get("models")
    if not isinstance(summaries, list):
        raise ValueError("aggregate has no model summaries")
    for raw_model_summary in summaries:
        if not isinstance(raw_model_summary, dict):
            raise ValueError("aggregate contains an invalid model summary")
        summary_model = raw_model_summary.get("model")
        if not isinstance(summary_model, dict):
            raise ValueError("aggregate model identity is missing")
        summary_model_id = str(summary_model.get("id", ""))
        if model_id is not None and summary_model_id != model_id:
            continue
        level_summaries = raw_model_summary.get("level_summaries")
        if not isinstance(level_summaries, list):
            raise ValueError("aggregate model has no level summaries")
        for raw_level_summary in level_summaries:
            if not isinstance(raw_level_summary, dict):
                raise ValueError("aggregate contains an invalid level summary")
            if str(raw_level_summary.get("level_id", "")).casefold() != level_id.casefold():
                continue
            attempt = raw_level_summary.get("best_attempt")
            if not isinstance(attempt, dict) or attempt.get("outcome") != "win":
                continue
            if str(attempt.get("model_id", "")) != summary_model_id:
                raise ValueError("attempt model identity differs from its summary")
            matches.append((dict(attempt), dict(summary_model)))
    if not matches:
        qualifier = f" for model {model_id!r}" if model_id is not None else ""
        raise ValueError(f"no winning attempt for {level_id!r}{qualifier}")
    return min(matches, key=lambda pair: (int(pair[0]["ticks"]), pair[0]["model_id"]))


def _annotate(
    frame: np.ndarray[Any, Any],
    *,
    level_id: str,
    display_name: str,
    seed: int,
    model_id: str,
    info: Mapping[str, Any] | None,
    desired_action: np.ndarray[Any, Any] | None,
) -> np.ndarray[Any, Any]:
    game = Image.fromarray(frame, mode="RGB").resize(
        (frame.shape[1] * 2, frame.shape[0] * 2),
        resample=Image.Resampling.BILINEAR,
    )
    header_height = 136
    image = Image.new("RGB", (game.width, game.height + header_height), (5, 8, 13))
    image.paste(game, (0, header_height))
    draw = ImageDraw.Draw(image)
    title_font = _font(20)
    body_font = _font(16)
    draw.text(
        (12, 7),
        "AlphaZuma V1 | verified blind replay | structured state",
        font=title_font,
        fill=(240, 244, 250),
    )
    draw.text(
        (12, 34),
        f"level={level_id} / {display_name}  seed={seed}",
        font=body_font,
        fill=(167, 222, 255),
    )
    draw.text(
        (12, 55),
        f"model={model_id}  reaction=120 ms  native clock=100 Hz",
        font=body_font,
        fill=(188, 199, 218),
    )
    draw.text(
        (12, 76),
        "simulator render; no pixels enter policy; not original-client RTA footage",
        font=body_font,
        fill=(255, 184, 133),
    )
    if info is None or desired_action is None:
        status = "time= 0.00 s  tick=0000  score=0000  balls=---  outcome=-"
        action = "intent=RESET             executed=WAIT"
    else:
        human = info["human_speedrun"]
        desired_verb = int(desired_action[0])
        desired_aim = int(desired_action[1])
        executed_verb = int(human["executed_verb"])
        executed_aim = int(human["executed_aim_bin"])
        ticks = int(info["ticks"])
        status = (
            f"time={ticks / 100.0:5.2f} s  tick={ticks:04d}  "
            f"score={int(info['score']):04d}  balls={int(info['chain_length']):03d}  "
            f"outcome={info.get('outcome') or '-'}"
        )
        action = (
            f"intent={_verb_name(desired_verb):4s}@{desired_aim:03d}  "
            f"executed={_verb_name(executed_verb):4s}@{executed_aim:03d}  "
            f"aim={float(human['aim_angle_degrees']):6.1f} deg"
        )
    draw.text((12, 98), status, font=body_font, fill=(206, 236, 210))
    draw.text((12, 118), action, font=body_font, fill=(255, 218, 133))
    return np.ascontiguousarray(np.asarray(image, dtype=np.uint8))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregate", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--models-manifest", type=Path, required=True)
    parser.add_argument("--original-root", type=Path, required=True)
    parser.add_argument("--level-id", required=True)
    parser.add_argument("--model-id")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--fps", type=int, default=50)
    parser.add_argument("--frame-stride", type=int, default=2)
    parser.add_argument("--final-hold-seconds", type=float, default=2.0)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.fps < 1 or args.frame_stride < 1:
        raise SystemExit("fps and frame-stride must be positive")
    if not math.isfinite(args.final_hold_seconds) or args.final_hold_seconds < 0:
        raise SystemExit("final-hold-seconds must be finite and non-negative")

    aggregate_path = args.aggregate.expanduser().resolve(strict=True)
    prereg_path = args.preregistration.expanduser().resolve(strict=True)
    manifest_path = args.models_manifest.expanduser().resolve(strict=True)
    original_root = args.original_root.expanduser().resolve(strict=True)
    output_path = args.output.expanduser().resolve()
    metadata_path = output_path.with_suffix(".json")
    poster_path = output_path.with_name(output_path.stem + "-poster.png")
    partial_path = output_path.with_name(output_path.stem + ".partial.mp4")
    for path in (output_path, metadata_path, poster_path, partial_path):
        if path.exists():
            raise SystemExit(f"refusing to overwrite existing output: {path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    aggregate = _read_json(aggregate_path)
    prereg = _load_preregistration(prereg_path)
    manifest = _read_json(manifest_path)
    models = _model_specs(manifest)
    if (
        aggregate.get("schema") != "zuma-rl.paired-multimodel-evaluation-aggregate"
        or aggregate.get("status") != "COMPLETE"
    ):
        raise ValueError("unexpected or incomplete paired aggregate")
    validation = aggregate.get("validation")
    if not isinstance(validation, dict) or not validation or not all(validation.values()):
        raise ValueError("paired aggregate did not pass every integrity check")
    if aggregate.get("preregistration", {}).get("sha256") != _sha256(prereg_path):
        raise ValueError("aggregate is not bound to this preregistration")
    if aggregate.get("models_manifest", {}).get("sha256") != _sha256(manifest_path):
        raise ValueError("aggregate is not bound to this models manifest")

    selected, aggregate_model = _select_attempt(
        aggregate,
        level_id=args.level_id,
        model_id=args.model_id,
    )
    level_id = str(selected["level_id"])
    display_name = str(selected["display_name"])
    if not any(str(row["id"]).casefold() == level_id.casefold() for row in prereg["levels"]):
        raise ValueError("selected level is absent from preregistration")
    model_id = str(selected["model_id"])
    model_spec = models.get(model_id)
    if model_spec is None:
        raise ValueError("selected model is absent from frozen manifest")
    for key in ("id", "training_steps", "sha256"):
        if aggregate_model.get(key) != model_spec.get(key):
            raise ValueError(f"aggregate and manifest differ for model field {key}")
    model_path = Path(str(model_spec["path"])).resolve(strict=True)
    model_sha = _sha256(model_path)
    if model_sha != model_spec["sha256"] or model_sha != selected["model_sha256"]:
        raise ValueError("selected model bytes differ from frozen receipts")

    import torch
    from sb3_contrib import MaskablePPO

    torch.set_num_threads(1)
    model = MaskablePPO.load(model_path, device=args.device)
    base_config, input_config, reward_config = _build_configs(prereg)
    common_env_args = {
        "original_root": original_root,
        "level": level_id,
        "profile_mode": str(prereg["environment"]["profile_mode"]),
        "base_config": base_config,
        "input_config": input_config,
        "reward_config": reward_config,
    }
    seed = int(selected["seed"])

    verification_env = _make_env(**common_env_args, render=False)
    try:
        if verification_env.observation_space != model.observation_space:
            raise ValueError("model observation space differs from replay environment")
        if verification_env.action_space != model.action_space:
            raise ValueError("model action space differs from replay environment")
        verified = _run_episode(
            model=model,
            model_spec=model_spec,
            seed=seed,
            threshold_ticks=0,
            env=verification_env,
            frame_stride=args.frame_stride,
        )
    finally:
        verification_env.close()
    _assert_identity(verified, selected, phase="pre-render")

    frame_count = 0
    gameplay_frame_count = 0
    last_pixels: np.ndarray[Any, Any] | None = None
    container = av.open(str(partial_path), mode="w")
    stream = container.add_stream("libx264", rate=args.fps)
    stream.width = 800
    stream.height = 736
    stream.pix_fmt = "yuv420p"
    stream.options = {"crf": "18", "preset": "medium"}

    def on_frame(
        pixels: np.ndarray[Any, Any],
        info: Mapping[str, Any] | None,
        desired_action: np.ndarray[Any, Any] | None,
    ) -> None:
        nonlocal frame_count, gameplay_frame_count, last_pixels
        last_pixels = _annotate(
            pixels,
            level_id=level_id,
            display_name=display_name,
            seed=seed,
            model_id=model_id,
            info=info,
            desired_action=desired_action,
        )
        if last_pixels.shape != (736, 800, 3):
            raise RuntimeError(f"unexpected annotated frame shape: {last_pixels.shape}")
        _encode_frame(container, stream, last_pixels)
        frame_count += 1
        gameplay_frame_count += 1

    render_env = _make_env(**common_env_args, render=True)
    try:
        rendered = _run_episode(
            model=model,
            model_spec=model_spec,
            seed=seed,
            threshold_ticks=0,
            env=render_env,
            frame_stride=args.frame_stride,
            on_frame=on_frame,
        )
        _assert_identity(rendered, selected, phase="rendered")
        if last_pixels is None:
            raise RuntimeError("recording produced no frames")
        Image.fromarray(last_pixels, mode="RGB").save(poster_path)
        hold_frames = int(round(args.final_hold_seconds * args.fps))
        for _ in range(hold_frames):
            _encode_frame(container, stream, last_pixels)
            frame_count += 1
        for packet in stream.encode():
            container.mux(packet)
    except Exception:
        container.close()
        for path in (partial_path, poster_path):
            if path.exists():
                path.unlink()
        raise
    else:
        container.close()
    finally:
        render_env.close()

    os.replace(partial_path, output_path)
    decoded_frames = 0
    decoded_width = decoded_height = 0
    with av.open(str(output_path), mode="r") as decoded:
        video_stream = decoded.streams.video[0]
        decoded_width = int(video_stream.codec_context.width)
        decoded_height = int(video_stream.codec_context.height)
        for frame in decoded.decode(video=0):
            decoded_frames += 1
            if frame.width != decoded_width or frame.height != decoded_height:
                raise RuntimeError("decoded video dimensions changed")
    if decoded_frames != frame_count:
        raise RuntimeError(f"encoded frame count mismatch: {decoded_frames} != {frame_count}")

    metadata = {
        "schema": "zuma-rl.paired-multimodel-level-video",
        "version": 1,
        "status": "COMPLETE",
        "source": {
            "aggregate": str(aggregate_path),
            "aggregate_sha256": _sha256(aggregate_path),
            "preregistration": str(prereg_path),
            "preregistration_sha256": _sha256(prereg_path),
            "models_manifest": str(manifest_path),
            "models_manifest_sha256": _sha256(manifest_path),
        },
        "selection": {
            "rule": "fastest winning formal blind attempt for level across frozen models",
            "requested_level_id": args.level_id,
            "requested_model_id": args.model_id,
        },
        "candidate": selected,
        "model": {"path": str(model_path), "sha256": model_sha, "device": args.device},
        "verification": {
            "identity_fields": list(IDENTITY_FIELDS),
            "pre_render_matches_formal_attempt": True,
            "rendered_matches_formal_attempt": True,
            "pre_render": verified,
            "rendered": rendered,
        },
        "video": {
            "path": str(output_path),
            "sha256": _sha256(output_path),
            "bytes": output_path.stat().st_size,
            "codec": "h264/libx264",
            "pixel_format": "yuv420p",
            "width": decoded_width,
            "height": decoded_height,
            "fps": int(args.fps),
            "frame_stride_native_ticks": int(args.frame_stride),
            "gameplay_frame_count": gameplay_frame_count,
            "frame_count": frame_count,
            "decoded_frame_count": decoded_frames,
            "duration_seconds": frame_count / float(args.fps),
            "final_hold_seconds": float(args.final_hold_seconds),
            "poster": str(poster_path),
            "poster_sha256": _sha256(poster_path),
        },
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
