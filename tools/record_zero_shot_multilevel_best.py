"""Replay and record the fastest verified multilevel zero-shot win."""

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

from tools.evaluate_zero_shot_multilevel import _build_configs, _load_preregistration
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
    header_height = 116
    image = Image.new("RGB", (game.width, game.height + header_height), (5, 8, 13))
    image.paste(game, (0, header_height))
    draw = ImageDraw.Draw(image)
    title_font = _font(20)
    body_font = _font(16)
    draw.text(
        (12, 7),
        "AlphaZuma V1 | frozen zero-shot replay | human-limited controls",
        font=title_font,
        fill=(240, 244, 250),
    )
    draw.text(
        (12, 34),
        f"unseen={level_id} / {display_name}  seed={seed}",
        font=body_font,
        fill=(167, 222, 255),
    )
    draw.text(
        (12, 55),
        f"model={model_id}  reaction=120 ms  no level training or adaptation",
        font=body_font,
        fill=(188, 199, 218),
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
    draw.text((12, 78), status, font=body_font, fill=(206, 236, 210))
    draw.text((12, 98), action, font=body_font, fill=(255, 218, 133))
    return np.ascontiguousarray(np.asarray(image, dtype=np.uint8))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregate", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--original-root", type=Path, required=True)
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
    if aggregate.get("schema") != "zuma-rl.zero-shot-multilevel-aggregate":
        raise ValueError("unexpected aggregate schema")
    if aggregate.get("status") != "COMPLETE":
        raise ValueError("aggregate is not complete")
    validation = aggregate.get("validation")
    if not isinstance(validation, dict) or not validation or not all(validation.values()):
        raise ValueError("aggregate did not pass every integrity check")
    if aggregate.get("preregistration", {}).get("sha256") != _sha256(prereg_path):
        raise ValueError("aggregate is not bound to this preregistration")

    selected = aggregate.get("overall", {}).get("best_attempt")
    if not isinstance(selected, dict) or selected.get("outcome") != "win":
        raise ValueError("aggregate has no fastest winning attempt")
    level_id = str(selected["level_id"])
    display_name = str(selected["display_name"])
    if not any(str(row["id"]) == level_id for row in prereg["levels"]):
        raise ValueError("selected level is absent from preregistration")

    model_spec = prereg["model"]
    model_path = Path(str(model_spec["path"])).resolve(strict=True)
    model_sha = _sha256(model_path)
    if model_sha != model_spec["sha256"] or model_sha != selected["model_sha256"]:
        raise ValueError("selected model bytes differ from the frozen receipts")

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
    stream.height = 716
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
            model_id=str(model_spec["id"]),
            info=info,
            desired_action=desired_action,
        )
        if last_pixels.shape != (716, 800, 3):
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
        "schema": "zuma-rl.zero-shot-multilevel-best-video",
        "version": 1,
        "status": "COMPLETE",
        "source": {
            "aggregate": str(aggregate_path),
            "aggregate_sha256": _sha256(aggregate_path),
            "preregistration": str(prereg_path),
            "preregistration_sha256": _sha256(prereg_path),
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
