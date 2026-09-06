"""Render the selected certified human-speedrun blind candidate as MP4.

The recorder is deliberately bound to a completed blind-evaluation receipt.
It verifies the selected trajectory once without rendering, records it a second
time with presentation-only rendering enabled, and refuses to publish the MP4
unless both trajectories match the frozen blind attempt.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping
from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

if __package__ in {None, ""}:
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_PROJECT_ROOT))
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from tools.evaluate_human_speedrun_blind import (
    EpisodeTracker,
    _build_configs,
    _load_and_validate_preregistration,
)
from zuma_rl.human_speedrun import HumanSpeedrunWrapper
from zuma_rl.revenge_env import RevengeEnv, RevengeEnvConfig


IDENTITY_FIELDS = (
    "outcome",
    "ticks",
    "score",
    "trajectory_sha256",
    "decisions",
    "desired_action_counts",
    "executed_action_counts",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = (
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"),
        Path("/mnt/c/Windows/Fonts/consola.ttf"),
    )
    for candidate in candidates:
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default(size=size)


def _verb_name(value: int) -> str:
    return ("WAIT", "FIRE", "SWAP")[value]


def _annotate(
    frame: np.ndarray[Any, Any],
    *,
    seed: int,
    model_id: str,
    info: Mapping[str, Any] | None,
    desired_action: np.ndarray[Any, Any] | None,
) -> np.ndarray[Any, Any]:
    game = Image.fromarray(frame, mode="RGB").resize(
        (frame.shape[1] * 2, frame.shape[0] * 2),
        resample=Image.Resampling.BILINEAR,
    )
    header_height = 96
    image = Image.new("RGB", (game.width, game.height + header_height), (5, 8, 13))
    image.paste(game, (0, header_height))
    draw = ImageDraw.Draw(image)
    title_font = _font(20)
    body_font = _font(16)
    draw.text(
        (12, 7),
        "AlphaZuma V1 | certified blind replay | human-limited controls",
        font=title_font,
        fill=(240, 244, 250),
    )
    draw.text(
        (12, 34),
        f"model={model_id}  seed={seed}  reaction=120 ms  record gate=<12.00 s",
        font=body_font,
        fill=(167, 222, 255),
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
    draw.text((12, 57), status, font=body_font, fill=(206, 236, 210))
    draw.text((12, 77), action, font=body_font, fill=(255, 218, 133))
    return np.ascontiguousarray(np.asarray(image, dtype=np.uint8))


def _encode_frame(
    container: av.container.OutputContainer,
    stream: Any,
    pixels: np.ndarray[Any, Any],
) -> None:
    video_frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
    for packet in stream.encode(video_frame):
        container.mux(packet)


def _make_env(
    *,
    original_root: Path,
    level: str,
    profile_mode: str,
    base_config: RevengeEnvConfig,
    input_config: Any,
    reward_config: Any,
    render: bool,
) -> HumanSpeedrunWrapper:
    config = base_config
    if render:
        # Rendering dimensions are absent from the policy observation and do
        # not alter simulation state.  The post-render trajectory check below
        # independently enforces that claim for the recorded episode.
        config = replace(config, render_width=400, render_height=300)
    base = RevengeEnv(
        config=config,
        level_id=level,
        root=original_root,
        hard=False,
        curve_index=0,
        profile_mode=profile_mode,
        render_mode="rgb_array" if render else None,
    )
    return HumanSpeedrunWrapper(
        base,
        input_config=input_config,
        reward_config=reward_config,
    )


def _run_episode(
    *,
    model: Any,
    model_spec: dict[str, Any],
    seed: int,
    threshold_ticks: int,
    env: HumanSpeedrunWrapper,
    frame_stride: int,
    on_frame: Callable[[np.ndarray[Any, Any], Mapping[str, Any] | None, np.ndarray[Any, Any] | None], None]
    | None = None,
) -> dict[str, Any]:
    observation, _ = env.reset(seed=seed)
    tracker = EpisodeTracker(seed=seed)
    if on_frame is not None:
        initial = env.render()
        if not isinstance(initial, np.ndarray):
            raise RuntimeError("render did not return an RGB array")
        on_frame(initial, None, None)

    final_info: dict[str, Any] | None = None
    for _ in range(int(env.revenge_env.config.max_ticks) + 1):
        action, _ = model.predict(
            observation,
            deterministic=True,
            action_masks=env.action_masks(),
        )
        action_array = np.asarray(action, dtype=np.int64).reshape(2)
        observation, reward, terminated, truncated, info = env.step(action_array)
        tracker.update(action_array, float(reward), info)
        final_info = dict(info)
        if on_frame is not None and (
            tracker.decisions % frame_stride == 0 or terminated or truncated
        ):
            rendered = env.render()
            if not isinstance(rendered, np.ndarray):
                raise RuntimeError("render did not return an RGB array")
            on_frame(rendered, info, action_array)
        if terminated or truncated:
            break
    else:
        raise RuntimeError("episode exceeded the configured max tick guard")

    if final_info is None:
        raise RuntimeError("episode ended without terminal info")
    return tracker.finish(
        info=final_info,
        model=model_spec,
        threshold_ticks=threshold_ticks,
    )


def _assert_identity(
    actual: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    phase: str,
) -> None:
    differences = {
        field: {"expected": expected.get(field), "actual": actual.get(field)}
        for field in IDENTITY_FIELDS
        if actual.get(field) != expected.get(field)
    }
    if differences:
        raise RuntimeError(
            f"{phase} trajectory differs from certified blind attempt: "
            + json.dumps(differences, ensure_ascii=False, sort_keys=True)
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blind-evaluation", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--original-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device")
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

    blind_path = args.blind_evaluation.expanduser().resolve(strict=True)
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

    blind = _read_json(blind_path)
    prereg = _load_and_validate_preregistration(prereg_path)
    prereg_sha = _sha256(prereg_path)
    if blind.get("status") != "COMPLETE" or blind.get("decision", {}).get("status") != "PASS":
        raise ValueError("blind evaluation is not a completed PASS")
    if blind.get("preregistration", {}).get("sha256") != prereg_sha:
        raise ValueError("blind evaluation is not bound to this preregistration")
    selected = blind.get("selected_candidate")
    if not isinstance(selected, dict):
        raise ValueError("blind evaluation has no selected candidate")
    if len(blind.get("replays", [])) != 3 or not all(
        replay.get("identical_to_blind_attempt") is True
        for replay in blind["replays"]
    ):
        raise ValueError("blind candidate lacks three identical replay receipts")

    model_spec = next(
        (
            model
            for model in prereg["models"]
            if model["id"] == selected["model_id"]
        ),
        None,
    )
    if model_spec is None:
        raise ValueError("selected model is absent from preregistration")
    model_path = Path(model_spec["path"]).resolve(strict=True)
    if _sha256(model_path) != selected["model_sha256"]:
        raise ValueError("selected model bytes differ from blind receipt")

    import torch
    from sb3_contrib import MaskablePPO

    torch.set_num_threads(1)
    device = args.device or str(blind["device"])
    model = MaskablePPO.load(model_path, device=device)
    base_config, input_config, reward_config = _build_configs(prereg)
    environment = prereg["environment"]
    common_env_args = {
        "original_root": original_root,
        "level": str(environment["level"]),
        "profile_mode": str(environment["profile_mode"]),
        "base_config": base_config,
        "input_config": input_config,
        "reward_config": reward_config,
    }
    threshold_ticks = int(prereg["human_record"]["strict_record_threshold_ticks"])
    seed = int(selected["seed"])

    verification_env = _make_env(**common_env_args, render=False)
    try:
        verified = _run_episode(
            model=model,
            model_spec=model_spec,
            seed=seed,
            threshold_ticks=threshold_ticks,
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
    stream.height = 696
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
            seed=seed,
            model_id=str(model_spec["id"]),
            info=info,
            desired_action=desired_action,
        )
        if last_pixels.shape != (696, 800, 3):
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
            threshold_ticks=threshold_ticks,
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
        if partial_path.exists():
            partial_path.unlink()
        if poster_path.exists():
            poster_path.unlink()
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
        "schema": "zuma-rl.certified-human-speedrun-blind-video",
        "version": 1,
        "status": "COMPLETE",
        "source": {
            "blind_evaluation": str(blind_path),
            "blind_evaluation_sha256": _sha256(blind_path),
            "preregistration": str(prereg_path),
            "preregistration_sha256": prereg_sha,
        },
        "candidate": selected,
        "model": {
            "path": str(model_path),
            "sha256": _sha256(model_path),
            "device": device,
        },
        "verification": {
            "identity_fields": list(IDENTITY_FIELDS),
            "pre_render_matches_blind_attempt": True,
            "rendered_matches_blind_attempt": True,
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
