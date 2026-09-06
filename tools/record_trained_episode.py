"""Record one deterministic trained-policy episode as an annotated MP4."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from dataclasses import replace
from pathlib import Path
from typing import Any

import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from sb3_contrib import MaskablePPO

from zuma_rl.revenge_env import RevengeEnv, RevengeEnvConfig


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = (
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"),
        Path("/mnt/c/Windows/Fonts/consola.ttf"),
    )
    for candidate in candidates:
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default(size=size)


def _annotate(
    frame: np.ndarray[Any, Any],
    *,
    title: str,
    status: str,
    action: str,
) -> np.ndarray[Any, Any]:
    image = Image.fromarray(frame, mode="RGB").resize(
        (frame.shape[1] * 2, frame.shape[0] * 2),
        resample=Image.Resampling.BILINEAR,
    )
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, image.width, 78), fill=(5, 8, 13))
    font = _font(20)
    small_font = _font(17)
    draw.text((12, 7), title, font=font, fill=(240, 244, 250))
    draw.text((12, 34), status, font=small_font, fill=(167, 222, 255))
    draw.text((12, 56), action, font=small_font, fill=(255, 218, 133))
    return np.ascontiguousarray(np.asarray(image, dtype=np.uint8))


def _encode_frame(container: av.container.OutputContainer, stream: Any, pixels: np.ndarray[Any, Any]) -> None:
    video_frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
    for packet in stream.encode(video_frame):
        container.mux(packet)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--run-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--fps", type=int, default=50)
    parser.add_argument("--frame-stride", type=int, default=2)
    parser.add_argument("--final-hold-seconds", type=float, default=2.0)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    model_path = args.model.expanduser().resolve()
    config_path = args.run_config.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    metadata_path = output_path.with_suffix(".json")
    partial_path = output_path.with_name(output_path.stem + ".partial.mp4")

    if args.fps < 1 or args.frame_stride < 1:
        raise SystemExit("fps and frame-stride must be positive")
    if not math.isfinite(args.final_hold_seconds) or args.final_hold_seconds < 0:
        raise SystemExit("final-hold-seconds must be finite and non-negative")
    if not model_path.is_file() or not config_path.is_file():
        raise SystemExit("model or run config does not exist")
    for path in (output_path, metadata_path, partial_path):
        if path.exists():
            raise SystemExit(f"refusing to overwrite existing output: {path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    run_config = json.loads(config_path.read_text(encoding="utf-8"))
    environment = run_config["environment"]
    env_config = RevengeEnvConfig(**environment["config"])
    # Rendering resolution is presentation-only and is absent from the actor
    # observation. Keep the trained simulation and reward configuration exact.
    env_config = replace(env_config, render_width=400, render_height=300)
    env = RevengeEnv(
        config=env_config,
        level_id=environment["level"],
        root=environment["original_root"],
        hard=bool(environment["hard"]),
        curve_index=int(environment["requested_curve_index"]),
        profile_mode=environment["profile_mode"],
        render_mode="rgb_array",
    )
    model = MaskablePPO.load(model_path, device=args.device)
    if model.observation_space != env.observation_space:
        raise RuntimeError("model and recording environment observation spaces differ")
    if model.action_space != env.action_space:
        raise RuntimeError("model and recording environment action spaces differ")

    title = "PC Golden | Jungle2 | deterministic final_model"
    observation, reset_info = env.reset(seed=args.seed)
    action_counts = {"wait": 0, "fire": 0, "swap": 0}
    accepted_counts = {"wait": 0, "fire": 0, "swap": 0}
    total_reward = 0.0
    decision_count = 0
    frame_count = 0
    terminated = truncated = False
    final_info: dict[str, Any] = {}
    last_action_text = "action=RESET"

    container = av.open(str(partial_path), mode="w")
    stream = container.add_stream("libx264", rate=args.fps)
    stream.width = 800
    stream.height = 600
    stream.pix_fmt = "yuv420p"
    stream.options = {"crf": "18", "preset": "medium"}
    last_pixels: np.ndarray[Any, Any] | None = None
    try:
        initial = env.render()
        assert isinstance(initial, np.ndarray)
        last_pixels = _annotate(
            initial,
            title=title,
            status=(
                "tick=00000  score=00000  "
                f"balls={int(reset_info['chain_length']):03d}  outcome=-"
            ),
            action=last_action_text,
        )
        _encode_frame(container, stream, last_pixels)
        frame_count += 1

        while not (terminated or truncated):
            action, _ = model.predict(
                observation,
                deterministic=True,
                action_masks=env.action_masks(),
            )
            verb, aim_bin, angle = env.decode_action(action)
            verb_name = ("wait", "fire", "swap")[verb]
            action_counts[verb_name] += 1
            observation, reward, terminated, truncated, info = env.step(action)
            decision_count += 1
            total_reward += float(reward)
            final_info = dict(info)
            if bool(info.get("action_accepted", False)):
                accepted_counts[verb_name] += 1
            last_action_text = (
                f"action={verb_name.upper():4s}  aim_bin={aim_bin:03d}  "
                f"angle={math.degrees(angle):6.1f} deg"
            )
            if decision_count % args.frame_stride == 0 or terminated or truncated:
                rendered = env.render()
                assert isinstance(rendered, np.ndarray)
                outcome = info.get("outcome") or "-"
                status = (
                    f"tick={int(info['ticks']):05d}  score={int(info['score']):05d}  "
                    f"balls={int(info['chain_length']):03d}  outcome={outcome}"
                )
                last_pixels = _annotate(
                    rendered,
                    title=title,
                    status=status,
                    action=last_action_text,
                )
                _encode_frame(container, stream, last_pixels)
                frame_count += 1

        assert last_pixels is not None
        hold_frames = int(round(args.final_hold_seconds * args.fps))
        for _ in range(hold_frames):
            _encode_frame(container, stream, last_pixels)
            frame_count += 1
        for packet in stream.encode():
            container.mux(packet)
    finally:
        container.close()
        env.close()

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
        raise RuntimeError(
            f"encoded frame count mismatch: {decoded_frames} != {frame_count}"
        )

    outcome = final_info.get("outcome")
    if outcome is None and truncated:
        outcome = "truncated"
    metadata = {
        "schema": "zuma-rl.trained-policy-episode-video",
        "version": 1,
        "status": "COMPLETE",
        "policy": {
            "algorithm": "MaskablePPO",
            "deterministic": True,
            "model": str(model_path),
            "model_sha256": _sha256(model_path),
            "model_num_timesteps": int(model.num_timesteps),
        },
        "environment": {
            "level": environment["level"],
            "profile_mode": environment["profile_mode"],
            "seed": int(args.seed),
            "run_config": str(config_path),
            "run_config_sha256": _sha256(config_path),
            "native_tick_hz": int(env.tick_hz),
            "frame_skip": int(env_config.frame_skip),
        },
        "episode": {
            "outcome": outcome,
            "score": int(final_info["score"]),
            "ticks": int(final_info["ticks"]),
            "decisions": decision_count,
            "total_reward": total_reward,
            "action_counts": action_counts,
            "accepted_action_counts": accepted_counts,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
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
            "frame_count": frame_count,
            "decoded_frame_count": decoded_frames,
            "duration_seconds": frame_count / float(args.fps),
            "final_hold_seconds": float(args.final_hold_seconds),
        },
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
