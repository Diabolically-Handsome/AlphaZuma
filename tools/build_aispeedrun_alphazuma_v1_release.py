"""Build the public AlphaZuma V1 evidence bundle for aispeedrun.ai.

The release is derived from frozen blind-evaluation receipts.  This builder
refuses stale or mismatched videos, strips local paths, and emits one public
manifest plus the verified media files used by the website.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from zuma_rl.original_data import OriginalGameCatalog


MORNING_AGGREGATE_SHA256 = (
    "sha256:b82e017511d39540f17e28ab50bfa451cdac9804f2b67b7764762eb94d8621c8"
)
MORNING_PREREGISTRATION_SHA256 = (
    "sha256:55543e20ecae3fac92072d8c40fcb65c686183cd292128d7504aa76eb0a4d3ad"
)
MORNING_MODELS_MANIFEST_SHA256 = (
    "sha256:5f9c76008d70e6ce8c156916d6a1b214ad34509399a3142f44d4158e382408e0"
)
JUNGLE2_EVALUATION_SHA256 = (
    "sha256:8af20964e41d18ad967a5432784b5f4acef75bc1ab94f6a5b1801b0ed25405e9"
)
JUNGLE2_PREREGISTRATION_SHA256 = (
    "sha256:6c9ecdae6ad42c2eefb1e987ea25840b13ca8985ab3cce73b53b23dbbd377005"
)


RECORD_SPECS: tuple[dict[str, Any], ...] = (
    {
        "adv": 1,
        "level_id": "Jungle1",
        "metadata": "outputs/aispeedrun-alpha-v1/adv1-shipwrecked-10.99s.json",
        "video": "outputs/aispeedrun-alpha-v1/adv1-shipwrecked-10.99s.mp4",
        "poster": "outputs/aispeedrun-alpha-v1/adv1-shipwrecked-10.99s-poster.png",
        "public_stem": "adv01-shipwrecked-10.99s",
        "human_seconds": 6,
        "human_runner": "UfahTrainer8",
        "human_url": "https://www.speedrun.com/zumas_revenge/runs/y2nwn17y",
    },
    {
        "adv": 2,
        "level_id": "Jungle2",
        "metadata": "outputs/alphazuma-v1-jungle2-blind-11.10s-seed20300823.json",
        "video": "outputs/alphazuma-v1-jungle2-blind-11.10s-seed20300823.mp4",
        "poster": "outputs/alphazuma-v1-jungle2-blind-11.10s-seed20300823-poster.png",
        "public_stem": "adv02-jungle2-11.10s",
        "human_seconds": 12,
        "human_runner": "WYL071018",
        "human_url": "https://www.speedrun.com/zumas_revenge/runs/zpd01drz",
        "independent_jungle2": True,
    },
    {
        "adv": 3,
        "level_id": "Jungle3",
        "metadata": "outputs/aispeedrun-alpha-v1/adv3-echo-beach-18.37s.json",
        "video": "outputs/aispeedrun-alpha-v1/adv3-echo-beach-18.37s.mp4",
        "poster": "outputs/aispeedrun-alpha-v1/adv3-echo-beach-18.37s-poster.png",
        "public_stem": "adv03-echo-beach-18.37s",
        "human_seconds": 7,
        "human_runner": "LookingForDreams",
        "human_url": "https://www.speedrun.com/zumas_revenge/runs/zgepd6ey",
    },
    {
        "adv": 4,
        "level_id": "Jungle4",
        "metadata": "outputs/aispeedrun-alpha-v1/adv4-bones-of-an-idol-16.97s.json",
        "video": "outputs/aispeedrun-alpha-v1/adv4-bones-of-an-idol-16.97s.mp4",
        "poster": "outputs/aispeedrun-alpha-v1/adv4-bones-of-an-idol-16.97s-poster.png",
        "public_stem": "adv04-bones-of-an-idol-16.97s",
        "human_seconds": 7,
        "human_runner": "LookingForDreams",
        "human_url": "https://www.speedrun.com/zumas_revenge/runs/z5prp15m",
    },
    {
        "adv": 6,
        "level_id": "jungle6",
        "metadata": "outputs/aispeedrun-alpha-v1/adv6-jungle-clearing-23.64s.json",
        "video": "outputs/aispeedrun-alpha-v1/adv6-jungle-clearing-23.64s.mp4",
        "poster": "outputs/aispeedrun-alpha-v1/adv6-jungle-clearing-23.64s-poster.png",
        "public_stem": "adv06-jungle-clearing-23.64s",
        "human_seconds": 18,
        "human_runner": "WYL071018",
        "human_url": "https://www.speedrun.com/zumas_revenge/runs/yd13e6xy",
    },
    {
        "adv": 8,
        "level_id": "Jungle8",
        "metadata": "outputs/aispeedrun-alpha-v1/adv8-nasal-passage-23.73s.json",
        "video": "outputs/aispeedrun-alpha-v1/adv8-nasal-passage-23.73s.mp4",
        "poster": "outputs/aispeedrun-alpha-v1/adv8-nasal-passage-23.73s-poster.png",
        "public_stem": "adv08-nasal-passage-23.73s",
        "human_seconds": 8,
        "human_runner": "WYL071018",
        "human_url": "https://www.speedrun.com/zumas_revenge/runs/zx0336gz",
    },
    {
        "adv": 9,
        "level_id": "Jungle9",
        "metadata": "outputs/aispeedrun-alpha-v1/adv9-creeper-copse-42.04s.json",
        "video": "outputs/aispeedrun-alpha-v1/adv9-creeper-copse-42.04s.mp4",
        "poster": "outputs/aispeedrun-alpha-v1/adv9-creeper-copse-42.04s-poster.png",
        "public_stem": "adv09-creeper-copse-42.04s",
        "human_seconds": 30,
        "human_runner": "Leshik_Doshik",
        "human_url": "https://www.speedrun.com/zumas_revenge/runs/y8vnel1z",
    },
    {
        "adv": 10,
        "level_id": "Jungle10",
        "metadata": "outputs/aispeedrun-alpha-v1/adv10-darkness-falls-21.35s.json",
        "video": "outputs/aispeedrun-alpha-v1/adv10-darkness-falls-21.35s.mp4",
        "poster": "outputs/aispeedrun-alpha-v1/adv10-darkness-falls-21.35s-poster.png",
        "public_stem": "adv10-darkness-falls-21.35s",
        "human_seconds": 25,
        "human_runner": "Leshik_Doshik",
        "human_url": "https://www.speedrun.com/zumas_revenge/runs/yv25oe6m",
    },
    {
        "adv": 11,
        "level_id": "village1",
        "metadata": "outputs/aispeedrun-alpha-v1/adv11-watering-hole-30.90s.json",
        "video": "outputs/aispeedrun-alpha-v1/adv11-watering-hole-30.90s.mp4",
        "poster": "outputs/aispeedrun-alpha-v1/adv11-watering-hole-30.90s-poster.png",
        "public_stem": "adv11-watering-hole-30.90s",
        "human_seconds": 13,
        "human_runner": "justcater",
        "human_url": "https://www.speedrun.com/zumas_revenge/runs/m7wkw79z",
    },
    {
        "adv": 12,
        "level_id": "village2",
        "metadata": "outputs/aispeedrun-alpha-v1/adv12-overgrown-plaza-23.68s.json",
        "video": "outputs/aispeedrun-alpha-v1/adv12-overgrown-plaza-23.68s.mp4",
        "poster": "outputs/aispeedrun-alpha-v1/adv12-overgrown-plaza-23.68s-poster.png",
        "public_stem": "adv12-overgrown-plaza-23.68s",
        "human_seconds": 24,
        "human_runner": "Zanum",
        "human_url": "https://www.speedrun.com/zumas_revenge/runs/y4gp2pdy",
    },
    {
        "adv": 14,
        "level_id": "village4",
        "metadata": "outputs/aispeedrun-alpha-v1/adv14-sandspirals-42.77s.json",
        "video": "outputs/aispeedrun-alpha-v1/adv14-sandspirals-42.77s.mp4",
        "poster": "outputs/aispeedrun-alpha-v1/adv14-sandspirals-42.77s-poster.png",
        "public_stem": "adv14-sandspirals-42.77s",
        "human_seconds": 29,
        "human_runner": "WYL071018",
        "human_url": "https://www.speedrun.com/zumas_revenge/runs/y8x9e2wm",
    },
    {
        "adv": 15,
        "level_id": "village5",
        "metadata": "outputs/aispeedrun-alpha-v1/adv15-abandoned-well-40.14s.json",
        "video": "outputs/aispeedrun-alpha-v1/adv15-abandoned-well-40.14s.mp4",
        "poster": "outputs/aispeedrun-alpha-v1/adv15-abandoned-well-40.14s-poster.png",
        "public_stem": "adv15-abandoned-well-40.14s",
        "human_seconds": 9,
        "human_runner": "LookingForDreams",
        "human_url": "https://www.speedrun.com/zumas_revenge/runs/z5xplojy",
    },
    {
        "adv": 17,
        "level_id": "village7",
        "metadata": "outputs/aispeedrun-alpha-v1/adv17-seven-sides-24.64s.json",
        "video": "outputs/aispeedrun-alpha-v1/adv17-seven-sides-24.64s.mp4",
        "poster": "outputs/aispeedrun-alpha-v1/adv17-seven-sides-24.64s-poster.png",
        "public_stem": "adv17-seven-sides-24.64s",
        "human_seconds": 41,
        "human_runner": "Zanum",
        "human_url": "https://www.speedrun.com/zumas_revenge/runs/y805d3dm",
    },
    {
        "adv": 18,
        "level_id": "village8",
        "metadata": "outputs/aispeedrun-alpha-v1/adv18-vines-54.89s.json",
        "video": "outputs/aispeedrun-alpha-v1/adv18-vines-54.89s.mp4",
        "poster": "outputs/aispeedrun-alpha-v1/adv18-vines-54.89s-poster.png",
        "public_stem": "adv18-vines-54.89s",
        "human_seconds": 37,
        "human_runner": "Leshik_Doshik",
        "human_url": "https://www.speedrun.com/zumas_revenge/runs/ydwjw6wm",
    },
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"expected object in {path}")
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _model_level_summary(
    aggregate: dict[str, Any], model_id: str, level_id: str
) -> dict[str, Any]:
    for model_summary in aggregate["models"]:
        if model_summary["model"]["id"] != model_id:
            continue
        for summary in model_summary["level_summaries"]:
            if summary["level_id"] == level_id:
                return summary
    raise ValueError(f"missing level summary for {model_id}/{level_id}")


def _copy_verified(source: Path, destination: Path, expected_sha256: str) -> None:
    _require(source.is_file(), f"missing media: {source}")
    actual = _sha256(source)
    _require(actual == expected_sha256, f"media hash mismatch: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    _require(_sha256(destination) == expected_sha256, f"copied hash mismatch: {destination}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--site-root", type=Path, required=True)
    parser.add_argument("--aggregate", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--models-manifest", type=Path, required=True)
    parser.add_argument("--jungle2-evaluation", type=Path, required=True)
    parser.add_argument("--jungle2-preregistration", type=Path, required=True)
    parser.add_argument("--original-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    project_root = Path(__file__).resolve().parents[1]
    site_root = args.site_root.resolve()
    media_root = site_root / "media" / "alphazuma-v1"
    records_root = site_root / "records"

    aggregate = _read_json(args.aggregate)
    jungle2_evaluation = _read_json(args.jungle2_evaluation)
    _require(_sha256(args.aggregate) == MORNING_AGGREGATE_SHA256, "aggregate drift")
    _require(
        _sha256(args.preregistration) == MORNING_PREREGISTRATION_SHA256,
        "morning preregistration drift",
    )
    _require(
        _sha256(args.models_manifest) == MORNING_MODELS_MANIFEST_SHA256,
        "models manifest drift",
    )
    _require(
        _sha256(args.jungle2_evaluation) == JUNGLE2_EVALUATION_SHA256,
        "Jungle2 blind evaluation drift",
    )
    _require(
        _sha256(args.jungle2_preregistration) == JUNGLE2_PREREGISTRATION_SHA256,
        "Jungle2 preregistration drift",
    )
    _require(aggregate["status"] == "COMPLETE", "morning aggregate incomplete")
    _require(all(aggregate["validation"].values()), "morning validation failed")
    _require(jungle2_evaluation["decision"]["status"] == "PASS", "Jungle2 failed")
    _require(len(jungle2_evaluation["replays"]) == 3, "Jungle2 replay count drift")

    jungle2_name = OriginalGameCatalog(args.original_root).level("Jungle2").display_name
    records: list[dict[str, Any]] = []
    unique_models: dict[str, dict[str, Any]] = {}

    for spec in RECORD_SPECS:
        metadata_path = project_root / spec["metadata"]
        metadata = _read_json(metadata_path)
        candidate = metadata["candidate"]
        video_info = metadata["video"]
        independent = bool(spec.get("independent_jungle2"))

        _require(metadata["status"] == "COMPLETE", f"incomplete video: {metadata_path}")
        _require(candidate["outcome"] == "win", f"non-winning record: Adv {spec['adv']}")
        _require(candidate["ticks"] / 100 == candidate["seconds"], "tick/time mismatch")
        if independent:
            _require(
                metadata["source"]["blind_evaluation_sha256"]
                == JUNGLE2_EVALUATION_SHA256,
                "Jungle2 evaluation binding mismatch",
            )
            _require(
                metadata["verification"]["pre_render_matches_blind_attempt"]
                and metadata["verification"]["rendered_matches_blind_attempt"],
                "Jungle2 rendered replay mismatch",
            )
            selected_model_summary = next(
                item
                for item in jungle2_evaluation["model_summaries"]
                if item["model_id"] == candidate["model_id"]
            )
            evaluation_id = "jungle2-independent-blind-352"
            display_name = jungle2_name
            level_stats = {
                "attempts": selected_model_summary["attempts"],
                "wins": selected_model_summary["wins"],
                "truncations": selected_model_summary["truncations"],
            }
        else:
            _require(candidate["level_id"] == spec["level_id"], "level binding mismatch")
            _require(
                metadata["source"]["aggregate_sha256"] == MORNING_AGGREGATE_SHA256,
                "aggregate binding mismatch",
            )
            _require(
                metadata["verification"]["pre_render_matches_formal_attempt"]
                and metadata["verification"]["rendered_matches_formal_attempt"],
                "formal rendered replay mismatch",
            )
            summary = _model_level_summary(
                aggregate, candidate["model_id"], spec["level_id"]
            )
            evaluation_id = "morning-paired-multimodel-blind-512"
            display_name = candidate["display_name"]
            level_stats = {
                "attempts": summary["attempts"],
                "wins": summary["wins"],
                "truncations": summary["truncations"],
            }

        source_video = project_root / spec["video"]
        source_poster = project_root / spec["poster"]
        public_video = media_root / f"{spec['public_stem']}.mp4"
        public_poster = media_root / f"{spec['public_stem']}-poster.png"
        _copy_verified(source_video, public_video, video_info["sha256"])
        _copy_verified(source_poster, public_poster, video_info["poster_sha256"])
        _require(source_video.stat().st_size == video_info["bytes"], "video size mismatch")

        model_key = candidate["model_sha256"]
        unique_models[model_key] = {
            "id": candidate["model_id"],
            "training_steps": candidate["training_steps"],
            "sha256": model_key,
        }
        delta = round(float(candidate["seconds"]) - spec["human_seconds"], 2)
        records.append(
            {
                "adv_level": spec["adv"],
                "environment_level_id": spec["level_id"],
                "display_name": display_name,
                "time": {
                    "native_ticks": candidate["ticks"],
                    "seconds": candidate["seconds"],
                    "clock_hz": 100,
                },
                "score": candidate["score"],
                "seed": candidate["seed"],
                "model": {
                    "id": candidate["model_id"],
                    "training_steps": candidate["training_steps"],
                    "sha256": candidate["model_sha256"],
                },
                "trajectory_sha256": candidate["trajectory_sha256"],
                "decisions": candidate["decisions"],
                "desired_action_counts": candidate["desired_action_counts"],
                "executed_action_counts": candidate["executed_action_counts"],
                "evaluation": {
                    "id": evaluation_id,
                    "selected_model_level_stats": level_stats,
                    "attempt_index": candidate.get("attempt_index"),
                    "pre_render_identity_match": True,
                    "rendered_identity_match": True,
                },
                "human_reference": {
                    "category": "original-client Any% IL",
                    "displayed_seconds": spec["human_seconds"],
                    "runner": spec["human_runner"],
                    "url": spec["human_url"],
                    "checked_utc_date": "2026-08-13",
                    "cross_domain_delta_seconds": delta,
                    "cross_domain_numerical_lead": delta < 0,
                    "comparable_for_official_record_claim": False,
                },
                "media": {
                    "video": f"media/alphazuma-v1/{public_video.name}",
                    "video_sha256": video_info["sha256"],
                    "video_bytes": video_info["bytes"],
                    "codec": video_info["codec"],
                    "pixel_format": video_info["pixel_format"],
                    "width": video_info["width"],
                    "height": video_info["height"],
                    "fps": video_info["fps"],
                    "duration_seconds": video_info["duration_seconds"],
                    "poster": f"media/alphazuma-v1/{public_poster.name}",
                    "poster_sha256": video_info["poster_sha256"],
                },
            }
        )

    _require(len(records) == 14, "record count drift")
    _require(len({record["adv_level"] for record in records}) == 14, "duplicate level")
    lead_count = sum(
        record["human_reference"]["cross_domain_numerical_lead"]
        for record in records
    )
    _require(lead_count == 4, "human-reference lead count drift")

    manifest = {
        "schema": "aispeedrun.alphazuma-v1-release",
        "version": 1,
        "release_id": "alphazuma-v1-structured-state-2026-08-13",
        "release_date": "2026-08-13",
        "status": "PUBLISHED_EVIDENCE",
        "title": "AlphaZuma V1 structured-state individual-level records",
        "claims": {
            "verified_blind_simulator_records": len(records),
            "cross_domain_numerical_leads_vs_current_human_displayed_times": lead_count,
            "original_client_rta_world_record_claims": 0,
        },
        "category_contract": {
            "id": "structured-state-deterministic-simulator-native-tick-v1",
            "policy_input": (
                "actor-observable structured state; no rendered pixels and no privileged "
                "debug state; tunnel-hidden balls omitted"
            ),
            "policy_architecture": "entity 1D CNN plus attention plus MLP",
            "action_space": "wait/fire/swap plus one of 180 aim bins",
            "action_selection": "deterministic argmax with actor-observable masks",
            "clock": "native simulator ticks at 100 Hz from reset to terminal win",
            "input_profile": {
                "id": "elite-human-v1",
                "reaction_delay_ticks": 12,
                "reaction_delay_milliseconds": 120,
                "max_aim_speed_degrees_per_second": 1080.0,
                "max_aim_acceleration_degrees_per_second_squared": 18000.0,
                "minimum_button_interval_ticks": 5,
                "minimum_button_interval_milliseconds": 50,
            },
            "rendering_role": "post-hoc simulator visualization only; pixels never enter the policy",
            "not_equivalent_to": [
                "original Zuma's Revenge executable RTA",
                "Speedrun.com PC individual-level timing",
                "a vision-controlled run",
            ],
        },
        "human_reference": {
            "source": "https://www.speedrun.com/zumas_revenge/levels",
            "checked_utc_date": "2026-08-13",
            "use": "context only; displayed leaderboard seconds are not a shared timing domain",
        },
        "evaluations": [
            {
                "id": "morning-paired-multimodel-blind-512",
                "status": aggregate["status"],
                "aggregate_sha256": MORNING_AGGREGATE_SHA256,
                "preregistration_sha256": MORNING_PREREGISTRATION_SHA256,
                "models_manifest_sha256": MORNING_MODELS_MANIFEST_SHA256,
                "levels": aggregate["levels"],
                "models": len(aggregate["models"]),
                "attempts_per_level_per_model": aggregate["attempts_per_level"],
                "total_attempts": sum(shard["attempts"] for shard in aggregate["shards"]),
                "seed_design": "paired unseen seeds across two frozen models",
                "selection_rule": "fastest winning formal blind attempt per level across frozen models",
                "all_validation_checks_passed": True,
            },
            {
                "id": "jungle2-independent-blind-352",
                "status": jungle2_evaluation["decision"]["status"],
                "evaluation_sha256": JUNGLE2_EVALUATION_SHA256,
                "preregistration_sha256": JUNGLE2_PREREGISTRATION_SHA256,
                "frozen_checkpoints": len(jungle2_evaluation["model_summaries"]),
                "attempts_per_checkpoint": 32,
                "total_attempts": jungle2_evaluation["completed_attempts"],
                "selected_candidate_replay_repetitions": len(jungle2_evaluation["replays"]),
                "selection_rule": "first sub-12 candidate by preregistered ranking",
                "selected_replays_identical": True,
            },
        ],
        "models": sorted(unique_models.values(), key=lambda item: item["training_steps"]),
        "records": records,
    }

    serialized = json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    for forbidden in ("/mnt/", "C:\\", "D:\\", "Users/Laure", "Users\\Laure"):
        _require(forbidden not in serialized, f"local path leaked: {forbidden}")

    records_root.mkdir(parents=True, exist_ok=True)
    manifest_path = records_root / "alphazuma-v1-2026-08-13.json"
    manifest_path.write_text(serialized, encoding="utf-8", newline="\n")
    manifest_sha256 = _sha256(manifest_path)
    sidecar = records_root / "alphazuma-v1-2026-08-13.sha256"
    sidecar.write_text(
        f"{manifest_sha256.removeprefix('sha256:')}  {manifest_path.name}\n",
        encoding="ascii",
        newline="\n",
    )
    print(
        json.dumps(
            {
                "status": "COMPLETE",
                "manifest": str(manifest_path),
                "manifest_sha256": manifest_sha256,
                "records": len(records),
                "cross_domain_numerical_leads": lead_count,
                "media_files": len(records) * 2,
                "media_bytes": sum(
                    record["media"]["video_bytes"] for record in records
                ),
                "jungle2_display_name": jungle2_name,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
