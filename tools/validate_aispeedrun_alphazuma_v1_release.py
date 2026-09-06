"""Validate the static aispeedrun.ai AlphaZuma V1 release."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit


class _Links(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.values: list[tuple[str, str]] = []

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        values = dict(attrs)
        for key in ("href", "src", "poster"):
            if values.get(key):
                self.values.append((key, values[key] or ""))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _local_target(site_root: Path, page: Path, value: str) -> Path | None:
    parts = urlsplit(value)
    if parts.scheme or parts.netloc or value.startswith(("mailto:", "data:")):
        return None
    if not parts.path:
        return None
    path = unquote(parts.path)
    if path.startswith("/"):
        return site_root / path.lstrip("/")
    return page.parent / path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--site-root", required=True, type=Path)
    args = parser.parse_args()
    root = args.site_root.resolve()
    manifest_path = root / "records" / "alphazuma-v1-2026-08-13.json"
    sidecar_path = root / "records" / "alphazuma-v1-2026-08-13.sha256"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = manifest["records"]

    _require(manifest["status"] == "PUBLISHED_EVIDENCE", "manifest status")
    _require(manifest["release_date"] == "2026-08-13", "release date")
    _require(len(records) == 14, "record count")
    _require(len({record["adv_level"] for record in records}) == 14, "duplicate level")
    leads = {
        record["adv_level"]
        for record in records
        if record["human_reference"]["cross_domain_numerical_lead"]
    }
    _require(leads == {2, 10, 12, 17}, f"unexpected lead set: {leads}")
    _require(
        manifest["claims"]["original_client_rta_world_record_claims"] == 0,
        "official claim count",
    )

    actual_manifest_hash = _sha256(manifest_path)
    sidecar_hash, sidecar_name = sidecar_path.read_text(encoding="ascii").split()
    _require(sidecar_name == manifest_path.name, "sidecar filename")
    _require(
        f"sha256:{sidecar_hash}" == actual_manifest_hash,
        "manifest sidecar mismatch",
    )

    media_paths: set[str] = set()
    media_bytes = 0
    for record in records:
        _require(record["time"]["native_ticks"] / 100 == record["time"]["seconds"], "time mismatch")
        _require(record["evaluation"]["pre_render_identity_match"], "pre-render mismatch")
        _require(record["evaluation"]["rendered_identity_match"], "render mismatch")
        _require(not record["human_reference"]["comparable_for_official_record_claim"], "bad human claim")
        media = record["media"]
        for key, hash_key in (("video", "video_sha256"), ("poster", "poster_sha256")):
            relative = media[key]
            _require(relative not in media_paths, f"duplicate media: {relative}")
            media_paths.add(relative)
            path = root / relative
            _require(path.is_file(), f"missing media: {relative}")
            _require(_sha256(path) == media[hash_key], f"media hash mismatch: {relative}")
        video_path = root / media["video"]
        poster_path = root / media["poster"]
        _require(video_path.stat().st_size == media["video_bytes"], "video size")
        _require(b"ftyp" in video_path.read_bytes()[:64], f"MP4 header: {video_path}")
        _require(poster_path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n", f"PNG header: {poster_path}")
        media_bytes += video_path.stat().st_size

    _require(len(media_paths) == 28, "media path count")
    html_pages = [root / "index.html", root / "zuma.html", root / "zuma-alphazuma-v1.html"]
    for page in html_pages:
        text = page.read_text(encoding="utf-8")
        links = _Links()
        links.feed(text)
        for attribute, value in links.values:
            target = _local_target(root, page, value)
            if target is None:
                continue
            _require(target.exists(), f"broken {attribute} in {page.name}: {value}")

    zuma_html = (root / "zuma.html").read_text(encoding="utf-8")
    detail_html = (root / "zuma-alphazuma-v1.html").read_text(encoding="utf-8")
    index_html = (root / "index.html").read_text(encoding="utf-8")
    _require(zuma_html.count('class="record"') == 14, "Zuma board record seats")
    _require(zuma_html.count('class="leadmark"') == 4, "Zuma board lead markers")
    _require("CATEGORY IN PREPARATION" not in zuma_html, "stale category status")
    _require("AI column is empty" not in zuma_html, "stale empty-board copy")
    _require("zuma-alphazuma-v1.html" in zuma_html, "missing release link")
    _require("14 V1 RUNS LIVE" in index_html, "homepage status")
    _require("records/alphazuma-v1-2026-08-13.json" in detail_html, "detail manifest link")
    _require('preload="none"' in detail_html, "video preload policy")
    _require("not original-client RTA runs" in detail_html, "category disclaimer")

    public_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in [*html_pages, root / "README.md", manifest_path]
    )
    forbidden_patterns = (
        r"/mnt/[a-z]/",
        r"[A-Za-z]:\\Users\\",
        r"Users[/\\]Laure",
        r"ZumaTraining",
    )
    for pattern in forbidden_patterns:
        _require(not re.search(pattern, public_text), f"local path leak: {pattern}")

    print(
        json.dumps(
            {
                "status": "PASS",
                "manifest_sha256": actual_manifest_hash,
                "records": len(records),
                "numerical_leads": sorted(leads),
                "media_files": len(media_paths),
                "video_bytes": media_bytes,
                "html_pages": [path.name for path in html_pages],
                "broken_local_links": 0,
                "local_path_leaks": 0,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
