"""Park-and-settle distillation v2: v1 recipe on memmap-backed observations.

THIN wrapper over ``tools.distill_alphazuma_55_park_settle_v1``.  The
recipe -- model construction, losses, ``annotate_park_settle``, the
training loop, receipts, checkpoints, calibration report, and the
completion.json format -- is v1's, imported and delegated to, never
re-implemented.  v2 exists for exactly one reason: v1's
``load_replay_dataset`` materializes the ``observations`` member in RAM
(``np.asarray(archive[key], dtype=np.float32)`` decompresses the whole
member, and the final ``np.concatenate`` copies again even for a single
shard).  At the coverage-v4 aggregate scale (799k rows x 22,904 float32
= ~73 GB) that cannot fit the ~40 GB WSL budget, while the v1 training
loop only ever touches ``dataset.observations`` through per-batch row
fancy-indexing plus ``obs_to_tensor`` -- semantically transparent for a
numpy memmap view.

What v2 adds
------------
1. ``--extract-observations`` step (automatic when needed): the NPZ's
   ``observations`` zip member is itself a complete ``.npy`` file, so it
   is stream-decompressed (bounded read buffer, sha256 computed on the
   fly, zip CRC verified by ``zipfile`` during the read) to
   ``<npz_dir>/<npz_stem>.observations.npy`` next to the NPZ, holding
   RAM to the copy buffer (default 64 MiB) instead of ~73 GB.  A JSON
   receipt (``<npz_stem>.observations.receipt.json``) binds the .npy
   sha256 to the source NPZ sha256.  Extraction is skipped when a
   receipted .npy already exists AND both hashes still verify; any
   mismatch (or an unreceipted/orphaned artifact) is refused rather than
   silently repaired.
2. ``load_replay_dataset_memmap``: builds the exact v1
   ``ParkSettleDataset`` -- identical manifest/sha256 verification,
   identical key detection via v1's ``_first_key`` and key tuples,
   identical relaxed-mask refusal and ``mask_semantics`` enforcement,
   identical episode-index remap -- but with ``observations`` backed by
   ``np.load(<stem>.observations.npy, mmap_mode="r")``.  Actions, masks
   and tick metadata still load from the NPZ into RAM (small).  The v1
   constructor path ``ParkSettleDataset.from_episode_rows`` is BYPASSED
   deliberately: it routes observations through
   ``np.asarray(..., dtype=np.float32)``, which is a zero-copy view for
   an aligned float32 memmap under current numpy but is not a contract;
   the raw dataclass constructor performs no coercion, so the memmap
   view is stored as-is and ``dataset.validate()`` (the identical v1
   validation) runs unchanged.  A guard raises if the ndarray view ever
   stops sharing the memmap's memory.  Memmap backing cannot span
   multiple shards without an in-RAM concatenation copy, so manifests
   naming more than one NPZ are refused (use v1 for those).
3. Training entry: the CLI is identical to v1 plus the extraction
   flags.  ``main`` strips the extraction flags, temporarily rebinds
   ``v1.load_replay_dataset`` to the memmap loader, and calls
   ``v1.main`` (restored in ``finally``), so hyperparameters, receipts,
   checkpoints, the calibration report and the completion.json format
   stay IDENTICAL.  The dataset receipt -- and therefore
   ``completion.json["dataset"]`` -- additionally records
   ``observation_backing`` (mode ``memmap_npy`` plus the extraction
   receipts).

Real-dataset launch (799,113 rows, cuda:0)::

    .venv/bin/python tools/distill_alphazuma_55_park_settle_v2.py \
        --dataset-manifest /mnt/d/ZumaTraining/\
alphazuma-55-coverage-replay-v4-r020-b300k/replay.dataset.json \
        --original-root "/mnt/d/SteamLibrary/steamapps/common/\
Zuma's Revenge" \
        --run-dir /mnt/d/ZumaTraining/\
alphazuma-55-park-settle-distill-s99081660-v1 \
        --device cuda:0 --epochs 20 --checkpoint-interval-epochs 2

``formal_seed_consumption`` stays ``false`` throughout (inherited from
v1's receipts): this wrapper never steps an environment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Sequence
import zipfile

if __package__ in {None, ""}:
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_PROJECT_ROOT))
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

import numpy as np

from tools import distill_alphazuma_55 as legacy
from tools import distill_alphazuma_55_park_settle_v1 as v1


SCRIPT_PATH = Path(__file__).resolve()
EXTRACTION_SCHEMA = (
    "zuma-rl.alphazuma-55-park-settle-distillation-observation-extraction"
)
EXTRACTION_VERSION = 1
DEFAULT_EXTRACTION_BUFFER_MIB = 64
_NPY_SUFFIX = ".npy"


def observation_npy_path(npz_path: Path) -> Path:
    """``<npz_dir>/<npz_stem>.observations.npy`` beside the source NPZ."""

    npz_path = Path(npz_path)
    return npz_path.with_name(npz_path.stem + ".observations.npy")


def extraction_receipt_path(npz_path: Path) -> Path:
    npz_path = Path(npz_path)
    return npz_path.with_name(npz_path.stem + ".observations.receipt.json")


class _MemberKeys:
    """Adapter presenting zip member stems through v1's ``_first_key``."""

    def __init__(self, files: Sequence[str]) -> None:
        self.files = list(files)


def _observation_member(npz_path: Path) -> str:
    """The NPZ zip member holding observations, via v1 key precedence.

    Only the zip central directory is read; no member is decompressed.
    """

    with zipfile.ZipFile(npz_path) as archive:
        names = archive.namelist()
    keys = _MemberKeys(
        [
            name[: -len(_NPY_SUFFIX)]
            for name in names
            if name.endswith(_NPY_SUFFIX)
        ]
    )
    key = v1._first_key(keys, v1._OBSERVATION_KEYS, required=True)
    return str(key) + _NPY_SUFFIX


def _read_npy_header(path: Path) -> tuple[tuple[int, ...], bool, Any]:
    """(shape, fortran_order, dtype) from a raw ``.npy`` header.

    Mirrors the header-only peek used by
    ``tools.run_alphazuma_55_coverage_replay_build_v1`` -- the payload
    is never loaded.
    """

    with path.open("rb") as stream:
        version = np.lib.format.read_magic(stream)
        if version == (1, 0):
            return np.lib.format.read_array_header_1_0(stream)
        if version in {(2, 0), (3, 0)}:
            # 3.0 only widens header field names to utf-8; the binary
            # header layout matches 2.0.
            return np.lib.format.read_array_header_2_0(stream)
        raise ValueError(
            f"unsupported NPY format version {version}: {path}"
        )


def _stream_extract_member(
    npz_path: Path,
    member: str,
    destination: Path,
    *,
    buffer_bytes: int,
) -> tuple[str, int]:
    """Stream one zip member to ``destination``; (sha256, bytes written).

    RAM is bounded by ``buffer_bytes`` (plus the zlib window).  The
    sha256 is computed on the fly, so the ~73 GB payload is read from
    the zip exactly once and never re-read for the receipt.  ``zipfile``
    verifies the member CRC as the stream is consumed.
    """

    digest = hashlib.sha256()
    written = 0
    with zipfile.ZipFile(npz_path) as archive:
        with archive.open(member) as source:
            with destination.open("xb") as sink:
                while True:
                    chunk = source.read(buffer_bytes)
                    if not chunk:
                        break
                    sink.write(chunk)
                    digest.update(chunk)
                    written += len(chunk)
                sink.flush()
                os.fsync(sink.fileno())
    return "sha256:" + digest.hexdigest(), written


def extract_observations(
    npz_path: Path,
    *,
    buffer_bytes: int = DEFAULT_EXTRACTION_BUFFER_MIB * 1024 * 1024,
    npz_sha256: str | None = None,
) -> dict[str, Any]:
    """Ensure a receipted raw-``.npy`` twin of the NPZ observations.

    Returns ``{"action": "extracted" | "reused", "npy_path": ...,
    "receipt_path": ..., "receipt": {...}}``.  ``npz_sha256`` may carry
    a precomputed source digest so callers that already hashed the NPZ
    (the loader verifies manifest receipts first) do not hash it twice.

    Honest outcomes: an existing .npy is REUSED only when its receipt's
    npy sha256 AND source NPZ sha256 both still verify; any mismatch,
    an unreceipted .npy, or an orphaned receipt is refused with the
    reason -- never silently re-extracted over.
    """

    npz_path = Path(npz_path).resolve(strict=True)
    if int(buffer_bytes) < 1:
        raise ValueError("extraction buffer must be at least 1 byte")
    member = _observation_member(npz_path)
    npy_path = observation_npy_path(npz_path)
    receipt_path = extraction_receipt_path(npz_path)
    if npz_sha256 is None:
        npz_sha256 = legacy._sha256(npz_path)

    if npy_path.exists():
        if not receipt_path.exists():
            raise ValueError(
                f"unreceipted observations extraction already exists; "
                f"refusing to trust or overwrite it (delete it to "
                f"re-extract): {npy_path}"
            )
        receipt = legacy._read_json(receipt_path)
        problems: list[str] = []
        if receipt.get("schema") != EXTRACTION_SCHEMA:
            problems.append(
                f"receipt schema is {receipt.get('schema')!r}, "
                f"expected {EXTRACTION_SCHEMA!r}"
            )
        source = receipt.get("source_npz")
        extracted = receipt.get("observations_npy")
        source = source if isinstance(source, dict) else {}
        extracted = extracted if isinstance(extracted, dict) else {}
        if source.get("member") != member:
            problems.append(
                f"receipt extracted member {source.get('member')!r}, "
                f"the NPZ observations member is {member!r}"
            )
        if source.get("sha256") != npz_sha256:
            problems.append(
                f"source NPZ sha256 changed (receipt "
                f"{source.get('sha256')}, disk {npz_sha256})"
            )
        actual_bytes = int(npy_path.stat().st_size)
        if int(extracted.get("bytes", -1)) != actual_bytes:
            problems.append(
                f".npy size changed (receipt {extracted.get('bytes')}, "
                f"disk {actual_bytes})"
            )
        else:
            npy_digest = legacy._sha256(npy_path)
            if extracted.get("sha256") != npy_digest:
                problems.append(
                    f".npy sha256 changed (receipt "
                    f"{extracted.get('sha256')}, disk {npy_digest})"
                )
        if problems:
            listing = "\n  ".join(problems)
            raise ValueError(
                f"observation extraction receipt mismatch for "
                f"{npy_path}; refusing to reuse or overwrite:\n  "
                f"{listing}"
            )
        return {
            "action": "reused",
            "npy_path": str(npy_path),
            "receipt_path": str(receipt_path),
            "receipt": receipt,
        }

    if receipt_path.exists():
        raise ValueError(
            f"stale extraction receipt without its .npy; refusing to "
            f"overwrite it (delete it to re-extract): {receipt_path}"
        )

    temporary = npy_path.with_name(f".{npy_path.name}.{os.getpid()}.tmp")
    temporary.unlink(missing_ok=True)
    try:
        npy_digest, written = _stream_extract_member(
            npz_path, member, temporary, buffer_bytes=int(buffer_bytes)
        )
        shape, fortran_order, dtype = _read_npy_header(temporary)
        if len(shape) != 2 or fortran_order or dtype != np.float32:
            raise ValueError(
                f"observations must be C-order 2-D float32, got shape "
                f"{shape} dtype {dtype}: {npz_path}"
            )
        os.replace(temporary, npy_path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    receipt = {
        "schema": EXTRACTION_SCHEMA,
        "version": EXTRACTION_VERSION,
        "created_utc": legacy._utc_now(),
        "extractor": {
            "path": str(SCRIPT_PATH),
            "sha256": legacy._sha256(SCRIPT_PATH),
        },
        "source_npz": {
            "path": str(npz_path),
            "sha256": npz_sha256,
            "member": member,
        },
        "observations_npy": {
            "path": str(npy_path),
            "sha256": npy_digest,
            "bytes": int(written),
            "shape": [int(value) for value in shape],
            "dtype": "float32",
            "fortran_order": False,
        },
        "ram_bound": {
            "strategy": "zip_member_stream_copy_with_inline_sha256",
            "buffer_bytes": int(buffer_bytes),
        },
    }
    legacy._write_json_atomic(receipt_path, receipt)
    return {
        "action": "extracted",
        "npy_path": str(npy_path),
        "receipt_path": str(receipt_path),
        "receipt": receipt,
    }


def _manifest_entries(
    manifest: dict[str, Any], manifest_path: Path
) -> tuple[list[Any], str, bool]:
    """(entries, layout, entry_is_episode) -- v1's manifest branching."""

    if "shards" in manifest:
        entries = list(manifest["shards"])
        layout = "motor-observable-replay-v2-shards"
        entry_is_episode = False
    elif "episodes" in manifest:
        entries = list(manifest["episodes"])
        layout = "motor-observable-replay-v4-coverage-episodes"
        entry_is_episode = True
    elif "files" in manifest:
        entries = list(manifest["files"])
        layout = "generic-files"
        entry_is_episode = False
    else:
        raise ValueError(
            "replay manifest needs a 'shards', 'episodes', or 'files' list"
        )
    if not entries:
        raise ValueError(f"replay manifest lists no data: {manifest_path}")
    return entries, layout, entry_is_episode


def _entry_npz_path(
    manifest_path: Path, ordinal: int, entry: Any
) -> tuple[Path, str]:
    """Resolve one manifest entry and verify its sha256 receipt (v1)."""

    if not isinstance(entry, dict) or "path" not in entry:
        raise ValueError(
            f"manifest entry {ordinal} lacks a 'path': {entry!r}"
        )
    path = Path(str(entry["path"]))
    if not path.is_absolute():
        path = manifest_path.parent / path
    path = path.resolve(strict=True)
    digest = legacy._sha256(path)
    declared = entry.get("sha256")
    if declared is not None and declared != digest:
        raise ValueError(f"shard bytes differ from manifest: {path}")
    return path, digest


def extract_manifest_observations(
    manifest_path: Path,
    *,
    buffer_bytes: int = DEFAULT_EXTRACTION_BUFFER_MIB * 1024 * 1024,
) -> list[dict[str, Any]]:
    """Run the extraction step for every NPZ a manifest names."""

    manifest_path = Path(manifest_path).resolve(strict=True)
    manifest = legacy._read_json(manifest_path)
    entries, _, _ = _manifest_entries(manifest, manifest_path)
    results: list[dict[str, Any]] = []
    for ordinal, entry in enumerate(entries):
        path, digest = _entry_npz_path(manifest_path, ordinal, entry)
        results.append(
            extract_observations(
                path, buffer_bytes=buffer_bytes, npz_sha256=digest
            )
        )
    return results


def load_replay_dataset_memmap(
    manifest_path: Path,
    *,
    extraction_buffer_bytes: int = (
        DEFAULT_EXTRACTION_BUFFER_MIB * 1024 * 1024
    ),
) -> tuple[v1.ParkSettleDataset, dict[str, Any]]:
    """v1's ``load_replay_dataset`` with memmap-backed observations.

    Same signature and receipt shape as v1 (plus
    ``observation_backing``), and the SAME checks in the SAME failure
    modes: manifest/shard sha256 receipts, v1 key precedence via
    ``v1._first_key``, refusal of relaxed training masks, enforcement of
    declared ``mask_semantics``, and the v1 episode-index remap
    (``np.unique(..., return_inverse=True)``), so splits and training
    are bit-identical to the v1 in-RAM path on the same manifest.

    v1's loader body cannot be reused directly because it materializes
    observations twice (``np.asarray(archive[key])`` decompresses the
    member into RAM; ``np.concatenate`` then copies even for a single
    shard); everything importable from v1 is imported instead of
    re-implemented, and the dataset object is built through the raw
    ``ParkSettleDataset`` constructor (no coercion) + the identical
    ``dataset.validate()``.  Only single-NPZ manifests are accepted --
    the coverage-v4 aggregate layout -- because multiple shards cannot
    share one memmap without an in-RAM copy.
    """

    manifest_path = Path(manifest_path).resolve(strict=True)
    manifest = legacy._read_json(manifest_path)
    entries, layout, entry_is_episode = _manifest_entries(
        manifest, manifest_path
    )
    if len(entries) != 1:
        raise ValueError(
            f"memmap observation backing supports exactly one aggregate "
            f"NPZ (the coverage-v4 layout); this manifest names "
            f"{len(entries)} -- use "
            f"tools.distill_alphazuma_55_park_settle_v1 for multi-shard "
            f"manifests: {manifest_path}"
        )
    declared_semantics = manifest.get("mask_semantics")
    if declared_semantics is not None and not isinstance(
        declared_semantics, dict
    ):
        raise ValueError(
            "manifest mask_semantics must map NPZ keys to semantics strings"
        )

    path, digest = _entry_npz_path(manifest_path, 0, entries[0])
    extraction = extract_observations(
        path, buffer_bytes=extraction_buffer_bytes, npz_sha256=digest
    )
    npy_path = Path(extraction["npy_path"]).resolve(strict=True)

    with np.load(path) as archive:
        observation_key = v1._first_key(
            archive, v1._OBSERVATION_KEYS, required=True
        )
        action_key = v1._first_key(archive, v1._ACTION_KEYS, required=True)
        mask_key = v1._first_key(archive, v1._MASK_KEYS, required=False)
        if mask_key is None:
            relaxed = [
                key
                for key in v1._RELAXED_MASK_KEYS
                if key in archive.files
            ]
            if relaxed:
                raise ValueError(
                    f"NPZ shard offers only relaxed training masks "
                    f"{relaxed}; the park-settle trainer requires the "
                    f"exact runtime valid-action mask (one of "
                    f"{v1._MASK_KEYS}): {path}"
                )
            raise ValueError(
                f"NPZ shard offers none of {v1._MASK_KEYS}: "
                f"{sorted(archive.files)}"
            )
        if declared_semantics is not None:
            semantics = declared_semantics.get(mask_key)
            if semantics not in v1._EXACT_MASK_SEMANTICS:
                raise ValueError(
                    f"manifest declares mask semantics {semantics!r} "
                    f"for NPZ key '{mask_key}'; the park-settle "
                    f"trainer requires one of "
                    f"{sorted(v1._EXACT_MASK_SEMANTICS)}"
                )
        episode_key = v1._first_key(
            archive, v1._EPISODE_KEYS, required=False
        )
        tick_key = v1._first_key(archive, v1._TICK_KEYS, required=False)
        if str(observation_key) + _NPY_SUFFIX != str(
            extraction["receipt"]["source_npz"]["member"]
        ):
            raise ValueError(
                f"extraction receipt member disagrees with the NPZ "
                f"observation key {observation_key!r}: {path}"
            )
        # The observations member is deliberately NEVER indexed here --
        # archive[observation_key] would decompress ~73 GB into RAM.
        raw_actions = np.asarray(archive[action_key], dtype=np.int64)
        masks = np.asarray(archive[mask_key], dtype=np.bool_)
        rows = int(raw_actions.shape[0])
        if episode_key is not None:
            local = np.asarray(archive[episode_key], dtype=np.int64)
            _, remapped = np.unique(local, return_inverse=True)
            shard_episodes = remapped.astype(np.int64)
        elif entry_is_episode:
            shard_episodes = np.full(rows, 0, np.int64)
        else:
            shard_episodes = None
        if tick_key is not None:
            shard_ticks = np.asarray(archive[tick_key], dtype=np.int64)
        elif entry_is_episode:
            shard_ticks = np.arange(rows, dtype=np.int64)
        else:
            shard_ticks = None

    memmap = np.load(npy_path, mmap_mode="r")
    if memmap.ndim != 2 or memmap.dtype != np.float32:
        raise ValueError(
            f"extracted observations must be 2-D float32: {npy_path}"
        )
    if int(memmap.shape[0]) != rows:
        raise ValueError(
            f"extracted observations rows ({int(memmap.shape[0])}) "
            f"disagree with the NPZ action rows ({rows}): {npy_path}"
        )
    observations = np.asarray(memmap)
    if observations.base is not memmap and observations is not memmap:
        raise RuntimeError(
            "np.asarray(memmap) stopped returning a zero-copy view; "
            "the memmap loading contract is broken"
        )

    if shard_episodes is not None and shard_ticks is not None:
        temporal_structure = "episode_ticks"
        episode_indices = shard_episodes
        tick_indices = shard_ticks
    else:
        temporal_structure = "absent"
        episode_indices = np.arange(rows, dtype=np.int64)
        tick_indices = np.zeros(rows, dtype=np.int64)
    # Raw dataclass constructor: from_episode_rows would route the
    # memmap through np.asarray(..., dtype=np.float32) (a no-copy view
    # today but not a contract); the raw constructor stores the view
    # untouched and validate() runs the identical v1 checks.
    dataset = v1.ParkSettleDataset(
        observations=observations,
        raw_actions=raw_actions,
        masks=masks,
        episode_indices=np.asarray(episode_indices, dtype=np.int64),
        tick_indices=np.asarray(tick_indices, dtype=np.int64),
        temporal_structure=temporal_structure,
    )
    dataset.validate()

    entry_receipt = {
        "path": str(path),
        "sha256": digest,
        "rows": rows,
        "keys": {
            "observations": observation_key,
            "actions": action_key,
            "masks": mask_key,
            "episode_indices": episode_key,
            "tick_indices": tick_key,
        },
    }
    receipt = {
        "manifest": {
            "path": str(manifest_path),
            "sha256": legacy._sha256(manifest_path),
            "schema": manifest.get("schema"),
            "version": manifest.get("version"),
        },
        "layout": layout,
        "mask_semantics": (
            dict(declared_semantics)
            if declared_semantics is not None
            else None
        ),
        "entries": [entry_receipt],
        "sample_count": dataset.sample_count,
        "temporal_structure": dataset.temporal_structure,
        "observation_backing": {
            "mode": "memmap",
            "storage": "raw_npy_memmap_read_only",
            "wrapper": {
                "path": str(SCRIPT_PATH),
                "sha256": legacy._sha256(SCRIPT_PATH),
            },
            "observations_npy": dict(
                extraction["receipt"]["observations_npy"]
            ),
            "extraction": {
                "action": extraction["action"],
                "receipt_path": extraction["receipt_path"],
                "receipt": extraction["receipt"],
            },
        },
    }
    return dataset, receipt


def build_parser() -> argparse.ArgumentParser:
    """v1's parser (identical CLI) plus the extraction flags."""

    parser = v1.build_parser()
    parser.description = __doc__
    parser.add_argument(
        "--extract-observations-only",
        action="store_true",
        help=(
            "run the receipted observations extraction for every NPZ "
            "the manifest names, print the receipts, and exit"
        ),
    )
    parser.add_argument(
        "--extraction-buffer-mib",
        type=int,
        default=DEFAULT_EXTRACTION_BUFFER_MIB,
        help="stream-copy buffer for the extraction step (MiB)",
    )
    return parser


def _extraction_flag_parser() -> argparse.ArgumentParser:
    """Consumes ONLY the v2 extraction flags (for v1 argv pass-through)."""

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--extract-observations-only", action="store_true")
    parser.add_argument("--extraction-buffer-mib", type=int)
    return parser


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:]) if argv is None else list(argv)
    args = build_parser().parse_args(argv)
    if int(args.extraction_buffer_mib) < 1:
        raise SystemExit("--extraction-buffer-mib must be at least 1")
    buffer_bytes = int(args.extraction_buffer_mib) * 1024 * 1024
    if args.extract_observations_only:
        results = extract_manifest_observations(
            args.dataset_manifest.expanduser(), buffer_bytes=buffer_bytes
        )
        print(
            json.dumps(
                v1._jsonable(
                    {"status": "EXTRACTED", "extractions": results}
                ),
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
        )
        return 0

    _, v1_argv = _extraction_flag_parser().parse_known_args(argv)

    def _memmap_loader(
        manifest_path: Path,
    ) -> tuple[v1.ParkSettleDataset, dict[str, Any]]:
        return load_replay_dataset_memmap(
            manifest_path, extraction_buffer_bytes=buffer_bytes
        )

    # Scoped rebinding: v1.main resolves ``load_replay_dataset`` through
    # its module globals, so this delegates the WHOLE v1 flow (config
    # construction, prototype interface, model build, execute_training,
    # summary printing) unchanged while only the dataset loading is
    # swapped for the memmap loader.  v1's file is never modified; the
    # binding is restored even on failure.
    original_loader = v1.load_replay_dataset
    v1.load_replay_dataset = _memmap_loader
    try:
        return v1.main(v1_argv)
    finally:
        v1.load_replay_dataset = original_loader


if __name__ == "__main__":
    raise SystemExit(main())
