from __future__ import annotations

import struct

import pytest

from tools.set_popcap_dmo_random_seed import set_random_seed
from zuma_rl import popcap_dmo


def _demo(seed: int = 123) -> bytes:
    product = b"1.0.4.9496"
    idle = bytes((0xFF, 0x03))
    return b"".join(
        (
            struct.pack(
                "<IIIH",
                popcap_dmo.DEMO_FILE_ID,
                1,
                seed,
                len(product),
            ),
            product,
            struct.pack("<I", 15),
            idle,
        )
    )


def test_set_random_seed_changes_only_header_seed() -> None:
    source = _demo()
    output, provenance = set_random_seed(
        source,
        popcap_dmo,
        expected_seed=123,
        new_seed=456,
    )

    assert output[:8] == source[:8]
    assert output[12:] == source[12:]
    assert struct.unpack_from("<I", output, 8)[0] == 456
    assert provenance["transformation"]["command_stream_preserved"] is True


def test_set_random_seed_rejects_unbound_source_seed() -> None:
    with pytest.raises(ValueError, match="random seed mismatch"):
        set_random_seed(
            _demo(),
            popcap_dmo,
            expected_seed=999,
            new_seed=456,
        )
