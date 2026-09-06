from __future__ import annotations

import struct

from zuma_rl.pc_protocol_evidence import (
    REGISTRY_ROOTS,
    PcStateSnapshot,
    SnapshotRegistryNode,
    SnapshotRegistryValue,
)
from tools.pc_state_transaction import overlay_snapshot_registry_values


def test_registry_overlay_changes_only_the_declared_existing_value() -> None:
    snapshot = PcStateSnapshot(
        session_nonce="ab" * 16,
        phase="pre-template",
        captured_perf_counter_ns=100,
        files=(),
        registry_nodes=(
            SnapshotRegistryNode(
                root=REGISTRY_ROOTS[0],
                subkey="",
                exists=True,
                values=(
                    SnapshotRegistryValue(
                        name="Is3D",
                        value_type=4,
                        data=struct.pack("<I", 1),
                    ),
                    SnapshotRegistryValue(
                        name="Other",
                        value_type=4,
                        data=struct.pack("<I", 9),
                    ),
                ),
            ),
            SnapshotRegistryNode(
                root=REGISTRY_ROOTS[1],
                subkey="",
                exists=False,
                values=(),
            ),
        ),
    )

    changed = overlay_snapshot_registry_values(
        snapshot,
        (
            (
                REGISTRY_ROOTS[0],
                "",
                "Is3D",
                4,
                struct.pack("<I", 0),
            ),
        ),
        phase="software-pre",
        captured_perf_counter_ns=200,
    )

    assert changed.phase == "software-pre"
    assert changed.captured_perf_counter_ns == 200
    assert changed.state_root != snapshot.state_root
    values = {
        item.name: item.data for item in changed.registry_nodes[0].values
    }
    assert values == {
        "Is3D": struct.pack("<I", 0),
        "Other": struct.pack("<I", 9),
    }
