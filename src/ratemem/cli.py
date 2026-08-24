from __future__ import annotations

import argparse
import json

import numpy as np

from ratemem.allocation.objective import CoverageOracle, PacketBundle
from ratemem.allocation.snapshot import allocate_snapshot
from ratemem.codec.progressive import ProgressiveCodec
from ratemem.state.model import Incidence
from ratemem.state.serialization import bundle_cost_bytes
from ratemem.state.store import PacketStore


def smoke_core() -> dict[str, int | str]:
    budget = 8192
    encoded = ProgressiveCodec(group_size=4).encode(
        "concept-a", np.linspace(-1.0, 1.0, 16, dtype=np.float32)
    )
    store = PacketStore.empty(budget).create(
        "concept-a", encoded.base_payload, created_at=0
    )
    packet = encoded.packets[0].packet
    incidence = Incidence("concept-a", packet.packet_id, gain_q=8)
    store = store.attach(packet, incidence)
    packet_bytes = bundle_cost_bytes(packet, (incidence,))
    oracle = CoverageOracle(
        {
            packet.packet_id: PacketBundle(
                packet.packet_id, packet_bytes, {"concept-a": (1.0,)}
            )
        },
        {"concept-a": 1.0},
        {"concept-a": (1.0,)},
    )
    chosen = allocate_snapshot(oracle, packet_bytes)
    if chosen != frozenset({packet.packet_id}):
        raise RuntimeError(
            "snapshot allocator rejected the only feasible useful packet"
        )
    return {
        "status": "passed",
        "serialized_bytes": store.state.serialized_bytes,
        "budget_bytes": budget,
    }


def main() -> None:
    parser = argparse.ArgumentParser(prog="ratemem")
    parser.add_argument("command", choices=("smoke-core",))
    args = parser.parse_args()
    if args.command == "smoke-core":
        print(json.dumps(smoke_core(), sort_keys=True))


if __name__ == "__main__":
    main()
