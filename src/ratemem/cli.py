from __future__ import annotations

import argparse
import json

import numpy as np

from ratemem.allocation.objective import CoverageOracle, PacketBundle
from ratemem.allocation.snapshot import allocate_snapshot
from ratemem.artifacts.schema import AttemptManifest
from ratemem.codec.progressive import ProgressiveCodec
from ratemem.lifecycle.events import CreateEvent, ProbeEvent
from ratemem.lifecycle.replay import replay
from ratemem.state.model import Incidence
from ratemem.state.serialization import bundle_cost_bytes
from ratemem.state.store import PacketStore


def smoke_core() -> dict[str, int | str]:
    budget = 8192
    source = np.linspace(-1.0, 1.0, 16, dtype=np.float32)
    encoded = ProgressiveCodec(group_size=4).encode("concept-a", source)
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

    selected = encoded.decode(packet_count=1)
    if selected.shape != encoded.shape or not np.all(np.isfinite(selected)):
        raise RuntimeError("decoded selected prefix is nonfinite or misshaped")
    base = encoded.decode(packet_count=0)
    if base.shape != encoded.shape or not np.all(np.isfinite(base)):
        raise RuntimeError("decoded base prefix is nonfinite or misshaped")
    base_error = float(
        np.mean(np.square(source.astype(np.float64) - base.astype(np.float64)))
    )
    selected_error = float(
        np.mean(
            np.square(source.astype(np.float64) - selected.astype(np.float64))
        )
    )
    if not selected_error < base_error:
        raise RuntimeError(
            "selected prefix did not strictly improve reconstruction error"
        )

    lifecycle = replay(
        (
            CreateEvent("smoke-create", "lifecycle-a", b"base"),
            ProbeEvent("smoke-probe", "lifecycle-a"),
        ),
        budget_bytes=budget,
    )
    lifecycle_record = lifecycle.state.bases["lifecycle-a"]
    if (
        lifecycle.errors
        or lifecycle_record.reads != 0
        or lifecycle.state.serialized_bytes > budget
        or lifecycle.probe_sizes != (lifecycle.state.serialized_bytes,)
    ):
        raise RuntimeError("lifecycle create-probe smoke invariant failed")

    manifest = AttemptManifest(
        run_id="cpu-smoke",
        git_revision="0" * 40,
        config_hash="0" * 64,
        status="passed",
    )
    revalidated_manifest = AttemptManifest.model_validate_json(
        manifest.model_dump_json()
    )
    if revalidated_manifest != manifest:
        raise RuntimeError("attempt manifest failed its serialization round trip")

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
