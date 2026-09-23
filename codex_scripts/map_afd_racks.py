#!/usr/bin/env python3
"""Order TP1 AFD ranks for rack-local fan-in, using an explicit node inventory.

Input: JSON list of {host, address, rack, gpus} records. Addresses must match Ray.
Output: placement manifest and environment to merge into a prepared plan's env.json.
Slots describe capacity, not CUDA ordinals: Ray chooses devices within each node.
This mapper does not establish fabric connectivity; the caller must verify it.
"""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path


def map_racks(inventory, ffn_racks):
    if not inventory or not ffn_racks or len(set(ffn_racks)) != len(ffn_racks):
        raise ValueError("Require nonempty inventory and distinct FFN racks")
    racks = defaultdict(list)
    hosts, addresses = set(), set()
    for node in inventory:
        host, address, rack = (node[key] for key in ("host", "address", "rack"))
        gpus = node["gpus"]
        if not all(isinstance(value, str) and value for value in (host, address, rack)):
            raise ValueError("Host, address and rack must be nonempty strings")
        if any(char in address for char in ",+\n "):
            raise ValueError(f"Invalid Ray node address: {address!r}")
        if host in hosts or address in addresses:
            raise ValueError(f"Duplicate host/address: {host}/{address}")
        if type(gpus) is not int or gpus < 1:
            raise ValueError(f"Invalid GPU capacity for {host}: {gpus}")
        hosts.add(host)
        addresses.add(address)
        racks[rack].extend(dict(host=host, address=address, rack=rack, slot=slot)
                           for slot in range(gpus))
    if missing := set(ffn_racks) - racks.keys():
        raise ValueError(f"FFN racks absent from inventory: {sorted(missing)}")
    for slots in racks.values():
        slots.sort(key=lambda slot: (slot["host"], slot["slot"]))

    # Reserve one FFN slot in each selected rack; all remaining local slots send
    # to it. Full-attention racks contribute equally to every FFN rank.
    ffn = [dict(racks[rack][0], dp_rank=rank) for rank, rack in enumerate(ffn_racks)]
    groups = [racks[rack][1:] for rack in ffn_racks]
    for rack in sorted(racks.keys() - set(ffn_racks)):
        slots = racks[rack]
        if len(slots) % len(ffn):
            raise ValueError(f"Attention-only rack {rack}: {len(slots)} GPUs cannot "
                             f"divide evenly among {len(ffn)} FFN GPUs")
        for index, slot in enumerate(slots):
            groups[index % len(ffn)].append(slot)
    fanins = [len(group) for group in groups]
    if len(set(fanins)) != 1 or not fanins[0]:
        raise ValueError(f"Current rank routing requires equal positive fan-in: {fanins}")
    attention = [dict(slot, ffn_dp_rank=rank)
                 for rank, group in enumerate(groups) for slot in group]
    for rank, slot in enumerate(attention):
        slot["dp_rank"] = rank
        # AfdTopology.mlp_dp_for_attn_dp uses this exact integer routing rule.
        if rank * len(ffn) // len(attention) != slot["ffn_dp_rank"]:
            raise ValueError("Placement does not match AFD integer rank routing")
    used = Counter(slot["host"] for slot in attention + ffn)
    if used != Counter({node["host"]: node["gpus"] for node in inventory}):
        raise ValueError("Placement does not exactly consume the supplied inventory")
    return {
        "schema_version": 1,
        "contract": "TP1; rack-local attention; evenly striped attention-only racks",
        "slot_semantics": "capacity slots, not pinned CUDA device ordinals",
        "inventory": sorted(inventory, key=lambda node: node["host"]),
        "attention_gpus": len(attention), "ffn_gpus": len(ffn),
        "fanin_per_ffn": fanins[0],
        "environment": {
            "MINISGL_AFD_ATTN_NODES": ",".join(slot["address"] for slot in attention),
            "MINISGL_AFD_MLP_NODES": ",".join(slot["address"] for slot in ffn),
            "MINISGL_AFD_PLACEMENT_CONTRACT": "rack_local_balanced",
            "AFD_ATTENTION_GPU_COUNT": str(len(attention)),
            "AFD_ATTN_TP_SIZE": "1",
            "AFD_MLP_EP_SIZE": str(len(ffn)),
        },
        "attention": attention, "ffn": ffn,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inventory", type=Path)
    parser.add_argument("--ffn-racks", required=True,
                        help="Comma-separated rack names, in FFN rank order")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = map_racks(json.loads(args.inventory.read_text()), args.ffn_racks.split(","))
    # Refuse to silently replace a placement already consumed by a job.
    with args.output.open("x") as output:
        json.dump(result, output, indent=2)
        output.write("\n")
    print(f"A{result['attention_gpus']}:F{result['ffn_gpus']}, "
          f"fan-in {result['fanin_per_ffn']}; {args.output}")


if __name__ == "__main__":
    main()
