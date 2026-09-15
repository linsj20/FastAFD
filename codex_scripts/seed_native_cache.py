#!/usr/bin/env python3
"""Seed compatible native DeepGEMM or TVM modules into isolated worker caches.

The caller must establish matching source/ABI, compiler target and execution image.
Device JIT kernels are never seeded. Existing unrelated cache entries are refused.
TVM module links retain generated-source paths; DeepGEMM uses locked hardlinks or
an explicit copy after an ABI-compatible device-only edit.
"""

import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil


def seed_deepgemm(args):
    donor = args.donor.resolve(strict=True)
    if not donor.is_file() or not donor.name.startswith("_minisgl_deepgemm."):
        raise ValueError("Expected a compiled DeepGEMM host extension")
    targets = args.cache_root
    if args.node_ip:
        targets = [root / ("ray-node-" + ip.replace(":", "_").replace(".", "_"))
                   for root in targets for ip in args.node_ip]
    records = []
    for root in targets:
        root.mkdir(parents=True, exist_ok=True)
        with (root / "build.lock").open("w") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            lib = root / "lib"
            lib.mkdir(exist_ok=True)
            existing = list(lib.glob("_minisgl_deepgemm*.so"))
            target = lib / donor.name
            if existing:
                if args.copy:
                    raise RuntimeError(f"Copy seeding requires an empty cache: {lib}")
                if len(existing) != 1 or not os.path.samefile(existing[0], donor):
                    raise RuntimeError(f"Refuse to replace an existing extension: {lib}")
                state = "already_linked"
            else:
                if args.copy:
                    shutil.copyfile(donor, target)
                    state = "copied"
                else:
                    os.link(donor, target)
                    state = "linked"
            records.append({"target": str(target), "status": state})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"donor": str(donor), "targets": records},
                                     indent=2) + "\n")
    print(json.dumps({"donor": str(donor), "targets": len(records)}), flush=True)


def seed_tvm(args):
    donor = args.donor.resolve(strict=True)
    modules = sorted(p for p in donor.iterdir() if p.is_dir())
    if not modules:
        raise ValueError(f"No completed native modules: {donor}")
    for module in modules:
        libraries = list(module.glob("*.so"))
        if len(libraries) != 1 or not libraries[0].stat().st_size or not (module / "build.ninja").is_file():
            raise ValueError(f"Incomplete module: {module}")
    roots = [args.cache_root[0]]
    if args.node_ip:
        roots = []
        for ip in dict.fromkeys(args.node_ip):
            node = args.cache_root[0] / ("ray-node-" + ip.replace(":", "_").replace(".", "_"))
            roots.extend([node, *(node / f"rank{rank}" for rank in range(1, args.rank_count + 1))])
    targets = [(root / module.name, module) for root in roots for module in modules]
    for target, _ in targets:
        if target.exists() or target.is_symlink():
            raise ValueError(f"Refuse to replace an existing cache module: {target}")
    for target, module in targets:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(module, target_is_directory=True)
    result = {"donor": str(donor), "modules": [p.name for p in modules],
              "cache_roots": [str(p) for p in roots], "module_links": len(targets)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"cache_roots": len(roots), "module_links": len(targets)}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('donor', type=Path)
    parser.add_argument('--kind', choices=('deepgemm', 'tvm'), default='deepgemm')
    parser.add_argument('--cache-root', type=Path, action='append', required=True)
    parser.add_argument('--node-ip', action='append')
    parser.add_argument('--rank-count', type=int, default=0,
                        help='TVM only: seed rank1..N plus each unranked node cache')
    parser.add_argument('--copy', action='store_true',
                        help='DeepGEMM only: copy into empty caches with current mtime')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.kind == 'tvm':
        if len(args.cache_root) != 1 or args.copy:
            parser.error('TVM seeding requires one cache root and does not support --copy')
        if args.rank_count < 0 or (args.rank_count and not args.node_ip):
            parser.error('Rank seeding requires node IPs and a nonnegative count')
        seed_tvm(args)
    else:
        if args.rank_count:
            parser.error('--rank-count is only supported for TVM modules')
        seed_deepgemm(args)


if __name__ == '__main__':
    main()
