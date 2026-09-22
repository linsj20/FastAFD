#!/usr/bin/env python3
"""Seed compatible native modules or precompiled kernels into worker caches.

The caller must establish matching source/ABI, compiler target and execution image.
Device kernels require an explicit --kind device donor built for the serving source.
Existing unrelated cache entries are refused.
TVM module links retain generated-source paths; DeepGEMM uses locked hardlinks or
an explicit copy after an ABI-compatible device-only edit.
"""

import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil


def seed_host(args):
    donor = args.donor.resolve(strict=True)
    prefix = "_minisgl_deepep_moe" if args.kind == "deepep" else "_minisgl_deepgemm"
    if not donor.is_file() or not donor.name.startswith(prefix + "."):
        raise ValueError(f"Expected a compiled {prefix} host extension")
    signature = None
    if args.kind == "deepep":
        signature = (donor.parent.parent / "build.signature").read_text()
    targets = args.cache_root
    if args.node_ip:
        targets = [root / ("ray-node-" + ip.replace(":", "_").replace(".", "_"))
                   for root in targets for ip in args.node_ip]
    records = []
    for root in targets:
        root.mkdir(parents=True, exist_ok=True)
        with (root / "build.lock").open("w") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            signature_path = root / "build.signature"
            if signature is not None and signature_path.exists() and signature_path.read_text() != signature:
                raise RuntimeError(f"Refuse to replace an existing build signature: {signature_path}")
            lib = root / "lib"
            lib.mkdir(exist_ok=True)
            existing = list(lib.glob(prefix + "*.so"))
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
            if signature is not None:
                signature_path.write_text(signature)
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


def seed_device(args):
    donor = (args.donor / "cache").resolve(strict=True)
    kernels = sorted(donor.glob("kernel.*"))
    if not kernels or any(not (p / "kernel.cu").is_file() or
                          not (p / "kernel.cubin").is_file() or
                          not (p / "kernel.cubin").stat().st_size for p in kernels):
        raise ValueError(f"Incomplete device cache: {donor}")
    roots = [args.cache_root[0]]
    if args.node_ip:
        roots = []
        for ip in dict.fromkeys(args.node_ip):
            node = args.cache_root[0] / ("ray-node-" + ip.replace(":", "_").replace(".", "_"))
            roots.extend([node, *(node / f"rank{rank}" for rank in range(1, args.rank_count + 1))])
    targets = [root / "cache" for root in roots]
    for target in targets:
        if (target.exists() or target.is_symlink()) and not (
                target.is_symlink() and target.resolve() == donor):
            raise ValueError(f"Refuse to replace an existing device cache: {target}")
    for target in targets:
        if not target.is_symlink():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.symlink_to(donor, target_is_directory=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"donor": str(donor), "kernels": len(kernels),
        "cache_links": [str(p) for p in targets]}, indent=2) + "\n")
    print(json.dumps({"kernels": len(kernels), "cache_links": len(targets)}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('donor', type=Path)
    parser.add_argument('--kind', choices=('deepgemm', 'deepep', 'tvm', 'device'), default='deepgemm')
    parser.add_argument('--cache-root', type=Path, action='append', required=True)
    parser.add_argument('--node-ip', action='append')
    parser.add_argument('--rank-count', type=int, default=0,
                        help='TVM/device: seed rank1..N plus each unranked node cache')
    parser.add_argument('--copy', action='store_true',
                        help='Native host modules: copy into empty caches with current mtime')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.kind in ('tvm', 'device'):
        if len(args.cache_root) != 1 or args.copy:
            parser.error('TVM/device seeding requires one cache root and does not support --copy')
        if args.rank_count < 0 or (args.rank_count and not args.node_ip):
            parser.error('Rank seeding requires node IPs and a nonnegative count')
        (seed_tvm if args.kind == 'tvm' else seed_device)(args)
    else:
        if args.rank_count:
            parser.error('--rank-count is only supported for TVM/device caches')
        seed_host(args)


if __name__ == '__main__':
    main()
