#!/usr/bin/env python3
"""Create and validate an immutable HF model-config profile.

The profile reuses every source-model artifact through symlinks and replaces
only config.json.  It is built atomically under the caller-selected output
parent so concurrent experiment submissions cannot observe a partial profile.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path


MANIFEST_NAME = "fastafd-model-profile.json"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile-id", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--expected-source-config-sha256", required=True)
    parser.add_argument("--max-position-embeddings", type=int, required=True)
    parser.add_argument("--rope-factor", type=float, required=True)
    parser.add_argument(
        "--original-max-position-embeddings", type=int, required=True
    )
    return parser.parse_args()


def expected_profile(args: argparse.Namespace) -> tuple[bytes, dict]:
    source = args.source.resolve(strict=True)
    output = args.output.absolute()
    if source == output or source in output.parents:
        raise RuntimeError("profile output must not be the source or its child")
    source_config_path = source / "config.json"
    source_index_path = source / "model.safetensors.index.json"
    if not source_config_path.is_file() or not source_index_path.is_file():
        raise RuntimeError(f"incomplete source model: {source}")

    source_config_bytes = source_config_path.read_bytes()
    source_config_sha256 = sha256(source_config_bytes)
    if source_config_sha256 != args.expected_source_config_sha256:
        raise RuntimeError(
            "source config hash mismatch: "
            f"{source_config_sha256} != {args.expected_source_config_sha256}"
        )
    source_config = json.loads(source_config_bytes)
    if source_config.get("rope_scaling") not in (None, {}):
        raise RuntimeError(
            f"source already has rope_scaling={source_config.get('rope_scaling')!r}"
        )
    if args.rope_factor <= 1:
        raise RuntimeError("rope factor must be greater than one")
    if args.original_max_position_embeddings <= 0:
        raise RuntimeError("original max position must be positive")
    if args.max_position_embeddings <= args.original_max_position_embeddings:
        raise RuntimeError("profile max position must exceed the original max")

    rope_scaling = {
        "factor": args.rope_factor,
        "original_max_position_embeddings": args.original_max_position_embeddings,
        "rope_type": "yarn",
    }
    profile_config = dict(source_config)
    profile_config["max_position_embeddings"] = args.max_position_embeddings
    profile_config["rope_scaling"] = rope_scaling
    profile_config_bytes = (
        json.dumps(profile_config, indent=2, sort_keys=True) + "\n"
    ).encode()
    manifest = {
        "schema_version": 1,
        "profile_id": args.profile_id,
        "source_model_path": str(source),
        "source_revision": args.source_revision,
        "source_config_sha256": source_config_sha256,
        "source_max_position_embeddings": source_config.get(
            "max_position_embeddings"
        ),
        "profile_model_path": str(output),
        "profile_config_sha256": sha256(profile_config_bytes),
        "profile_max_position_embeddings": args.max_position_embeddings,
        "rope_scaling": rope_scaling,
        "artifact_policy": "all non-config artifacts symlink to pinned source model",
    }
    return profile_config_bytes, manifest


def validate_profile(
    source: Path,
    output: Path,
    profile_config_bytes: bytes,
    manifest: dict,
) -> None:
    if not output.is_dir():
        raise RuntimeError(f"profile path is not a directory: {output}")
    config_path = output / "config.json"
    manifest_path = output / MANIFEST_NAME
    if config_path.read_bytes() != profile_config_bytes:
        raise RuntimeError(f"profile config mismatch: {config_path}")
    observed_manifest = json.loads(manifest_path.read_text())
    observed_comparable = dict(observed_manifest)
    expected_comparable = dict(manifest)
    for path_field in ("source_model_path", "profile_model_path"):
        observed_path = Path(observed_comparable.pop(path_field)).resolve(strict=True)
        expected_path = Path(expected_comparable.pop(path_field)).resolve(strict=True)
        if observed_path != expected_path:
            raise RuntimeError(
                f"profile manifest {path_field} mismatch: "
                f"{observed_path} != {expected_path}"
            )
    if observed_comparable != expected_comparable:
        raise RuntimeError(f"profile manifest mismatch: {manifest_path}")

    source_names = {item.name for item in source.iterdir()}
    expected_names = (source_names - {"config.json"}) | {
        "config.json",
        MANIFEST_NAME,
    }
    observed_names = {item.name for item in output.iterdir()}
    if observed_names != expected_names:
        raise RuntimeError(
            f"profile artifact set mismatch: {observed_names ^ expected_names}"
        )
    for item in source.iterdir():
        if item.name == "config.json":
            continue
        link = output / item.name
        if not link.is_symlink() or link.resolve(strict=True) != item.resolve(strict=True):
            raise RuntimeError(f"invalid profile artifact link: {link}")


def main() -> None:
    args = parse_args()
    source = args.source.resolve(strict=True)
    output = args.output.absolute()
    profile_config_bytes, manifest = expected_profile(args)
    if output.exists():
        validate_profile(source, output, profile_config_bytes, manifest)
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        build = Path(
            tempfile.mkdtemp(
                prefix=f".{output.name}.build-", dir=output.parent
            )
        )
        try:
            for item in source.iterdir():
                if item.name != "config.json":
                    os.symlink(item.resolve(strict=True), build / item.name)
            (build / "config.json").write_bytes(profile_config_bytes)
            (build / MANIFEST_NAME).write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n"
            )
            try:
                build.rename(output)
            except OSError:
                if not output.exists():
                    raise
            validate_profile(source, output, profile_config_bytes, manifest)
        finally:
            if build.exists():
                shutil.rmtree(build)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
