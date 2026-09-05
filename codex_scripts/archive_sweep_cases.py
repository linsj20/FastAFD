#!/usr/bin/env python3
"""Create a verified tar.zst from whole sweep cases or selected result files.

The legacy mode archives direct-child case directories named by ``--case-list``.
The filtered mode uses repeatable ``--include`` globs relative to the source
root. Record-only mode preserves a case inventory without payloads.
"""

from __future__ import annotations

import argparse
import csv
import fnmatch
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import threading
import time


README_NAME = "README.md"
MANIFEST_NAME = "MANIFEST.tsv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument(
        "--source-parent",
        "--source-root",
        dest="source_root",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--case-list",
        type=Path,
        help="Direct-child directory names for whole-case or record-only mode.",
    )
    parser.add_argument(
        "--include",
        action="append",
        default=[],
        help="Source-root-relative file glob for filtered mode; repeat as needed.",
    )
    parser.add_argument("--sweep-name", "--title", dest="sweep_name", required=True)
    parser.add_argument("--start-time", default="not recorded")
    parser.add_argument("--classification", default="not recorded")
    parser.add_argument("--note", action="append", default=[])
    parser.add_argument(
        "--record-only",
        action="store_true",
        help="Archive only the case manifest, as an audit/deletion record.",
    )
    parser.add_argument("--zstd-threads", type=int, default=8)
    parser.add_argument(
        "--exclude-dir-name",
        action="append",
        default=[],
        help="Exclude every subtree whose directory has this exact name.",
    )
    parser.add_argument(
        "--skip-full-integrity-test",
        action="store_true",
        help="Skip zstd -t; tar/zstd exit and filtered-member audit still apply.",
    )
    parser.add_argument(
        "--finalize-existing-partial",
        action="store_true",
        help="Validate and promote an existing successful .partial without rewriting it.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.archive.suffixes[-2:] != [".tar", ".zst"]:
        raise ValueError("archive must end in .tar.zst")
    if not args.source_root.is_dir():
        raise FileNotFoundError(args.source_root)
    if args.zstd_threads < 1:
        raise ValueError("zstd threads must be positive")
    if args.record_only and (not args.case_list or args.include):
        raise ValueError("record-only mode requires --case-list and forbids --include")
    if not args.case_list and not args.include:
        raise ValueError("provide --case-list for whole cases or --include for files")
    for pattern in args.include:
        parsed = PurePosixPath(pattern)
        if not pattern or parsed.is_absolute() or ".." in parsed.parts:
            raise ValueError(f"unsafe include glob: {pattern!r}")
    for name in args.exclude_dir_name:
        if not name or "/" in name or name in {".", ".."}:
            raise ValueError(f"unsafe excluded directory name: {name!r}")


def read_cases(path: Path | None) -> list[str]:
    if path is None:
        return []
    cases: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for raw in handle:
            name = raw.rstrip("\n")
            if not name or name.startswith("#"):
                continue
            if "\t" in name or "\n" in name or name in {".", ".."}:
                raise ValueError(f"unsafe case name: {name!r}")
            if Path(name).name != name:
                raise ValueError(f"case must be one direct child: {name!r}")
            cases.append(name)
    if len(cases) != len(set(cases)):
        raise ValueError("case list contains duplicates")
    return sorted(cases)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def select_files(
    root: Path,
    includes: list[str],
    excluded_dir_names: set[str],
    cases: list[str],
) -> list[Path]:
    selected: list[Path] = []
    allowed_cases = set(cases)
    for current, directories, files in os.walk(root, followlinks=False):
        directories[:] = sorted(
            name for name in directories if name not in excluded_dir_names
        )
        current_path = Path(current)
        for name in sorted(files):
            source = current_path / name
            relative = source.relative_to(root)
            if allowed_cases and relative.parts[0] not in allowed_cases:
                continue
            relative_text = relative.as_posix()
            if not any(
                fnmatch.fnmatchcase(relative_text, pattern) for pattern in includes
            ):
                continue
            if not source.is_file() or source.is_symlink():
                raise ValueError(f"selected payload must be a regular file: {source}")
            selected.append(relative)
    selected = sorted(set(selected), key=lambda path: path.as_posix())
    if not selected:
        raise RuntimeError("include globs selected no regular files")
    return selected


def write_case_manifest(
    work: Path, args: argparse.Namespace, cases: list[str]
) -> None:
    manifest = work / MANIFEST_NAME
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(
            [
                "case_index",
                "case_name",
                "archive_path",
                "source_path",
                "source_mtime_utc",
                "payload_archived",
            ]
        )
        for index, name in enumerate(cases, 1):
            source = args.source_root / name
            stat = source.stat()
            writer.writerow(
                [
                    index,
                    name,
                    "" if args.record_only else name,
                    str(source),
                    time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(stat.st_mtime)),
                    "no" if args.record_only else "yes",
                ]
            )


def write_file_manifest(
    work: Path, args: argparse.Namespace, selected: list[Path]
) -> int:
    total_bytes = 0
    manifest = work / MANIFEST_NAME
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(
            ["file_index", "archive_path", "size_bytes", "mtime_utc", "sha256"]
        )
        for index, relative in enumerate(selected, 1):
            source = args.source_root / relative
            stat = source.stat()
            total_bytes += stat.st_size
            writer.writerow(
                [
                    index,
                    relative.as_posix(),
                    stat.st_size,
                    time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(stat.st_mtime)),
                    sha256(source),
                ]
            )
    return total_bytes


def write_readme(
    work: Path,
    args: argparse.Namespace,
    cases: list[str],
    selected: list[Path],
    payload_bytes: int | None,
) -> None:
    if args.record_only:
        payload_text = (
            "This is an audit-only deletion record. The listed case payloads are "
            "intentionally absent."
        )
        manifest_text = "The manifest has one row per recorded case."
    else:
        payload_text = (
            "Every manifest archive_path names one selectively extractable case tree."
        )
        manifest_text = "The manifest has one row per archived case directory."

    notes = list(args.note)
    if args.include:
        notes.append("Included file globs: " + ", ".join(args.include))
    if args.exclude_dir_name:
        notes.append(
            "Excluded directory subtrees: "
            + ", ".join(f"`{name}/`" for name in args.exclude_dir_name)
        )
    note_text = "\n".join(f"- {note}" for note in notes) or "- None."
    (work / README_NAME).write_text(
        f"# {args.sweep_name}\n\n"
        f"- Start time: {args.start_time}\n"
        f"- Classification: {args.classification}\n"
        f"- Source root: `{args.source_root}`\n"
        f"- Cases recorded: {len(cases)}\n\n"
        f"{payload_text}\n\n{manifest_text}\n\n"
        "## Notes\n\n"
        f"{note_text}\n\n"
        "## Validation and extraction\n\n"
        "List members with:\n\n"
        "```bash\n"
        "tar --use-compress-program=unzstd -tf SWEEP.tar.zst\n"
        "```\n\n"
        "Extract selected member paths with the same command using `-x` and "
        "`-C DEST`. Validate extracted filtered payloads against `MANIFEST.tsv`.\n",
        encoding="utf-8",
    )


def archive_pipeline(
    args: argparse.Namespace,
    work: Path,
    cases: list[str],
    selected: list[Path],
    partial: Path,
) -> None:
    tar_cmd = ["tar", "--sort=name", "--sparse", "-cf", "-"]
    for name in args.exclude_dir_name:
        tar_cmd.extend(["--exclude", f"*/{name}"])
    if args.include:
        # Filtered result archives contain payloads only.  The manifest and
        # summary are written beside the archive so scripts/docs never enter
        # the result tarball.
        tar_cmd.extend(["-C", str(args.source_root)])
        member_list = work / "payload-members.nul"
        with member_list.open("wb") as handle:
            for relative in selected:
                handle.write(os.fsencode(relative.as_posix()))
                handle.write(b"\0")
        tar_cmd.extend(["--null", "-T", str(member_list)])
    else:
        tar_cmd.extend(["-C", str(work), README_NAME, MANIFEST_NAME])
    if not args.record_only and not args.include:
        tar_cmd.extend(["-C", str(args.source_root)])
        tar_cmd.extend(cases)

    zstd_cmd = [
        "zstd",
        f"-T{args.zstd_threads}",
        "-1",
        "--check",
        "--quiet",
        "-o",
        str(partial),
    ]
    print(
        f"ARCHIVE_START name={args.archive.name} cases={len(cases)} "
        f"files={len(selected)} record_only={args.record_only}",
        flush=True,
    )
    tar_process = subprocess.Popen(tar_cmd, stdout=subprocess.PIPE)
    assert tar_process.stdout is not None
    zstd_process = subprocess.Popen(zstd_cmd, stdin=tar_process.stdout)
    tar_process.stdout.close()

    stop = threading.Event()

    def report_progress() -> None:
        while not stop.wait(30):
            size = partial.stat().st_size if partial.exists() else 0
            print(
                f"ARCHIVE_PROGRESS name={args.archive.name} bytes={size}", flush=True
            )

    reporter = threading.Thread(target=report_progress, daemon=True)
    reporter.start()
    zstd_status = zstd_process.wait()
    tar_status = tar_process.wait()
    stop.set()
    reporter.join()
    if tar_status != 0 or zstd_status != 0:
        raise RuntimeError(
            f"archive pipeline failed: tar={tar_status} zstd={zstd_status}"
        )


def audit_filtered_members(
    archive: Path, selected: list[Path], work: Path
) -> None:
    expected = [relative.as_posix() for relative in selected]
    observed = subprocess.run(
        ["tar", "--use-compress-program=unzstd", "-tf", str(archive)],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.splitlines()
    if observed != expected:
        mismatch = work / "member-audit-mismatch.json"
        mismatch.write_text(
            json.dumps({"expected": expected, "observed": observed}, indent=2) + "\n",
            encoding="utf-8",
        )
        raise RuntimeError(
            f"archive member mismatch: expected={len(expected)} observed={len(observed)}"
        )


def finalize(
    args: argparse.Namespace,
    partial: Path,
    work: Path,
    cases: list[str],
    selected: list[Path],
    payload_bytes: int | None,
    integrity_status: str,
) -> None:
    if args.include:
        audit_filtered_members(partial, selected, work)
    os.replace(partial, args.archive)
    archive_digest = sha256(args.archive)
    summary = {
        "archive": str(args.archive),
        "archive_bytes": args.archive.stat().st_size,
        "archive_sha256": archive_digest,
        "case_count": len(cases),
        "file_count": len(selected) if args.include else None,
        "payload_bytes": payload_bytes,
        "classification": args.classification,
        "payload_archived": not args.record_only,
        "source_parent": str(args.source_root),
        "source_root": str(args.source_root),
        "start_time": args.start_time,
        "sweep_name": args.sweep_name,
        "include_globs": args.include,
        "excluded_directory_names": args.exclude_dir_name,
        "zstd_integrity_test": integrity_status,
        "tar_member_audit": "passed" if args.include else "not_applicable",
    }
    (work / "SUMMARY.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    args.archive.with_name(args.archive.name + ".summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    args.archive.with_name(args.archive.name + ".sha256").write_text(
        f"{archive_digest}  {args.archive.name}\n", encoding="utf-8"
    )
    if args.include:
        shutil.copy2(
            work / MANIFEST_NAME,
            args.archive.with_name(args.archive.name + ".manifest.tsv"),
        )
    work_root = work.parent
    shutil.rmtree(work)
    try:
        work_root.rmdir()
    except OSError:
        pass
    print(
        f"ARCHIVE_DONE name={args.archive.name} cases={len(cases)} "
        f"files={len(selected)} bytes={summary['archive_bytes']} "
        f"integrity={integrity_status} sha256={archive_digest}",
        flush=True,
    )


def main() -> int:
    args = parse_args()
    args.source_root = args.source_root.resolve()
    args.archive = args.archive.resolve()
    validate_args(args)
    cases = read_cases(args.case_list)
    for name in cases:
        source = args.source_root / name
        if not source.is_dir():
            raise FileNotFoundError(source)

    selected: list[Path] = []
    if args.include:
        selected = select_files(
            args.source_root,
            args.include,
            set(args.exclude_dir_name),
            cases,
        )

    args.archive.parent.mkdir(parents=True, exist_ok=True)
    if args.archive.exists():
        raise FileExistsError(args.archive)
    partial = args.archive.with_name(args.archive.name + ".partial")
    work = (
        args.archive.parent
        / "archive_work"
        / args.archive.name.removesuffix(".tar.zst")
    )
    if args.finalize_existing_partial:
        if not partial.is_file():
            raise FileNotFoundError(partial)
        required_metadata = (MANIFEST_NAME,) if args.include else (README_NAME, MANIFEST_NAME)
        for metadata_name in required_metadata:
            if not (work / metadata_name).is_file():
                raise FileNotFoundError(work / metadata_name)
        payload_bytes = (
            sum((args.source_root / relative).stat().st_size for relative in selected)
            if selected
            else None
        )
        finalize(
            args,
            partial,
            work,
            cases,
            selected,
            payload_bytes,
            "reused_successful_pipeline",
        )
        return 0
    if partial.exists():
        raise FileExistsError(partial)
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)

    payload_bytes: int | None = None
    if args.include:
        payload_bytes = write_file_manifest(work, args, selected)
    else:
        write_case_manifest(work, args, cases)
    if not args.include:
        write_readme(work, args, cases, selected, payload_bytes)
    archive_pipeline(args, work, cases, selected, partial)

    integrity_status = "skipped_by_request"
    if not args.skip_full_integrity_test:
        subprocess.run(["zstd", "-t", "--quiet", str(partial)], check=True)
        integrity_status = "passed"
    finalize(
        args,
        partial,
        work,
        cases,
        selected,
        payload_bytes,
        integrity_status,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr, flush=True)
        raise
