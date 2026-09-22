#!/usr/bin/env python3
"""Create a fresh header overlay, owning every directory along replaced paths.

Unchanged entries link to the read-only base. Replacement leaves are copied;
existing outputs are rejected so a live overlay cannot be overwritten.
"""

import argparse
import json
from pathlib import Path
import shutil


def prepare(base, output, replacements):
    base = base.resolve(strict=True)
    if not base.is_dir() or output.exists() or output.is_symlink():
        raise ValueError('Require a directory base and a new output path')
    if output.resolve().is_relative_to(base):
        raise ValueError('Output must be outside the read-only base')
    for relative, source in replacements.items():
        if relative.is_absolute() or not relative.parts or '..' in relative.parts:
            raise ValueError(f'Invalid replacement path: {relative}')
        if not (base / relative).is_file() or not source.is_file():
            raise ValueError(f'Replacement must name existing files: {relative}')

    def populate(directory):
        destination = output / directory
        destination.mkdir()
        for entry in (base / directory).iterdir():
            relative = directory / entry.name
            target = output / relative
            if relative in replacements:
                shutil.copyfile(replacements[relative], target)
            elif any(relative in path.parents for path in replacements):
                if not entry.is_dir():
                    raise ValueError(f'Expected parent directory: {entry}')
                populate(relative)
            else:
                target.symlink_to(entry, target_is_directory=entry.is_dir())

    output.parent.mkdir(parents=True, exist_ok=True)
    populate(Path())
    return dict(base=str(base), output=str(output),
                replacements={str(k): str(v) for k, v in replacements.items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--replace', action='append', required=True, metavar='RELATIVE=FILE')
    args = parser.parse_args()
    replacements = {}
    for item in args.replace:
        relative, source = map(Path, item.split('=', 1))
        if relative in replacements:
            raise ValueError(f'Duplicate replacement: {relative}')
        replacements[relative] = source.resolve(strict=True)
    print(json.dumps(prepare(args.base, args.output, replacements)))


if __name__ == '__main__':
    main()
