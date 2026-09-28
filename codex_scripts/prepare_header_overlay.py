#!/usr/bin/env python3
"""Create a fresh header overlay, owning every directory along replaced paths.

Unchanged entries link to the read-only base. Replacement leaves are copied;
existing outputs are rejected so a live overlay cannot be overwritten.
"""

import argparse
import json
import re
from pathlib import Path
import shutil


def prepare(base, output, replacements, own_native=False):
    base = base.resolve(strict=True)
    if not base.is_dir() or output.exists() or output.is_symlink():
        raise ValueError('Require a directory base and a new output path')
    if output.resolve().is_relative_to(base):
        raise ValueError('Output must be outside the read-only base')
    replacements = dict(replacements)
    if own_native:
        # Native quoted includes and the Python binding resolve symlinks. Copy
        # the whole small host tree so they cannot escape to the base source.
        def own_tree(relative, ancestors=()):
            entry = base / relative
            if entry.is_dir():
                resolved = entry.resolve(strict=True)
                if resolved in ancestors:
                    raise ValueError(f'Cyclic native source directory: {relative}')
                for child in sorted(entry.iterdir()):
                    own_tree(relative / child.name, (*ancestors, resolved))
            elif entry.is_file():
                replacements.setdefault(relative, entry)
            else:
                raise ValueError(f'Invalid native source entry: {relative}')
        own_tree(Path('python/minisgl/kernel/csrc/deepgemm/csrc'))
        own_tree(Path('python/minisgl/kernel/deepgemm.py'))
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
                own_native=own_native,
                replacements={str(k): str(v) for k, v in replacements.items()})


EVENTS = ("MegaMmaProbeFC1", "MegaMmaProbeFC2", "MegaMmaReuseWait",
          "MegaMmaReuseReady", "MegaMmaOperandWait", "MegaMmaOperandReady")
CORE_EVENTS = {"MegaDispatchStart", "MegaDispatchEnd", "MegaMmaIssue",
               "MegaComputeEnd", "MegaCombineStart", "MegaCombineEnd"}


def keep_core_markers(source, declared):
    """Reserve the 5-bit event space for core boundaries and the wait probes."""
    for name in declared - CORE_EVENTS:
        source = source.replace(f'CREATE_IKET_MARKER({name});', '')
        calls = re.compile(r'\b(?:MEGAMOE_IKET_(?:MARK|DETAIL_MARK|ONCE)|'
                           r'GENPHASE_MARK(?:_WARP|_ALL)?)\(\s*' + name + r'\b')
        while match := calls.search(source):
            start = source.index('(', match.start())
            depth, end = 1, start + 1
            while depth and end < len(source):
                depth += (source[end] == '(') - (source[end] == ')')
                end += 1
            if depth:
                raise ValueError(f'Unbalanced marker invocation: {name}')
            invocation = source[match.start():end]
            if any(c in invocation for c in ('"', "'")):
                raise ValueError('Quoted marker argument requires explicit parsing')
            source = source[:match.start()] + '((void)0)' + '\n' * invocation.count('\n') + source[end:]
    return source


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError(f"Expected one source anchor: {old!r}")
    return source.replace(old, new, 1)


def instrument(source, all_push):
    source = replace_once(source, "CREATE_IKET_MARKER(MegaMmaIssue);",
        "CREATE_IKET_MARKER(MegaMmaIssue);\n" + "\n".join(
            f"CREATE_IKET_MARKER({event});" for event in EVENTS))
    anchor = "            uint32_t current_iter_idx = 0;"
    source = replace_once(source, anchor, anchor +
        "\n            MEGAMOE_IKET_FIRST(probe_first_fc2);")
    anchor = "                tmem_empty_barriers[accum_stage_idx]->wait(accum_phase ^ 1);"
    source = replace_once(source, anchor, """#ifdef IKET_BUILD_ENABLED
                const bool probe_tile = current_iter_idx == 1 ||
                    (block_phase == sched::BlockPhase::Linear2 && probe_first_fc2);
#endif
                MEGAMOE_IKET_ONCE(MegaMmaProbeFC1, probe_tile && block_phase == sched::BlockPhase::Linear1);
                MEGAMOE_IKET_ONCE(MegaMmaProbeFC2, probe_tile && block_phase == sched::BlockPhase::Linear2);
                MEGAMOE_IKET_ONCE(MegaMmaReuseWait, probe_tile);
""" + anchor + "\n                MEGAMOE_IKET_ONCE(MegaMmaReuseReady, probe_tile);")
    anchor = "                    full_barriers[stage_idx]->wait(phase);"
    source = replace_once(source, anchor,
        "                    MEGAMOE_IKET_ONCE(MegaMmaOperandWait, probe_tile);\n" + anchor)
    anchor = ("                        weight_full_barriers[weight_stage_idx]->wait(weight_phase);") if all_push else (
              "                    full_barriers[stage_idx]->wait(phase);")
    source = replace_once(source, anchor, anchor +
        "\n                    MEGAMOE_IKET_ONCE(MegaMmaOperandReady, probe_tile);")
    anchor = "                    empty_barrier_arrive(k_block_idx == num_k_blocks - 1);\n                }"
    source = replace_once(source, anchor, anchor + """
                if (block_phase == sched::BlockPhase::Linear2)
                    MEGAMOE_IKET_DONE(probe_first_fc2);""")
    return source



def prepare_mma_probe(base, output):
    root = base.resolve(strict=True)
    changed = {}
    for all_push in (False, True):
        name = "sm100_fp8_fp4_mega_moe" + ("_all_push" if all_push else "") + ".cuh"
        path = Path("include/deep_gemm/impls") / name
        dispatch = Path('include/deep_gemm/impls') / (
            'mega_moe_all_push_dispatch.cuh' if all_push else 'mega_moe_dispatch.cuh')
        sources = {p: (root / p).read_text() for p in (path, dispatch)}
        declared = set(re.findall(r'CREATE_IKET_MARKER\((\w+)\);',
                                  '\n'.join(sources.values())))
        changed[path] = instrument(keep_core_markers(sources[path], declared), all_push)
        changed[dispatch] = keep_core_markers(sources[dispatch], declared)
    shutil.copytree(root, output, symlinks=False)
    for path, source in changed.items():
        (output / path).write_text(source)
    (output / "probe.json").write_text(json.dumps(dict(
        source=str(root), events=EVENTS, retained_core_events=sorted(CORE_EVENTS),
        omitted_events="Non-core metadata, readiness and local-reduction probes; no detailed-phase claims",
        sampled_tiles="first FC1 and first FC2 per leader CTA",
        required_disabled_features=["all_push_weight_stages"],
        interpretation="Instrumented wait intervals, not unperturbed stall cycles. Require exact phase/K-count coverage, resource gates, ordinary instruction preservation and ordinary/profile bias."), indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--own-native', action='store_true',
                        help='Copy the DeepGEMM binding and native C++ tree for host/configuration changes')
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--replace', action='append', metavar='RELATIVE=FILE')
    mode.add_argument('--mma-wait-probe', action='store_true',
                      help='Create isolated first-FC1/FC2 operand and accumulator wait probes')
    args = parser.parse_args()
    if args.mma_wait_probe:
        if args.own_native:
            parser.error('--own-native requires explicit --replace entries')
        prepare_mma_probe(args.base, args.output)
        return
    replacements = {}
    for item in args.replace:
        relative, source = map(Path, item.split('=', 1))
        if relative in replacements:
            raise ValueError(f'Duplicate replacement: {relative}')
        replacements[relative] = source.resolve(strict=True)
    print(json.dumps(prepare(args.base, args.output, replacements, args.own_native)))


if __name__ == '__main__':
    main()
