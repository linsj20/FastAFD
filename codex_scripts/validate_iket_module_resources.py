#!/usr/bin/env python3
"""Check a replacement library's IKET allocation using saved discovery metadata.

CPU-only: this validates marker/resource allocation, not launch ABI, numerical
correctness or performance. A library must contain exactly one CUDA image;
standalone cubins are accepted explicitly.
"""

import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracker", type=Path, required=True)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--library", type=Path)
    inputs.add_argument("--cubin", type=Path, help="Explicit standalone CUDA image")
    parser.add_argument("--kernel", required=True, help="Substring identifying the module to replace")
    parser.add_argument("--function-contains", action="append", default=[],
                        help="Select library functions containing every substring, instead of only discovered functions")
    parser.add_argument("--replacement-function",
                        help="Explicit symbol for a single discovered kernel; permits a different template in this CPU-only fixture")
    parser.add_argument("--expect-events", nargs='+',
                        help="Require exactly these source marker names in every selected function")
    parser.add_argument("--smodel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    library = (args.library or args.cubin).resolve(strict=True)
    tracker = json.loads(args.tracker.read_text())
    launches = tracker["launches"]
    ids = {x["moduleId"] for x in launches if args.kernel in x["kernelName"]}
    if len(ids) != 1:
        raise ValueError(f"Expected one matching module: {ids}")
    module_id = ids.pop()
    old_names = sorted({x["kernelName"] for x in launches if x["moduleId"] == module_id})
    if args.cubin is not None:
        shutil.copyfile(library, args.output / "replacement.cubin")
    else:
        subprocess.run(["/usr/local/cuda/bin/cuobjdump", "--extract-elf", "all", str(library)],
                       cwd=args.output, check=True, stdout=subprocess.DEVNULL)
    images = list(args.output.glob("*.cubin"))
    if len(images) != 1:
        raise ValueError(f"Expected one extracted CUDA image: {images}")
    resources = subprocess.check_output(
        ["/usr/local/cuda/bin/cuobjdump", "--dump-resource-usage", str(library)], text=True)
    names = re.findall(r"Function\s+(\S+):", resources)
    # Match function name and template parameters, allowing the parameter list
    # to change. Discovery metadata is used only for resource selection here.
    def keys(symbols):
        demangled = subprocess.check_output(["c++filt", *symbols], text=True).splitlines()
        if len(demangled) != len(symbols):
            raise ValueError("Incomplete symbol demangling")
        # Anonymous-namespace parameter types contain parentheses too. Remove
        # that demangler annotation before locating the argument-list opener.
        return [name.replace("(anonymous namespace)", "{anonymous}").split("(", 1)[0]
                for name in demangled]
    new_by_key = {}
    for name, key in zip(names, keys(names)):
        if key in new_by_key:
            raise ValueError(f"Ambiguous replacement function: {key}")
        new_by_key[key] = name
    if args.replacement_function is not None:
        if len(old_names) != 1 or args.replacement_function not in names:
            raise ValueError("Explicit replacement requires one discovered kernel and an existing symbol")
        mapping = {old_names[0]: args.replacement_function}
    else:
        mapping = {name: new_by_key[key] for name, key in zip(old_names, keys(old_names))}

    def replace(value):
        if isinstance(value, str):
            return mapping.get(value, value)
        if isinstance(value, list):
            return [replace(item) for item in value]
        if isinstance(value, dict):
            return {mapping.get(key, key): replace(item) for key, item in value.items()}
        return value

    tracker = replace(tracker)
    modules = [m for m in tracker["modules"] if m["moduleId"] == module_id]
    if len(modules) != 1:
        raise ValueError("Missing replacement module metadata")
    modules[0]["image"] = str(images[0].resolve())
    modules[0]["size"] = images[0].stat().st_size
    # Isolate this module so no unrelated kernel is reanalyzed or mistaken for
    # evidence that the replacement library can be instrumented.
    tracker["modules"] = modules
    tracker["launches"] = [x for x in tracker["launches"] if x["moduleId"] == module_id]
    tracker["graphLaunches"] = {key: [x for x in rows if x["moduleId"] == module_id]
                               for key, rows in tracker["graphLaunches"].items()}
    fixture = args.output / "tracker.json"
    fixture.write_text(json.dumps(tracker) + "\n")
    sys.path.insert(0, str(args.smodel))
    from iket.profiler import TrackerRecord, module_select_resource
    record = TrackerRecord.parse_json(fixture)
    targets = record.make_iket_modules()
    if set(targets) != {module_id}:
        raise ValueError(f"Unexpected analyzed modules: {set(targets)}")
    module = targets[module_id]
    if args.function_contains:
        selected = {name for name in names if all(part in name for part in args.function_contains)}
        if not selected:
            raise ValueError("No matching library functions")
        module.functions = selected
    event_counts = {}
    for name in module.functions:
        points = module.iket_cubin_context.get_kernel_instrument_points(name)
        if not points:
            raise ValueError(f"Missing instrumentation points: {name}")
        events = [module.events[p.evt_attr.event_id].name for p in points]
        if args.expect_events and set(events) != set(args.expect_events):
            raise ValueError(f"Source markers differ: {name}: {events}")
        event_counts[name] = {event: events.count(event) for event in sorted(set(events))}
    config = module_select_resource(module, 256, None)
    (args.output / "module_config.json").write_text(json.dumps(config.to_dict(), indent=2) + "\n")
    result = dict(status="PASS_OFFLINE_RESOURCE_SELECTION", library=str(library),
                  tracker=str(args.tracker), functions=sorted(module.functions), discovery_mapping=mapping,
                  source_marker_counts=event_counts,
                  scope="CPU-only marker/resource allocation; no GPU, numerical, ABI or timing claim")
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
