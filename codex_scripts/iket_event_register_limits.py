"""Explicit per-event register bounds for a scoped IKET profiler process.

Set FASTAFD_IKET_EVENT_REGISTER_LIMITS to a JSON name -> exclusive-register-limit
mapping. This tightens SModel's existing liveness-aware allocation; it never picks
a register directly or changes the workload. Bounds require manual role review.
"""

import json
import os


def install():
    limits = json.loads(os.environ["FASTAFD_IKET_EVENT_REGISTER_LIMITS"])
    if not isinstance(limits, dict) or any(
        not isinstance(name, str) or type(limit) is not int or not 0 < limit < 256
        for name, limit in limits.items()
    ):
        raise ValueError("Expected event names mapped to exclusive register limits")
    import iket.profiler as profiler
    from iket.profiler import utils
    from smodel.tools.iket.resource_selection import RegisterLimiter

    original_limit = RegisterLimiter.get_regs_limit
    original_select = utils.module_select_resource
    bounds = {}

    def get_limit(self, block, pc):
        original = original_limit(self, block, pc)
        return min(original, bounds.get((self.func.name, pc), original))

    def select(module, *args, **kwargs):
        bounds.clear()
        found = set()
        for func in module.functions:
            for point in module.iket_cubin_context.get_kernel_instrument_points(func):
                name = module.events[point.evt_attr.event_id].name
                if name in limits:
                    found.add(name)
                    for pc in range(point.offset_start, point.offset_end + 1, 16):
                        bounds[func, pc] = limits[name]
        result = original_select(module, *args, **kwargs)
        print("IKET_EXPLICIT_REGISTER_LIMITS " + json.dumps({
            "events": {name: limits[name] for name in sorted(found)},
            "points": len(bounds),
        }), flush=True)
        return result

    RegisterLimiter.get_regs_limit = get_limit
    utils.module_select_resource = select
    profiler.module_select_resource = select


if __name__ == "__main__":
    install()
