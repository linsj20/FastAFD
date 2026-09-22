"""Opt-in, graph-safe collective barrier for standalone decomposition only."""

import functools
import os
from pathlib import Path


@functools.cache
def build():
    from minisgl.kernel.utils import load_jit
    return load_jit(
        "decomposition_barrier",
        cuda_files=[str(Path(__file__).with_suffix('.cu').resolve())],
        cuda_wrappers=[('launch', 'DecompositionBarrier::run')],
    )


class CollectiveBarrier:
    def __init__(self, group, device):
        if os.environ.get('MEGAMOE_DECOMPOSITION', '0') != '1':
            raise ValueError('Collective preparation barrier is decomposition-only')
        import torch
        import torch.distributed as dist
        import torch.distributed._symmetric_memory as symm_mem
        self.rank, self.ranks = dist.get_rank(group), dist.get_world_size(group)
        if not 1 <= self.ranks <= 64:
            raise ValueError('Collective preparation supports 1..64 ranks')
        self.module = build()
        self.flags = symm_mem.empty(self.ranks + 1, dtype=torch.int64, device=device)
        self.handle = symm_mem.rendezvous(self.flags, group=group)
        self.peers = torch.tensor(self.handle.buffer_ptrs, dtype=torch.int64, device=device)
        self.flags.zero_()
        torch.cuda.synchronize()
        dist.barrier(group=group)

    def __call__(self):
        self.module.launch(self.flags, self.peers, self.rank, self.ranks)
