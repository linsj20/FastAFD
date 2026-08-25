from __future__ import annotations

from dataclasses import dataclass


def fmha_epoch(
    *,
    step_id: int,
    layer_id: int,
    microbatch_id: int,
    num_layers: int,
    num_microbatches: int,
) -> int:
    """Return the globally ordered FMHA round for one layer/microbatch."""

    if num_layers < 1 or num_microbatches < 1:
        raise ValueError(
            "FMHA epoch requires positive layer and microbatch counts: "
            f"layers={num_layers} microbatches={num_microbatches}"
        )
    if not 0 <= layer_id < num_layers:
        raise ValueError(f"FMHA layer {layer_id} is outside [0, {num_layers})")
    if not 0 <= microbatch_id < num_microbatches:
        raise ValueError(
            f"FMHA microbatch {microbatch_id} is outside [0, {num_microbatches})"
        )
    return (
        int(step_id) * int(num_layers) * int(num_microbatches)
        + int(layer_id) * int(num_microbatches)
        + int(microbatch_id)
    )


def validate_decode_microbatch_token_counts(
    *,
    microbatch_real_token_counts: tuple[int, ...],
    microbatch_token_offsets: tuple[int, ...],
) -> tuple[int, ...]:
    """Validate per-lane real-token counts for a padded decode graph."""

    counts = tuple(int(value) for value in microbatch_real_token_counts)
    offsets = tuple(int(value) for value in microbatch_token_offsets)
    if len(offsets) != len(counts) + 1:
        raise ValueError(
            "decode graph requires one token span per microbatch: "
            f"counts={counts} offsets={offsets}"
        )
    if not offsets or offsets[0] != 0:
        raise ValueError(
            "decode graph token offsets must start at zero: "
            f"offsets={offsets}"
        )
    if any(end < start for start, end in zip(offsets, offsets[1:])):
        raise ValueError(
            "decode graph token offsets must be nondecreasing: "
            f"offsets={offsets}"
        )

    for microbatch, real_count in enumerate(counts):
        start, end = offsets[microbatch], offsets[microbatch + 1]
        span = end - start
        if not 0 <= real_count <= span:
            raise ValueError(
                "decode real-token count exceeds its graph span: "
                f"microbatch={microbatch} count={real_count} span={span}"
            )
    return counts


def build_moe_tp_column_groups(
    *,
    mlp_dp_size: int,
    mlp_tp_size: int,
    ep_size: int,
) -> tuple[tuple[tuple[int, int], ...], ...]:
    """Return ordered ``(dp_rank, tp_rank)`` DeepEP groups.

    DeepEP owns one dense-TP column: full-world EP communicates between DP
    replicas at a fixed TP rank, while per-replica EP has one singleton group
    for every model worker.  Dense TP all-reduce combines the columns.
    """

    for name, value in (
        ("mlp_dp_size", mlp_dp_size),
        ("mlp_tp_size", mlp_tp_size),
        ("ep_size", ep_size),
    ):
        if value < 1:
            raise ValueError(f"afd MoE groups require {name} >= 1, got {value}")
    mlp_world_size = mlp_dp_size * mlp_tp_size
    if ep_size == 1:
        return ()
    if ep_size == mlp_tp_size:
        return tuple(
            ((dp_rank, tp_rank),)
            for dp_rank in range(mlp_dp_size)
            for tp_rank in range(mlp_tp_size)
        )
    if ep_size == mlp_world_size:
        return tuple(
            tuple((dp_rank, tp_rank) for dp_rank in range(mlp_dp_size))
            for tp_rank in range(mlp_tp_size)
        )
    raise ValueError(
        "afd MoE groups require ep_size in "
        "{1, mlp_tp_size, mlp_dp_size*mlp_tp_size}: "
        f"ep_size={ep_size} mlp_tp_size={mlp_tp_size} "
        f"mlp_world_size={mlp_world_size}"
    )


@dataclass(frozen=True)
class AfdHeadSlice:
    """One immutable TP edge, expressed in global and rank-local heads."""

    source_tp_rank: int
    destination_tp_rank: int
    global_head_start: int
    source_head_start: int
    destination_head_start: int
    head_count: int


@dataclass(frozen=True)
class AfdTpSliceTable:
    """Fixed Q/K/V/O TP bridge for one mutually-divisible TP pair."""

    attn_tp_size: int
    model_tp_size: int
    num_qo_heads: int
    num_kv_heads: int
    q_slices: tuple[AfdHeadSlice, ...]
    kv_slices: tuple[AfdHeadSlice, ...]
    o_slices: tuple[AfdHeadSlice, ...]
    attn_local_q_heads: int
    model_local_q_heads: int
    attn_local_kv_heads: int
    model_local_kv_heads: int

    @classmethod
    def build(
        cls,
        *,
        attn_tp_size: int,
        model_tp_size: int,
        num_qo_heads: int,
        num_kv_heads: int,
    ) -> AfdTpSliceTable:
        for name, value in (
            ("attn_tp_size", attn_tp_size),
            ("model_tp_size", model_tp_size),
            ("num_qo_heads", num_qo_heads),
            ("num_kv_heads", num_kv_heads),
        ):
            if value < 1:
                raise ValueError(f"afd TP bridge {name} must be >= 1, got {value}")
        small, large = sorted((attn_tp_size, model_tp_size))
        if large % small:
            raise ValueError(
                "afd TP bridge sizes must be mutually divisible: "
                f"attn_tp_size={attn_tp_size} model_tp_size={model_tp_size}"
            )
        if num_qo_heads % attn_tp_size or num_qo_heads % model_tp_size:
            raise ValueError(
                "afd TP bridge requires Q/O heads divisible by both TP sizes: "
                f"num_qo_heads={num_qo_heads} attn_tp_size={attn_tp_size} "
                f"model_tp_size={model_tp_size}"
            )

        attn_q = _partitioned_head_owners(num_qo_heads, attn_tp_size, replicate=False)
        model_q = _partitioned_head_owners(num_qo_heads, model_tp_size, replicate=False)
        attn_kv = _partitioned_head_owners(num_kv_heads, attn_tp_size, replicate=True)
        model_kv = _partitioned_head_owners(num_kv_heads, model_tp_size, replicate=True)

        q_slices = _intersection_slices(model_q, attn_q)
        o_slices = _intersection_slices(attn_q, model_q)
        kv_slices = _replicated_writer_slices(model_kv, attn_kv, num_kv_heads)
        table = cls(
            attn_tp_size=attn_tp_size,
            model_tp_size=model_tp_size,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            q_slices=q_slices,
            kv_slices=kv_slices,
            o_slices=o_slices,
            attn_local_q_heads=num_qo_heads // attn_tp_size,
            model_local_q_heads=num_qo_heads // model_tp_size,
            attn_local_kv_heads=len(attn_kv[0]),
            model_local_kv_heads=len(model_kv[0]),
        )
        table.validate_exact_coverage()
        return table

    def validate_exact_coverage(self) -> None:
        _validate_destination_coverage(
            "Q",
            self.q_slices,
            self.attn_tp_size,
            self.attn_local_q_heads,
        )
        _validate_destination_coverage(
            "K/V",
            self.kv_slices,
            self.attn_tp_size,
            self.attn_local_kv_heads,
        )
        _validate_destination_coverage(
            "O",
            self.o_slices,
            self.model_tp_size,
            self.model_local_q_heads,
        )
        _validate_source_bounds(
            "Q", self.q_slices, self.model_tp_size, self.model_local_q_heads
        )
        _validate_source_bounds(
            "K/V", self.kv_slices, self.model_tp_size, self.model_local_kv_heads
        )
        _validate_source_bounds(
            "O", self.o_slices, self.attn_tp_size, self.attn_local_q_heads
        )


def _partitioned_head_owners(
    num_heads: int,
    tp_size: int,
    *,
    replicate: bool,
) -> tuple[tuple[int, ...], ...]:
    if tp_size <= num_heads:
        if num_heads % tp_size:
            raise ValueError(
                f"num_heads={num_heads} must be divisible by tp_size={tp_size}"
            )
        local_heads = num_heads // tp_size
        return tuple(
            tuple(range(rank * local_heads, (rank + 1) * local_heads))
            for rank in range(tp_size)
        )
    if not replicate or tp_size % num_heads:
        raise ValueError(
            f"tp_size={tp_size} cannot shard num_heads={num_heads} exactly"
        )
    replicas = tp_size // num_heads
    return tuple((rank // replicas,) for rank in range(tp_size))


def _intersection_slices(
    source_heads: tuple[tuple[int, ...], ...],
    destination_heads: tuple[tuple[int, ...], ...],
) -> tuple[AfdHeadSlice, ...]:
    slices: list[AfdHeadSlice] = []
    for source_rank, source in enumerate(source_heads):
        source_index = {head: index for index, head in enumerate(source)}
        for destination_rank, destination in enumerate(destination_heads):
            destination_index = {head: index for index, head in enumerate(destination)}
            common = sorted(source_index.keys() & destination_index.keys())
            for head in common:
                slices.append(
                    AfdHeadSlice(
                        source_tp_rank=source_rank,
                        destination_tp_rank=destination_rank,
                        global_head_start=head,
                        source_head_start=source_index[head],
                        destination_head_start=destination_index[head],
                        head_count=1,
                    )
                )
    return _merge_adjacent_slices(slices)


def _replicated_writer_slices(
    source_heads: tuple[tuple[int, ...], ...],
    destination_heads: tuple[tuple[int, ...], ...],
    num_heads: int,
) -> tuple[AfdHeadSlice, ...]:
    source_owners = _owners_by_head(source_heads, num_heads)
    destination_owners = _owners_by_head(destination_heads, num_heads)
    slices: list[AfdHeadSlice] = []
    for head in range(num_heads):
        sources = source_owners[head]
        destinations = destination_owners[head]
        for replica_index, (destination_rank, destination_local) in enumerate(destinations):
            source_rank, source_local = sources[replica_index % len(sources)]
            slices.append(
                AfdHeadSlice(
                    source_tp_rank=source_rank,
                    destination_tp_rank=destination_rank,
                    global_head_start=head,
                    source_head_start=source_local,
                    destination_head_start=destination_local,
                    head_count=1,
                )
            )
    return _merge_adjacent_slices(slices)


def _owners_by_head(
    rank_heads: tuple[tuple[int, ...], ...],
    num_heads: int,
) -> tuple[tuple[tuple[int, int], ...], ...]:
    owners: list[list[tuple[int, int]]] = [[] for _ in range(num_heads)]
    for rank, heads in enumerate(rank_heads):
        for local_index, head in enumerate(heads):
            owners[head].append((rank, local_index))
    if any(not head_owners for head_owners in owners):
        raise ValueError("afd TP bridge leaves a global head without an owner")
    return tuple(tuple(head_owners) for head_owners in owners)


def _merge_adjacent_slices(slices: list[AfdHeadSlice]) -> tuple[AfdHeadSlice, ...]:
    merged: list[AfdHeadSlice] = []
    for head_slice in slices:
        previous = merged[-1] if merged else None
        if (
            previous is not None
            and previous.source_tp_rank == head_slice.source_tp_rank
            and previous.destination_tp_rank == head_slice.destination_tp_rank
            and previous.global_head_start + previous.head_count
            == head_slice.global_head_start
            and previous.source_head_start + previous.head_count
            == head_slice.source_head_start
            and previous.destination_head_start + previous.head_count
            == head_slice.destination_head_start
        ):
            merged[-1] = AfdHeadSlice(
                source_tp_rank=previous.source_tp_rank,
                destination_tp_rank=previous.destination_tp_rank,
                global_head_start=previous.global_head_start,
                source_head_start=previous.source_head_start,
                destination_head_start=previous.destination_head_start,
                head_count=previous.head_count + head_slice.head_count,
            )
        else:
            merged.append(head_slice)
    return tuple(merged)


def _validate_destination_coverage(
    name: str,
    slices: tuple[AfdHeadSlice, ...],
    tp_size: int,
    local_heads: int,
) -> None:
    coverage = [[0] * local_heads for _ in range(tp_size)]
    for head_slice in slices:
        if not 0 <= head_slice.destination_tp_rank < tp_size:
            raise ValueError(f"afd {name} slice has invalid destination rank")
        end = head_slice.destination_head_start + head_slice.head_count
        if head_slice.destination_head_start < 0 or end > local_heads:
            raise ValueError(f"afd {name} slice exceeds destination head bounds")
        for head in range(head_slice.destination_head_start, end):
            coverage[head_slice.destination_tp_rank][head] += 1
    for rank, rank_coverage in enumerate(coverage):
        if any(count != 1 for count in rank_coverage):
            raise ValueError(
                f"afd {name} destination rank {rank} does not have exact writer coverage: "
                f"{rank_coverage}"
            )


def _validate_source_bounds(
    name: str,
    slices: tuple[AfdHeadSlice, ...],
    tp_size: int,
    local_heads: int,
) -> None:
    for head_slice in slices:
        if not 0 <= head_slice.source_tp_rank < tp_size:
            raise ValueError(f"afd {name} slice has invalid source rank")
        end = head_slice.source_head_start + head_slice.head_count
        if head_slice.source_head_start < 0 or end > local_heads:
            raise ValueError(f"afd {name} slice exceeds source head bounds")


__all__ = [
    "AfdHeadSlice",
    "AfdTpSliceTable",
    "build_moe_tp_column_groups",
    "fmha_epoch",
    "validate_decode_microbatch_token_counts",
]
