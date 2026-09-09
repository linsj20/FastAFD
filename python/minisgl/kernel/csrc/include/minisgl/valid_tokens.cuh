#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace minisgl {

struct TokenValidity {
  bool valid;
  int compact_index;
  int total;
};

// CPU-prepared metadata is [total, compact_index_for_each_row], with -1 for
// padding. This moves source lookup/division and prefix arithmetic off the
// GPU. A scalar retains the original contiguous-prefix contract; an empty
// tensor means all rows are real. Normal routing only needs the row entry;
// perfect balancing also reads total, with no change to its expert sequence.
__device__ __forceinline__ TokenValidity token_validity(
    int token, int num_tokens, const int64_t* counts, int count_size) {
  if (count_size == 0)
    return {token < num_tokens, token, num_tokens};
  if (count_size == 1) {
    const int total = min(static_cast<int>(counts[0]), num_tokens);
    return {token < total, token, total};
  }
  if (token >= num_tokens)
    return {false, 0, 0};
  const int compact_index = static_cast<int>(counts[token + 1]);
  return {compact_index >= 0, compact_index, static_cast<int>(counts[0])};
}

}  // namespace minisgl
