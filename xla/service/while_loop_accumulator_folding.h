/* Copyright 2026 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef XLA_SERVICE_WHILE_LOOP_ACCUMULATOR_FOLDING_H_
#define XLA_SERVICE_WHILE_LOOP_ACCUMULATOR_FOLDING_H_

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"

namespace xla {

// Folds the addition of a scan's stacked output into a running accumulator
// into the scan itself, so that the scan writes into the accumulator in place.
//
// This is the pattern produced by gradient accumulation over a scanned layer
// stack (e.g. `acc + grad(loss)(params)` inside a `jax.lax.scan` over
// microbatches, where the backward pass of a `jax.lax.scan` over layers emits
// the stacked parameter gradients):
//
//   buf = while(..., init, ...)     // body: buf' = dus(buf, g(i), idx(i))
//   result = add(acc, f(get-tuple-element(buf), index=k))
//
// =>
//
//   buf = while(..., acc, ...)
//     // body: s = ds(buf, idx'(i)); buf' = dus(buf, add(s, f'(g(i))), idx'(i))
//   result = get-tuple-element(buf), index=k
//
// where `f` is a chain of convert, copy, transpose and reshape instructions
// that maps the loop buffer onto the accumulator's shape, `f'` is the same
// chain applied to the per-iteration update, and `idx'` is the update's
// position in the accumulator. This removes the loop buffer (typically
// zero-initialized by jax.lax.scan) and the full-size add, which otherwise
// keep a second copy of the accumulator's size alive for the duration of the
// loop.
//
// The rewrite is applied only when it is bitwise exact, which requires that
// the loop writes every element of the buffer exactly once: the buffer slot is
// only used by a single dynamic-update-slice that covers all dimensions but
// one, the loop has a static trip count, and the update's offsets along that
// dimension (evaluated for every iteration) tile it without overlap. Each
// element of the result is then `acc[x] + f(g)[x]` before and after the
// rewrite, whatever `init` is. Slots written more than once (e.g. accumulated
// gradients of a loop-invariant weight) are not rewritten, since that would
// reassociate the additions. Copies that keep the sharding, as Shardy's export
// leaves them, may sit around the dynamic-update-slice.
//
// The rewrite only saves memory if the accumulator dies at the add, so the
// pass also requires the accumulator (looking through copies) to have no other
// users. The offsets into the accumulator are computed in the loop's offset
// type, which must be able to index the accumulator's dimension.
//
// The pass is meant to run on global shapes after sharding propagation and
// before SPMD partitioning: the partitioner then inserts the same collectives
// for the per-iteration update that it inserts today. To avoid resharding the
// loop buffer on every iteration, the pass only fires when the buffer's
// sharding, mapped through `f`, matches the accumulator's sharding, and when
// the dynamic dimension is not partitioned.
//
// The GPU compiler runs this pass by default (in the SPMD pipeline, or in the
// sharding-removal pipeline for a single partition);
// --xla_gpu_enable_while_loop_accumulator_folding=false disables it.
class WhileLoopAccumulatorFolding : public HloModulePass {
 public:
  absl::string_view name() const override {
    return "while-loop-accumulator-folding";
  }

 protected:
  absl::StatusOr<bool> RunImpl(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;
};

}  // namespace xla

#endif  // XLA_SERVICE_WHILE_LOOP_ACCUMULATOR_FOLDING_H_
