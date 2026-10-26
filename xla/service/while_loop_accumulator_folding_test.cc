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

#include "xla/service/while_loop_accumulator_folding.h"

#include <memory>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/str_replace.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/evaluator/hlo_evaluator.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/utils/hlo_matchers.h"
#include "xla/literal.h"
#include "xla/service/hlo_verifier.h"
#include "xla/tests/test_utils.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace {

namespace op = xla::testing::opcode_matchers;

class WhileLoopAccumulatorFoldingTest : public HloHardwareIndependentTestBase {
 protected:
  // Runs the pass and, if it changed the module, checks that the result is
  // bitwise identical to the original on random inputs.
  bool RunAndCheck(absl::string_view hlo) {
    std::unique_ptr<HloModule> module =
        ParseAndReturnVerifiedModule(hlo).value();
    std::unique_ptr<HloModule> reference = module->Clone();
    bool changed =
        RunHloPass(WhileLoopAccumulatorFolding(), module.get()).value();
    if (!changed) {
      return false;
    }
    EXPECT_TRUE(HloVerifier(/*layout_sensitive=*/false,
                            /*allow_mixed_precision=*/true)
                    .Run(module.get())
                    .status()
                    .ok());
    std::vector<Literal> args = MakeFakeArguments(reference.get()).value();
    std::vector<const Literal*> arg_ptrs;
    for (const Literal& arg : args) {
      arg_ptrs.push_back(&arg);
    }
    Literal want = HloEvaluator()
                       .Evaluate(*reference->entry_computation(), arg_ptrs)
                       .value();
    Literal got =
        HloEvaluator().Evaluate(*module->entry_computation(), arg_ptrs).value();
    EXPECT_EQ(want, got);
    module_ = std::move(module);
    return true;
  }

  std::unique_ptr<HloModule> module_;
};

// A reverse scan (as emitted for the backward pass of jax.lax.scan) writing
// one row per iteration, whose output is added to an accumulator.
constexpr absl::string_view kBasic = R"(
HloModule m

body {
  p = (s32[], f32[4,3], f32[4,3]) parameter(0)
  i = s32[] get-tuple-element(p), index=0
  buf = f32[4,3] get-tuple-element(p), index=1
  xs = f32[4,3] get-tuple-element(p), index=2
  c3 = s32[] constant(3)
  idx = s32[] subtract(c3, i)
  z = s32[] constant(0)
  x = f32[1,3] dynamic-slice(xs, idx, z), dynamic_slice_sizes={1,3}
  g = f32[1,3] multiply(x, x)
  buf2 = f32[4,3] dynamic-update-slice(buf, g, idx, z)
  one = s32[] constant(1)
  i2 = s32[] add(i, one)
  ROOT t = (s32[], f32[4,3], f32[4,3]) tuple(i2, buf2, xs)
}

cond {
  p = (s32[], f32[4,3], f32[4,3]) parameter(0)
  i = s32[] get-tuple-element(p), index=0
  n = s32[] constant(4)
  ROOT lt = pred[] compare(i, n), direction=LT
}

ENTRY e {
  acc = f32[4,3] parameter(0)
  xs = f32[4,3] parameter(1)
  zero = f32[] constant(0)
  b = f32[4,3] broadcast(zero), dimensions={}
  c0 = s32[] constant(0)
  init = (s32[], f32[4,3], f32[4,3]) tuple(c0, b, xs)
  w = (s32[], f32[4,3], f32[4,3]) while(init), condition=cond, body=body
  out = f32[4,3] get-tuple-element(w), index=1
  ROOT r = f32[4,3] add(acc, out)
}
)";

TEST_F(WhileLoopAccumulatorFoldingTest, FoldsReverseScan) {
  ASSERT_TRUE(RunAndCheck(kBasic));
  HloInstruction* root = module_->entry_computation()->root_instruction();
  EXPECT_THAT(root, op::GetTupleElement(op::While(op::Tuple(
                        op::Constant(), op::Parameter(0), op::Parameter(1)))));
  HloInstruction* body_root =
      root->operand(0)->while_body()->root_instruction();
  EXPECT_THAT(
      body_root->operand(1),
      op::DynamicUpdateSlice(op::GetTupleElement(),
                             op::Add(op::DynamicSlice(), op::Multiply()),
                             op::Subtract(), op::Constant()));
}

TEST_F(WhileLoopAccumulatorFoldingTest, FoldsForwardScanWithOperandsSwapped) {
  std::string hlo = absl::StrReplaceAll(
      kBasic,
      {{"idx = s32[] subtract(c3, i)", "idx = s32[] add(i, z)"},
       {"ROOT r = f32[4,3] add(acc, out)", "ROOT r = f32[4,3] add(out, acc)"}});
  // `z` must be defined before `idx`.
  hlo =
      absl::StrReplaceAll(hlo, {{"  c3 = s32[] constant(3)\n",
                                 "  c3 = s32[] constant(3)\n  z = s32[] "
                                 "constant(0)\n"},
                                {"  z = s32[] constant(0)\n  x = ", "  x = "}});
  ASSERT_TRUE(RunAndCheck(hlo));
  HloInstruction* body_root = module_->entry_computation()
                                  ->root_instruction()
                                  ->operand(0)
                                  ->while_body()
                                  ->root_instruction();
  // The original operand order of the add is kept.
  EXPECT_THAT(
      body_root->operand(1),
      op::DynamicUpdateSlice(op::GetTupleElement(),
                             op::Add(op::Multiply(), op::DynamicSlice()),
                             op::Add(), op::Constant()));
}

// param_scan_axis != 0 and a wider accumulator: the loop buffer is converted
// and transposed before the add.
TEST_F(WhileLoopAccumulatorFoldingTest, FoldsThroughConvertAndTranspose) {
  constexpr absl::string_view hlo = R"(
HloModule m

body {
  p = (s32[], bf16[4,2,3], f32[4,2,3]) parameter(0)
  i = s32[] get-tuple-element(p), index=0
  buf = bf16[4,2,3] get-tuple-element(p), index=1
  xs = f32[4,2,3] get-tuple-element(p), index=2
  c3 = s32[] constant(3)
  idx = s32[] subtract(c3, i)
  z = s32[] constant(0)
  x = f32[1,2,3] dynamic-slice(xs, idx, z, z), dynamic_slice_sizes={1,2,3}
  g = bf16[1,2,3] convert(x)
  buf2 = bf16[4,2,3] dynamic-update-slice(buf, g, idx, z, z)
  one = s32[] constant(1)
  i2 = s32[] add(i, one)
  ROOT t = (s32[], bf16[4,2,3], f32[4,2,3]) tuple(i2, buf2, xs)
}

cond {
  p = (s32[], bf16[4,2,3], f32[4,2,3]) parameter(0)
  i = s32[] get-tuple-element(p), index=0
  n = s32[] constant(4)
  ROOT lt = pred[] compare(i, n), direction=LT
}

ENTRY e {
  acc = f32[2,4,3] parameter(0)
  xs = f32[4,2,3] parameter(1)
  zero = bf16[] constant(0)
  b = bf16[4,2,3] broadcast(zero), dimensions={}
  c0 = s32[] constant(0)
  init = (s32[], bf16[4,2,3], f32[4,2,3]) tuple(c0, b, xs)
  w = (s32[], bf16[4,2,3], f32[4,2,3]) while(init), condition=cond, body=body
  out = bf16[4,2,3] get-tuple-element(w), index=1
  tr = bf16[2,4,3] transpose(out), dimensions={1,0,2}
  cv = f32[2,4,3] convert(tr)
  ROOT r = f32[2,4,3] add(acc, cv)
}
)";
  ASSERT_TRUE(RunAndCheck(hlo));
  HloInstruction* while_instr =
      module_->entry_computation()->root_instruction()->mutable_operand(0);
  EXPECT_EQ(while_instr->shape().tuple_shapes(1).ToString(), "f32[2,4,3]");
  EXPECT_THAT(
      while_instr->while_body()->root_instruction()->operand(1),
      op::DynamicUpdateSlice(
          op::GetTupleElement(),
          op::Add(op::DynamicSlice(op::GetTupleElement(), op::Constant(),
                                   op::Subtract(), op::Constant()),
                  op::Convert(op::Transpose(op::Convert()))),
          op::Constant(), op::Subtract(), op::Constant()));
}

// jax.lax.scan(..., unroll=2): two layers are written per iteration into a
// [L/2, 2, ...] buffer that is reshaped to [L, ...].
TEST_F(WhileLoopAccumulatorFoldingTest, FoldsThroughReshape) {
  constexpr absl::string_view hlo = R"(
HloModule m

body {
  p = (s32[], f32[2,2,3], f32[2,2,3]) parameter(0)
  i = s32[] get-tuple-element(p), index=0
  buf = f32[2,2,3] get-tuple-element(p), index=1
  xs = f32[2,2,3] get-tuple-element(p), index=2
  c1 = s32[] constant(1)
  idx = s32[] subtract(c1, i)
  z = s32[] constant(0)
  x = f32[1,2,3] dynamic-slice(xs, idx, z, z), dynamic_slice_sizes={1,2,3}
  g = f32[1,2,3] exponential(x)
  buf2 = f32[2,2,3] dynamic-update-slice(buf, g, idx, z, z)
  one = s32[] constant(1)
  i2 = s32[] add(i, one)
  ROOT t = (s32[], f32[2,2,3], f32[2,2,3]) tuple(i2, buf2, xs)
}

cond {
  p = (s32[], f32[2,2,3], f32[2,2,3]) parameter(0)
  i = s32[] get-tuple-element(p), index=0
  n = s32[] constant(2)
  ROOT lt = pred[] compare(i, n), direction=LT
}

ENTRY e {
  acc = f32[4,3] parameter(0)
  xs = f32[2,2,3] parameter(1)
  zero = f32[] constant(0)
  b = f32[2,2,3] broadcast(zero), dimensions={}
  c0 = s32[] constant(0)
  init = (s32[], f32[2,2,3], f32[2,2,3]) tuple(c0, b, xs)
  w = (s32[], f32[2,2,3], f32[2,2,3]) while(init), condition=cond, body=body
  out = f32[2,2,3] get-tuple-element(w), index=1
  rs = f32[4,3] reshape(out)
  ROOT r = f32[4,3] add(acc, rs)
}
)";
  ASSERT_TRUE(RunAndCheck(hlo));
  HloInstruction* while_instr =
      module_->entry_computation()->root_instruction()->mutable_operand(0);
  EXPECT_THAT(
      while_instr->while_body()->root_instruction()->operand(1),
      op::DynamicUpdateSlice(
          op::GetTupleElement(),
          op::Add(op::DynamicSlice(), op::Reshape(op::Exp())),
          op::Multiply(op::Subtract(), op::Constant()), op::Constant()));
}

// The microbatch loop of gradient accumulation around the layer loop, with
// shardings as after sharding propagation.
TEST_F(WhileLoopAccumulatorFoldingTest, FoldsInsideOuterLoopWithSharding) {
  constexpr absl::string_view hlo = R"(
HloModule m

inner_body {
  p = (s32[], f32[4,8], f32[4,8]) parameter(0), sharding={{replicated}, {devices=[1,2]<=[2]}, {devices=[1,2]<=[2]}}
  i = s32[] get-tuple-element(p), index=0, sharding={replicated}
  buf = f32[4,8] get-tuple-element(p), index=1, sharding={devices=[1,2]<=[2]}
  xs = f32[4,8] get-tuple-element(p), index=2, sharding={devices=[1,2]<=[2]}
  c3 = s32[] constant(3), sharding={replicated}
  idx = s32[] subtract(c3, i), sharding={replicated}
  z = s32[] constant(0), sharding={replicated}
  x = f32[1,8] dynamic-slice(xs, idx, z), dynamic_slice_sizes={1,8}, sharding={devices=[1,2]<=[2]}
  g = f32[1,8] tanh(x), sharding={devices=[1,2]<=[2]}
  buf2 = f32[4,8] dynamic-update-slice(buf, g, idx, z), sharding={devices=[1,2]<=[2]}
  one = s32[] constant(1), sharding={replicated}
  i2 = s32[] add(i, one), sharding={replicated}
  ROOT t = (s32[], f32[4,8], f32[4,8]) tuple(i2, buf2, xs), sharding={{replicated}, {devices=[1,2]<=[2]}, {devices=[1,2]<=[2]}}
}

inner_cond {
  p = (s32[], f32[4,8], f32[4,8]) parameter(0), sharding={{replicated}, {devices=[1,2]<=[2]}, {devices=[1,2]<=[2]}}
  i = s32[] get-tuple-element(p), index=0, sharding={replicated}
  n = s32[] constant(4), sharding={replicated}
  ROOT lt = pred[] compare(i, n), direction=LT, sharding={replicated}
}

outer_body {
  p = (s32[], f32[4,8], f32[3,4,8]) parameter(0), sharding={{replicated}, {devices=[1,2]<=[2]}, {devices=[1,1,2]<=[2]}}
  j = s32[] get-tuple-element(p), index=0, sharding={replicated}
  acc = f32[4,8] get-tuple-element(p), index=1, sharding={devices=[1,2]<=[2]}
  data = f32[3,4,8] get-tuple-element(p), index=2, sharding={devices=[1,1,2]<=[2]}
  z = s32[] constant(0), sharding={replicated}
  mb = f32[1,4,8] dynamic-slice(data, j, z, z), dynamic_slice_sizes={1,4,8}, sharding={devices=[1,1,2]<=[2]}
  xs = f32[4,8] reshape(mb), sharding={devices=[1,2]<=[2]}
  zero = f32[] constant(0), sharding={replicated}
  b = f32[4,8] broadcast(zero), dimensions={}, sharding={devices=[1,2]<=[2]}
  init = (s32[], f32[4,8], f32[4,8]) tuple(z, b, xs), sharding={{replicated}, {devices=[1,2]<=[2]}, {devices=[1,2]<=[2]}}
  w = (s32[], f32[4,8], f32[4,8]) while(init), condition=inner_cond, body=inner_body, sharding={{replicated}, {devices=[1,2]<=[2]}, {devices=[1,2]<=[2]}}
  out = f32[4,8] get-tuple-element(w), index=1, sharding={devices=[1,2]<=[2]}
  acc2 = f32[4,8] add(out, acc), sharding={devices=[1,2]<=[2]}
  one = s32[] constant(1), sharding={replicated}
  j2 = s32[] add(j, one), sharding={replicated}
  ROOT t = (s32[], f32[4,8], f32[3,4,8]) tuple(j2, acc2, data), sharding={{replicated}, {devices=[1,2]<=[2]}, {devices=[1,1,2]<=[2]}}
}

outer_cond {
  p = (s32[], f32[4,8], f32[3,4,8]) parameter(0), sharding={{replicated}, {devices=[1,2]<=[2]}, {devices=[1,1,2]<=[2]}}
  j = s32[] get-tuple-element(p), index=0, sharding={replicated}
  n = s32[] constant(3), sharding={replicated}
  ROOT lt = pred[] compare(j, n), direction=LT, sharding={replicated}
}

ENTRY e {
  data = f32[3,4,8] parameter(0), sharding={devices=[1,1,2]<=[2]}
  zero = f32[] constant(0), sharding={replicated}
  acc = f32[4,8] broadcast(zero), dimensions={}, sharding={devices=[1,2]<=[2]}
  c0 = s32[] constant(0), sharding={replicated}
  init = (s32[], f32[4,8], f32[3,4,8]) tuple(c0, acc, data), sharding={{replicated}, {devices=[1,2]<=[2]}, {devices=[1,1,2]<=[2]}}
  w = (s32[], f32[4,8], f32[3,4,8]) while(init), condition=outer_cond, body=outer_body, sharding={{replicated}, {devices=[1,2]<=[2]}, {devices=[1,1,2]<=[2]}}
  ROOT out = f32[4,8] get-tuple-element(w), index=1, sharding={devices=[1,2]<=[2]}
}
)";
  ASSERT_TRUE(RunAndCheck(hlo));
  // Also when the accumulator's get-tuple-element carries no sharding of its
  // own: it takes its element of the loop parameter's tuple sharding.
  {
    std::unique_ptr<HloModule> folded = std::move(module_);
    EXPECT_TRUE(RunAndCheck(absl::StrReplaceAll(
        hlo, {{"index=1, sharding={devices=[1,2]<=[2]}\n  data =",
               "index=1\n  data ="}})));
    module_ = std::move(folded);
  }
  HloComputation* outer_body = module_->GetComputationWithName("outer_body");
  EXPECT_THAT(outer_body->root_instruction()->operand(1),
              op::GetTupleElement(op::While(op::Tuple(
                  op::Constant(), op::GetTupleElement(), op::Reshape()))));
  HloInstruction* inner = module_->GetComputationWithName("inner_body")
                              ->root_instruction()
                              ->mutable_operand(1);
  EXPECT_THAT(inner,
              op::DynamicUpdateSlice(op::GetTupleElement(),
                                     op::Add(op::Tanh(), op::DynamicSlice()),
                                     op::Subtract(), op::Constant()));
  EXPECT_EQ(inner->operand(1)->sharding().ToString(), "{devices=[1,2]<=[2]}");
}

// As emitted by Shardy: named shardings, and a reshape (scan unroll=2) between
// the loop and the add.
TEST_F(WhileLoopAccumulatorFoldingTest, FoldsThroughReshapeWithNamedSharding) {
  constexpr absl::string_view hlo = R"(
HloModule m

body {
  p = (s32[], f32[2,2,8], f32[2,2,8]) parameter(0), sharding={{mesh[], replicated}, {mesh['x'=2], [{}, {}, {'x'}]}, {mesh['x'=2], [{}, {}, {'x'}]}}
  i = s32[] get-tuple-element(p), index=0, sharding={mesh[], replicated}
  buf = f32[2,2,8] get-tuple-element(p), index=1, sharding={mesh['x'=2], [{}, {}, {'x'}]}
  xs = f32[2,2,8] get-tuple-element(p), index=2, sharding={mesh['x'=2], [{}, {}, {'x'}]}
  c1 = s32[] constant(1)
  idx = s32[] subtract(c1, i)
  z = s32[] constant(0)
  x = f32[1,2,8] dynamic-slice(xs, idx, z, z), dynamic_slice_sizes={1,2,8}, sharding={mesh['x'=2], [{}, {}, {'x'}]}
  g = f32[1,2,8] exponential(x), sharding={mesh['x'=2], [{}, {}, {'x'}]}
  buf2 = f32[2,2,8] dynamic-update-slice(buf, g, idx, z, z), sharding={mesh['x'=2], [{}, {}, {'x'}]}
  one = s32[] constant(1)
  i2 = s32[] add(i, one)
  ROOT t = (s32[], f32[2,2,8], f32[2,2,8]) tuple(i2, buf2, xs), sharding={{mesh[], replicated}, {mesh['x'=2], [{}, {}, {'x'}]}, {mesh['x'=2], [{}, {}, {'x'}]}}
}

cond {
  p = (s32[], f32[2,2,8], f32[2,2,8]) parameter(0), sharding={{mesh[], replicated}, {mesh['x'=2], [{}, {}, {'x'}]}, {mesh['x'=2], [{}, {}, {'x'}]}}
  i = s32[] get-tuple-element(p), index=0
  n = s32[] constant(2)
  ROOT lt = pred[] compare(i, n), direction=LT
}

ENTRY e {
  acc = f32[4,8] parameter(0), sharding={mesh['x'=2], [{}, {'x'}]}
  xs = f32[2,2,8] parameter(1), sharding={mesh['x'=2], [{}, {}, {'x'}]}
  zero = f32[] constant(0)
  b = f32[2,2,8] broadcast(zero), dimensions={}, sharding={mesh['x'=2], [{}, {}, {'x'}]}
  c0 = s32[] constant(0)
  init = (s32[], f32[2,2,8], f32[2,2,8]) tuple(c0, b, xs), sharding={{mesh[], replicated}, {mesh['x'=2], [{}, {}, {'x'}]}, {mesh['x'=2], [{}, {}, {'x'}]}}
  w = (s32[], f32[2,2,8], f32[2,2,8]) while(init), condition=cond, body=body, sharding={{mesh[], replicated}, {mesh['x'=2], [{}, {}, {'x'}]}, {mesh['x'=2], [{}, {}, {'x'}]}}
  out = f32[2,2,8] get-tuple-element(w), index=1, sharding={mesh['x'=2], [{}, {}, {'x'}]}
  rs = f32[4,8] reshape(out), sharding={mesh['x'=2], [{}, {'x'}]}
  ROOT r = f32[4,8] add(acc, rs), sharding={mesh['x'=2], [{}, {'x'}]}
}
)";
  ASSERT_TRUE(RunAndCheck(hlo));
}

TEST_F(WhileLoopAccumulatorFoldingTest, KeepsShardingMismatch) {
  constexpr absl::string_view hlo = R"(
HloModule m

body {
  p = (s32[], f32[4,8]) parameter(0), sharding={{replicated}, {devices=[1,2]<=[2]}}
  i = s32[] get-tuple-element(p), index=0, sharding={replicated}
  buf = f32[4,8] get-tuple-element(p), index=1, sharding={devices=[1,2]<=[2]}
  z = s32[] constant(0), sharding={replicated}
  c = f32[] constant(1), sharding={replicated}
  g = f32[1,8] broadcast(c), dimensions={}, sharding={devices=[1,2]<=[2]}
  buf2 = f32[4,8] dynamic-update-slice(buf, g, i, z), sharding={devices=[1,2]<=[2]}
  one = s32[] constant(1), sharding={replicated}
  i2 = s32[] add(i, one), sharding={replicated}
  ROOT t = (s32[], f32[4,8]) tuple(i2, buf2), sharding={{replicated}, {devices=[1,2]<=[2]}}
}

cond {
  p = (s32[], f32[4,8]) parameter(0), sharding={{replicated}, {devices=[1,2]<=[2]}}
  i = s32[] get-tuple-element(p), index=0, sharding={replicated}
  n = s32[] constant(4), sharding={replicated}
  ROOT lt = pred[] compare(i, n), direction=LT, sharding={replicated}
}

ENTRY e {
  acc = f32[4,8] parameter(0), sharding={devices=[2,1]<=[2]}
  zero = f32[] constant(0), sharding={replicated}
  b = f32[4,8] broadcast(zero), dimensions={}, sharding={devices=[1,2]<=[2]}
  c0 = s32[] constant(0), sharding={replicated}
  init = (s32[], f32[4,8]) tuple(c0, b), sharding={{replicated}, {devices=[1,2]<=[2]}}
  w = (s32[], f32[4,8]) while(init), condition=cond, body=body, sharding={{replicated}, {devices=[1,2]<=[2]}}
  out = f32[4,8] get-tuple-element(w), index=1, sharding={devices=[1,2]<=[2]}
  ROOT r = f32[4,8] add(acc, out), sharding={devices=[2,1]<=[2]}
}
)";
  EXPECT_FALSE(RunAndCheck(hlo));
  // Partitioning the dynamic dimension is also rejected.
  std::string layer_sharded =
      absl::StrReplaceAll(hlo, {{"devices=[1,2]<=[2]", "devices=[2,1]<=[2]"}});
  EXPECT_FALSE(RunAndCheck(layer_sharded));
}

// A slot that the body reads, e.g. the gradient of a loop-invariant weight
// that every iteration adds to: folding would reassociate the additions.
TEST_F(WhileLoopAccumulatorFoldingTest, KeepsSlotReadInBody) {
  std::string hlo = absl::StrReplaceAll(
      kBasic, {{"  buf2 = f32[4,3] dynamic-update-slice(buf, g, idx, z)\n",
                "  old = f32[1,3] dynamic-slice(buf, idx, z), "
                "dynamic_slice_sizes={1,3}\n"
                "  g2 = f32[1,3] add(old, g)\n"
                "  buf2 = f32[4,3] dynamic-update-slice(buf, g2, idx, z)\n"}});
  EXPECT_FALSE(RunAndCheck(hlo));
}

TEST_F(WhileLoopAccumulatorFoldingTest, KeepsPartialCoverage) {
  // Two iterations over four rows.
  std::string hlo = absl::StrReplaceAll(
      kBasic, {{"n = s32[] constant(4)", "n = s32[] constant(2)"}});
  EXPECT_FALSE(RunAndCheck(hlo));
}

TEST_F(WhileLoopAccumulatorFoldingTest, KeepsRepeatedOffsets) {
  // Rows 0 and 1 are each written twice, rows 2 and 3 never.
  std::string hlo = absl::StrReplaceAll(
      kBasic, {{"idx = s32[] subtract(c3, i)",
                "two = s32[] constant(2)\n  idx = s32[] divide(i, two)"}});
  EXPECT_FALSE(RunAndCheck(hlo));
}

// Every element of the loop buffer is overwritten, so its initial value does
// not matter.
TEST_F(WhileLoopAccumulatorFoldingTest, FoldsAnyInit) {
  std::string hlo = absl::StrReplaceAll(
      kBasic, {{"zero = f32[] constant(0)", "zero = f32[] constant(1)"}});
  EXPECT_TRUE(RunAndCheck(hlo));
}

TEST_F(WhileLoopAccumulatorFoldingTest, KeepsAccumulatorThatDependsOnLoop) {
  std::string hlo = absl::StrReplaceAll(
      kBasic, {{"ROOT r = f32[4,3] add(acc, out)",
                "xs2 = f32[4,3] get-tuple-element(w), index=2\n"
                "  ROOT r = f32[4,3] add(xs2, out)"}});
  EXPECT_FALSE(RunAndCheck(hlo));
}

TEST_F(WhileLoopAccumulatorFoldingTest, KeepsLoopOutputWithOtherUsers) {
  std::string hlo = absl::StrReplaceAll(
      kBasic, {{"ROOT r = f32[4,3] add(acc, out)",
                "r = f32[4,3] add(acc, out)\n"
                "  ROOT t = (f32[4,3], f32[4,3]) tuple(r, out)"}});
  EXPECT_FALSE(RunAndCheck(hlo));
}

// Two loops over the same stack (e.g. a weight used by two scans) whose
// outputs are summed: the second loop starts from the first loop's output.
TEST_F(WhileLoopAccumulatorFoldingTest, FoldsSumOfTwoLoops) {
  std::string hlo = absl::StrReplaceAll(
      kBasic,
      {{"  ROOT r = f32[4,3] add(acc, out)\n",
        "  init2 = (s32[], f32[4,3], f32[4,3]) tuple(c0, b, acc)\n"
        "  w2 = (s32[], f32[4,3], f32[4,3]) while(init2), condition=cond2, "
        "body=body2\n"
        "  out2 = f32[4,3] get-tuple-element(w2), index=1\n"
        "  ROOT r = f32[4,3] add(out, out2)\n"}});
  // Give the second loop its own computations.
  std::string body2(kBasic.substr(
      kBasic.find("body {"), kBasic.find("cond {") - kBasic.find("body {")));
  std::string cond2(kBasic.substr(
      kBasic.find("cond {"), kBasic.find("ENTRY") - kBasic.find("cond {")));
  hlo = absl::StrReplaceAll(
      hlo,
      {{"ENTRY e {", absl::StrReplaceAll(body2, {{"body {", "body2 {"}}) +
                         absl::StrReplaceAll(cond2, {{"cond {", "cond2 {"}}) +
                         "ENTRY e {"}});
  ASSERT_TRUE(RunAndCheck(hlo));
}

// As emitted by Shardy: named shardings and copies that restate a sharding
// around the loop's dynamic-update-slice, with the layer axis moved to
// position 1 (MaxText's param_scan_axis=1) before the add.
constexpr absl::string_view kShardyCopies = R"(
HloModule m

body {
  p = (s32[], f32[4,2,8], f32[4,2,8]) parameter(0), sharding={{mesh[], replicated}, {mesh['x'=2], [{}, {}, {'x'}]}, {mesh['x'=2], [{}, {}, {'x'}]}}
  i = s32[] get-tuple-element(p), index=0, sharding={mesh[], replicated}
  buf = f32[4,2,8] get-tuple-element(p), index=1, sharding={mesh['x'=2], [{}, {}, {'x'}]}
  bufc = f32[4,2,8] copy(buf), sharding={mesh['x'=2], [{}, {}, {'x'}]}
  xs = f32[4,2,8] get-tuple-element(p), index=2, sharding={mesh['x'=2], [{}, {}, {'x'}]}
  c3 = s32[] constant(3)
  idx = s32[] subtract(c3, i)
  z = s32[] constant(0)
  x = f32[1,2,8] dynamic-slice(xs, idx, z, z), dynamic_slice_sizes={1,2,8}, sharding={mesh['x'=2], [{}, {}, {'x'}]}
  g = f32[1,2,8] exponential(x), sharding={mesh['x'=2], [{}, {}, {'x'}]}
  buf2 = f32[4,2,8] dynamic-update-slice(bufc, g, idx, z, z), sharding={mesh['x'=2], [{}, {}, {'x'}]}
  buf3 = f32[4,2,8] copy(buf2), sharding={mesh['x'=2], [{}, {}, {'x'}]}
  buf4 = f32[4,2,8] copy(buf3), sharding={mesh['x'=2], [{}, {}, {'x'}]}
  one = s32[] constant(1)
  i2 = s32[] add(i, one)
  ROOT t = (s32[], f32[4,2,8], f32[4,2,8]) tuple(i2, buf4, xs), sharding={{mesh[], replicated}, {mesh['x'=2], [{}, {}, {'x'}]}, {mesh['x'=2], [{}, {}, {'x'}]}}
}

cond {
  p = (s32[], f32[4,2,8], f32[4,2,8]) parameter(0), sharding={{mesh[], replicated}, {mesh['x'=2], [{}, {}, {'x'}]}, {mesh['x'=2], [{}, {}, {'x'}]}}
  i = s32[] get-tuple-element(p), index=0
  n = s32[] constant(4)
  ROOT lt = pred[] compare(i, n), direction=LT
}

ENTRY e {
  a = f32[2,4,8] parameter(0), sharding={mesh['x'=2], [{}, {}, {'x'}]}
  acc = f32[2,4,8] copy(a), sharding={mesh['x'=2], [{}, {}, {'x'}]}
  xs = f32[4,2,8] parameter(1), sharding={mesh['x'=2], [{}, {}, {'x'}]}
  zero = f32[] constant(0)
  b = f32[4,2,8] broadcast(zero), dimensions={}, sharding={mesh['x'=2], [{}, {}, {'x'}]}
  c0 = s32[] constant(0)
  init = (s32[], f32[4,2,8], f32[4,2,8]) tuple(c0, b, xs), sharding={{mesh[], replicated}, {mesh['x'=2], [{}, {}, {'x'}]}, {mesh['x'=2], [{}, {}, {'x'}]}}
  w = (s32[], f32[4,2,8], f32[4,2,8]) while(init), condition=cond, body=body, sharding={{mesh[], replicated}, {mesh['x'=2], [{}, {}, {'x'}]}, {mesh['x'=2], [{}, {}, {'x'}]}}
  out = f32[4,2,8] get-tuple-element(w), index=1, sharding={mesh['x'=2], [{}, {}, {'x'}]}
  outc = f32[4,2,8] copy(out), sharding={mesh['x'=2], [{}, {}, {'x'}]}
  tr = f32[2,4,8] transpose(outc), dimensions={1,0,2}, sharding={mesh['x'=2], [{}, {}, {'x'}]}
  ROOT r = f32[2,4,8] add(acc, tr), sharding={mesh['x'=2], [{}, {}, {'x'}]}
}
)";

TEST_F(WhileLoopAccumulatorFoldingTest, FoldsThroughShardyCopies) {
  ASSERT_TRUE(RunAndCheck(kShardyCopies));
  HloInstruction* while_instr =
      module_->entry_computation()->root_instruction()->mutable_operand(0);
  EXPECT_EQ(while_instr->shape().tuple_shapes(1).ToString(), "f32[2,4,8]");
  EXPECT_THAT(
      while_instr->while_body()->root_instruction()->operand(1),
      op::DynamicUpdateSlice(
          op::GetTupleElement(),
          op::Add(op::DynamicSlice(op::GetTupleElement(), op::Constant(),
                                   op::Subtract(), op::Constant()),
                  op::Transpose(op::Exp())),
          op::Constant(), op::Subtract(), op::Constant()));
}

TEST_F(WhileLoopAccumulatorFoldingTest, KeepsCopyThatChangesSharding) {
  std::string hlo = absl::StrReplaceAll(
      kShardyCopies,
      {{"buf3 = f32[4,2,8] copy(buf2), sharding={mesh['x'=2], [{}, {}, {'x'}]}",
        "buf3 = f32[4,2,8] copy(buf2), sharding={mesh['x'=2], [{}, {'x'}, "
        "{}]}"}});
  EXPECT_FALSE(RunAndCheck(hlo));
}

// The loop would need its own copy of an accumulator that stays alive.
TEST_F(WhileLoopAccumulatorFoldingTest, KeepsAccumulatorWithOtherUsers) {
  std::string hlo = absl::StrReplaceAll(
      kBasic, {{"ROOT r = f32[4,3] add(acc, out)",
                "r = f32[4,3] add(acc, out)\n"
                "  ROOT t = (f32[4,3], f32[4,3]) tuple(r, acc)"}});
  EXPECT_FALSE(RunAndCheck(hlo));
  // Also through a copy of the accumulator.
  hlo = absl::StrReplaceAll(
      kBasic, {{"ROOT r = f32[4,3] add(acc, out)",
                "accc = f32[4,3] copy(acc)\n"
                "  r = f32[4,3] add(accc, out)\n"
                "  ROOT t = (f32[4,3], f32[4,3]) tuple(r, acc)"}});
  EXPECT_FALSE(RunAndCheck(hlo));
}

// Two rows per iteration of an [8,3] buffer, reshaped to [4,6]: the offset into
// the accumulator is the loop offset divided by 2.
constexpr absl::string_view kMergeRows = R"(
HloModule m

body {
  p = (s8[], f32[8,3], f32[8,3]) parameter(0)
  i = s8[] get-tuple-element(p), index=0
  buf = f32[8,3] get-tuple-element(p), index=1
  xs = f32[8,3] get-tuple-element(p), index=2
  two = s8[] constant(2)
  idx = s8[] multiply(i, two)
  z = s8[] constant(0)
  x = f32[2,3] dynamic-slice(xs, idx, z), dynamic_slice_sizes={2,3}
  g = f32[2,3] exponential(x)
  buf2 = f32[8,3] dynamic-update-slice(buf, g, idx, z)
  one = s8[] constant(1)
  i2 = s8[] add(i, one)
  ROOT t = (s8[], f32[8,3], f32[8,3]) tuple(i2, buf2, xs)
}

cond {
  p = (s8[], f32[8,3], f32[8,3]) parameter(0)
  i = s8[] get-tuple-element(p), index=0
  n = s8[] constant(4)
  ROOT lt = pred[] compare(i, n), direction=LT
}

ENTRY e {
  acc = f32[4,6] parameter(0)
  xs = f32[8,3] parameter(1)
  zero = f32[] constant(0)
  b = f32[8,3] broadcast(zero), dimensions={}
  c0 = s8[] constant(0)
  init = (s8[], f32[8,3], f32[8,3]) tuple(c0, b, xs)
  w = (s8[], f32[8,3], f32[8,3]) while(init), condition=cond, body=body
  out = f32[8,3] get-tuple-element(w), index=1
  rs = f32[4,6] reshape(out)
  ROOT r = f32[4,6] add(acc, rs)
}
)";

TEST_F(WhileLoopAccumulatorFoldingTest, FoldsReshapeThatMergesRows) {
  ASSERT_TRUE(RunAndCheck(kMergeRows));
  HloInstruction* while_instr =
      module_->entry_computation()->root_instruction()->mutable_operand(0);
  EXPECT_THAT(while_instr->while_body()->root_instruction()->operand(1),
              op::DynamicUpdateSlice(
                  op::GetTupleElement(),
                  op::Add(op::DynamicSlice(), op::Reshape(op::Exp())),
                  op::Divide(op::Multiply(), op::Constant()), op::Constant()));
}

// An s8 offset can index the [4,64] loop buffer, but not the [256]
// accumulator it is reshaped to.
TEST_F(WhileLoopAccumulatorFoldingTest, KeepsOffsetTypeTooNarrow) {
  std::string hlo = absl::StrReplaceAll(
      kMergeRows, {{"f32[8,3]", "f32[4,64]"},
                   {"f32[2,3]", "f32[1,64]"},
                   {"dynamic_slice_sizes={2,3}", "dynamic_slice_sizes={1,64}"},
                   {"idx = s8[] multiply(i, two)", "idx = s8[] add(i, z)"},
                   {"f32[4,6]", "f32[256]"}});
  // `z` must be defined before `idx`.
  hlo = absl::StrReplaceAll(
      hlo, {{"  two = s8[] constant(2)\n", "  z = s8[] constant(0)\n"},
            {"  z = s8[] constant(0)\n  x = ", "  x = "}});
  EXPECT_FALSE(RunAndCheck(hlo));
  // The same reshape folds with s32 offsets.
  EXPECT_TRUE(RunAndCheck(absl::StrReplaceAll(hlo, {{"s8[]", "s32[]"}})));
}

// A tuple-shaped loop may carry a single sharding for all of its elements.
TEST_F(WhileLoopAccumulatorFoldingTest, FoldsWithSingleTupleSharding) {
  std::string hlo = absl::StrReplaceAll(
      kBasic,
      {{"while(init), condition=cond, body=body",
        "while(init), condition=cond, body=body, sharding={replicated}"},
       {"out = f32[4,3] get-tuple-element(w), index=1",
        "out = f32[4,3] get-tuple-element(w), index=1, sharding={replicated}"},
       {"acc = f32[4,3] parameter(0)",
        "acc = f32[4,3] parameter(0), sharding={replicated}"},
       {"ROOT r = f32[4,3] add(acc, out)",
        "ROOT r = f32[4,3] add(acc, out), sharding={replicated}"}});
  EXPECT_TRUE(RunAndCheck(hlo));
}

}  // namespace
}  // namespace xla
