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

#include <algorithm>
#include <cstdint>
#include <functional>
#include <memory>
#include <numeric>
#include <optional>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/analysis/while_loop_analysis.h"
#include "xla/hlo/evaluator/hlo_evaluator.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/hlo/utils/hlo_sharding_util.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/primitive_util.h"
#include "xla/service/value_range.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {

// Loops with more iterations than this are not analyzed: the pass evaluates
// the update offset once per iteration.
constexpr int64_t kMaxTripCount = 1 << 16;

// How a while body writes one slot of its state: a single
// dynamic-update-slice per iteration that writes `slice` elements along `dim`
// and all elements of the other dimensions, such that the iterations together
// write every element exactly once.
struct SlotWrite {
  HloInstruction* gte;   // The body's get-tuple-element of the slot.
  HloInstruction* dus;   // The body's dynamic-update-slice of the slot.
  HloInstruction* tail;  // What the body returns: `dus` or a copy of it.
  int64_t dim;
  int64_t slice;
};

// The sharding of element `index` of a tuple-shaped value with sharding `s`,
// which may be a single sharding for all elements.
HloSharding ElementSharding(const HloSharding& s, const Shape& shape,
                            int64_t index) {
  return s.IsTuple() ? s.GetSubSharding(shape, {index}) : s;
}

// After sharding propagation, an instruction without a sharding is replicated,
// except for a get-tuple-element, which takes its element of the tuple's
// sharding.
HloSharding EffectiveSharding(const HloInstruction* instr) {
  if (instr->has_sharding()) {
    return instr->sharding();
  }
  if (instr->opcode() == HloOpcode::kGetTupleElement) {
    const HloInstruction* tuple = instr->operand(0);
    return ElementSharding(EffectiveSharding(tuple), tuple->shape(),
                           instr->tuple_index());
  }
  return HloSharding::Replicate();
}

// Whether `instr` is a copy that keeps its operand's sharding, as Shardy's
// export leaves between the ops of a loop body.
bool IsShardingPreservingCopy(const HloInstruction* instr) {
  return instr->opcode() == HloOpcode::kCopy &&
         !instr->HasControlDependencies() &&
         EffectiveSharding(instr) == EffectiveSharding(instr->operand(0));
}

// Returns the last of the single-user, sharding-preserving copies that follow
// `instr`, or `instr` itself.
HloInstruction* SkipCopiesForward(HloInstruction* instr) {
  while (instr->user_count() == 1 &&
         IsShardingPreservingCopy(instr->users().front())) {
    instr = instr->users().front();
  }
  return instr;
}

// Returns the get-tuple-element users of `tuple` with index `index`, or
// nullopt if `tuple` has a user that is not a get-tuple-element.
std::optional<std::vector<HloInstruction*>> TupleIndexUsers(
    HloInstruction* tuple, int64_t index) {
  std::vector<HloInstruction*> gtes;
  for (HloInstruction* user : tuple->users()) {
    if (user->opcode() != HloOpcode::kGetTupleElement) {
      return std::nullopt;
    }
    if (user->tuple_index() == index) {
      gtes.push_back(user);
    }
  }
  return gtes;
}

std::optional<SlotWrite> MatchCoveringSlotWrite(HloInstruction* while_instr,
                                                int64_t k) {
  HloComputation* body = while_instr->while_body();
  HloComputation* cond = while_instr->while_condition();
  if (body->caller_instructions().size() != 1 ||
      cond->caller_instructions().size() != 1) {
    VLOG(3) << "Loop computations have multiple callers.";
    return std::nullopt;
  }

  // The condition must not read the slot.
  std::optional<std::vector<HloInstruction*>> cond_gtes =
      TupleIndexUsers(cond->parameter_instruction(0), k);
  if (!cond_gtes.has_value() || !cond_gtes->empty()) {
    VLOG(3) << "Loop condition reads the slot.";
    return std::nullopt;
  }

  // The body must only pass the slot to one dynamic-update-slice, as its
  // operand, and return the result in the same slot. Copies that keep the
  // sharding may sit on either side of the dynamic-update-slice.
  std::optional<std::vector<HloInstruction*>> body_gtes =
      TupleIndexUsers(body->parameter_instruction(0), k);
  if (!body_gtes.has_value() || body_gtes->size() != 1) {
    VLOG(3) << "Loop body does not read the slot exactly once.";
    return std::nullopt;
  }
  HloInstruction* gte = body_gtes->front();
  HloInstruction* slot = SkipCopiesForward(gte);
  if (slot->user_count() != 1) {
    return std::nullopt;
  }
  HloInstruction* dus = slot->users().front();
  HloInstruction* tail = SkipCopiesForward(dus);
  HloInstruction* root = body->root_instruction();
  if (dus->opcode() != HloOpcode::kDynamicUpdateSlice ||
      dus->operand_index(slot) != 0 || dus->operand(1) == slot ||
      root->opcode() != HloOpcode::kTuple || root->operand(k) != tail ||
      tail->user_count() != 1 || root->OperandIndices(tail).size() != 1 ||
      dus->HasControlDependencies()) {
    VLOG(3) << "Slot is not updated by a single dynamic-update-slice.";
    return std::nullopt;
  }

  // Exactly one dimension may have a non-constant offset; the update must span
  // all other dimensions.
  const Shape& buffer_shape = dus->shape();
  const Shape& update_shape = dus->operand(1)->shape();
  int64_t dim = -1;
  for (int64_t d = 0; d < buffer_shape.dimensions().size(); ++d) {
    const HloInstruction* offset = dus->operand(2 + d);
    if (offset->opcode() == HloOpcode::kConstant) {
      if (!offset->literal().IsAll(0) ||
          update_shape.dimensions(d) != buffer_shape.dimensions(d)) {
        return std::nullopt;
      }
      continue;
    }
    if (dim != -1) {
      VLOG(3) << "Multiple dynamic offsets: " << dus->ToString();
      return std::nullopt;
    }
    dim = d;
  }
  if (dim == -1) {
    return std::nullopt;
  }

  std::optional<int64_t> indvar_index =
      GetLoopInductionVarTupleIdx(while_instr);
  std::optional<int64_t> trip_count = ComputeWhileLoopTripCount(while_instr);
  std::optional<Range> range = MatchTrivialLoopRange(while_instr);
  if (!indvar_index.has_value() || *indvar_index == k ||
      !trip_count.has_value() || *trip_count > kMaxTripCount ||
      !range.has_value() || !range->IsStepKnown()) {
    VLOG(3) << "Loop has no static trip count: " << while_instr->name();
    return std::nullopt;
  }
  std::optional<std::vector<HloInstruction*>> indvars =
      TupleIndexUsers(body->parameter_instruction(0), *indvar_index);
  if (!indvars.has_value() || indvars->empty()) {
    return std::nullopt;
  }

  // Evaluate the offset of every iteration; they must tile the dimension.
  const int64_t slice = update_shape.dimensions(dim);
  const int64_t size = buffer_shape.dimensions(dim);
  if (*trip_count * slice != size) {
    VLOG(3) << "Trip count " << *trip_count << " times slice " << slice
            << " does not cover " << size;
    return std::nullopt;
  }
  const PrimitiveType indvar_type = indvars->front()->shape().element_type();
  const int64_t start = range->min().GetSignedValue();
  const int64_t step = range->step()->GetSignedValue();
  std::vector<bool> written(*trip_count, false);
  HloEvaluator evaluator(/*max_loop_iterations=*/0);
  for (int64_t i = 0; i < *trip_count; ++i) {
    absl::StatusOr<Literal> value =
        LiteralUtil::CreateR0<int64_t>(start + i * step).Convert(indvar_type);
    if (!value.ok()) {
      return std::nullopt;
    }
    absl::flat_hash_map<const HloInstruction*, const LiteralBase*> subst;
    for (HloInstruction* indvar : *indvars) {
      subst[indvar] = &*value;
    }
    evaluator.ResetVisitStates();
    absl::StatusOr<Literal> offset = evaluator.Evaluate(
        dus->operand(2 + dim), /*precomputed_analyses=*/{},
        /*recursively_evaluate_nonconstant_operands=*/true, subst);
    if (!offset.ok()) {
      VLOG(3) << "Cannot evaluate offset: " << offset.status();
      return std::nullopt;
    }
    std::optional<int64_t> o = LiteralUtil::LiteralAsScalarInt64(*offset);
    if (!o.has_value() || *o < 0 || *o > size - slice || *o % slice != 0 ||
        written[*o / slice]) {
      VLOG(3) << "Offsets do not tile the dimension: " << dus->ToString();
      return std::nullopt;
    }
    written[*o / slice] = true;
  }
  return SlotWrite{gte, dus, tail, dim, slice};
}

// Tracks the slab that one iteration writes as the loop buffer is mapped onto
// the accumulator's shape: `slice` elements along `dim`. The loop writes its
// slabs at multiples of the loop buffer's slice (see MatchCoveringSlotWrite),
// so the slab at offset `q * slice` in the loop buffer lands at `q * slice` in
// every shape along the chain, with that shape's `slice`.
struct SlabMap {
  Shape shape;
  int64_t dim;
  int64_t slice;
  std::optional<HloSharding> sharding;
};

int64_t Product(absl::Span<const int64_t> dims) {
  return std::accumulate(dims.begin(), dims.end(), int64_t{1},
                         std::multiplies<int64_t>());
}

// Maps `map` through `op`; returns false if the slab is not a slab of the
// result, or if `op` changes the sharding.
bool MapThrough(const HloInstruction* op, SlabMap& map) {
  std::optional<HloSharding> expected = map.sharding;
  switch (op->opcode()) {
    case HloOpcode::kConvert:
    case HloOpcode::kCopy:
      break;
    case HloOpcode::kTranspose: {
      absl::Span<const int64_t> perm = op->dimensions();
      map.dim = std::find(perm.begin(), perm.end(), map.dim) - perm.begin();
      if (expected.has_value()) {
        expected = hlo_sharding_util::TransposeSharding(*expected, perm);
      }
      break;
    }
    case HloOpcode::kReshape: {
      absl::Span<const int64_t> in = map.shape.dimensions();
      absl::Span<const int64_t> out = op->shape().dimensions();
      const int64_t outer = Product(in.subspan(0, map.dim));
      const int64_t inner = Product(in.subspan(map.dim + 1));
      const int64_t span = in[map.dim] * inner;
      int64_t e = 0;
      for (; e < out.size(); ++e) {
        const int64_t out_inner = Product(out.subspan(e + 1));
        if (Product(out.subspan(0, e)) == outer && out[e] * out_inner == span &&
            out[e] > 1 && (map.slice * inner) % out_inner == 0) {
          map.slice = map.slice * inner / out_inner;
          break;
        }
      }
      if (e == out.size()) {
        VLOG(3) << "Reshape does not map the slab to a slab: "
                << op->ToString();
        return false;
      }
      map.dim = e;
      if (expected.has_value()) {
        // ReshapeSharding only supports tile-based shardings.
        if (expected->UseNamedShardingLeaf()) {
          expected = HloSharding::V3ToV2Sharding(*expected);
        }
        expected = hlo_sharding_util::ReshapeSharding(map.shape, op->shape(),
                                                      *expected);
        if (!expected.has_value()) {
          return false;
        }
      }
      break;
    }
    default:
      return false;
  }
  map.shape = op->shape();
  if (expected.has_value() && EffectiveSharding(op) != *expected) {
    VLOG(3) << "Sharding changes along the chain at " << op->ToString();
    return false;
  }
  if (expected.has_value()) {
    map.sharding = EffectiveSharding(op);
  }
  return true;
}

// Returns true if `target` is `start` or one of its transitive operands.
bool DependsOn(const HloInstruction* start, const HloInstruction* target) {
  std::vector<const HloInstruction*> stack = {start};
  absl::flat_hash_set<const HloInstruction*> visited;
  while (!stack.empty()) {
    const HloInstruction* instr = stack.back();
    stack.pop_back();
    if (instr == target) {
      return true;
    }
    if (!visited.insert(instr).second) {
      continue;
    }
    for (const HloInstruction* operand : instr->operands()) {
      stack.push_back(operand);
    }
    for (const HloInstruction* pred : instr->control_predecessors()) {
      stack.push_back(pred);
    }
  }
  return false;
}

HloSharding WithTupleElement(const HloSharding& tuple_sharding,
                             const Shape& tuple_shape, int64_t index,
                             const HloSharding& element) {
  std::vector<HloSharding> elements = tuple_sharding.tuple_elements();
  CHECK_LT(index, elements.size());
  elements[index] = element;
  return HloSharding::Tuple(tuple_shape, elements);
}

// Sets element `index` of the tuple-shaped `instr` to `shape`, and its
// sharding to `sharding` if it has a tuple sharding.
void SetTupleElement(HloInstruction* instr, int64_t index, const Shape& shape,
                     const std::optional<HloSharding>& sharding) {
  *instr->mutable_shape()->mutable_tuple_shapes(index) = shape;
  if (sharding.has_value() && instr->has_sharding() &&
      instr->sharding().IsTuple()) {
    instr->set_sharding(
        WithTupleElement(instr->sharding(), instr->shape(), index, *sharding));
  }
}

// Whether `type` can hold every offset into a dimension of size `size`.
bool HoldsOffsets(PrimitiveType type, int64_t size) {
  if (!primitive_util::IsIntegralType(type)) {
    return false;
  }
  const int bits = primitive_util::BitWidth(type) -
                   (primitive_util::IsSignedIntegralType(type) ? 1 : 0);
  return bits >= 63 || size - 1 <= (int64_t{1} << bits) - 1;
}

absl::StatusOr<HloInstruction*> MakeScalarIndexOp(HloOpcode opcode,
                                                  HloInstruction* index,
                                                  int64_t value) {
  HloComputation* comp = index->parent();
  HloInstruction* constant = comp->AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<int64_t>(value)
                                         .Convert(index->shape().element_type())
                                         .value()));
  return comp->AddInstruction(
      HloInstruction::CreateBinary(index->shape(), opcode, index, constant));
}

// Tries to fold `add`'s operand `operand_index` (a chain of layout and dtype
// ops on a while loop's output) into the loop, as described in the header.
absl::StatusOr<bool> TryFold(HloInstruction* add, int64_t operand_index) {
  HloComputation* comp = add->parent();
  HloInstruction* acc = add->mutable_operand(1 - operand_index);
  HloInstruction* top = add->mutable_operand(operand_index);
  if (top == acc || top->user_count() != 1 || add->HasControlDependencies()) {
    return false;
  }

  // Walk from the add operand down to the loop output.
  std::vector<HloInstruction*> chain;
  HloInstruction* instr = top;
  while (instr->opcode() == HloOpcode::kConvert ||
         instr->opcode() == HloOpcode::kCopy ||
         instr->opcode() == HloOpcode::kTranspose ||
         instr->opcode() == HloOpcode::kReshape) {
    if (instr->user_count() != 1) {
      return false;
    }
    chain.push_back(instr);
    instr = instr->mutable_operand(0);
  }
  std::reverse(chain.begin(), chain.end());
  if (instr->opcode() != HloOpcode::kGetTupleElement ||
      instr->user_count() != 1 ||
      instr->operand(0)->opcode() != HloOpcode::kWhile) {
    return false;
  }
  HloInstruction* out_gte = instr;
  HloInstruction* while_instr = out_gte->mutable_operand(0);
  const int64_t k = out_gte->tuple_index();
  HloInstruction* init = while_instr->mutable_operand(0);
  std::optional<std::vector<HloInstruction*>> out_gtes =
      TupleIndexUsers(while_instr, k);
  // The slot's initial value is not checked: the loop overwrites every element
  // (see MatchCoveringSlotWrite), so the loop output does not depend on it.
  if (init->opcode() != HloOpcode::kTuple || !out_gtes.has_value() ||
      out_gtes->size() != 1) {
    return false;
  }
  if (DependsOn(acc, while_instr)) {
    VLOG(3) << "Accumulator depends on the loop: " << add->ToString();
    return false;
  }
  // Folding only saves memory if the accumulator dies at the add; otherwise
  // the loop needs its own copy of it and only gains the per-iteration add.
  for (const HloInstruction* value = acc;; value = value->operand(0)) {
    if (value->user_count() != 1) {
      VLOG(3) << "Accumulator has other users: " << acc->ToString();
      return false;
    }
    if (value->opcode() != HloOpcode::kCopy) {
      break;
    }
  }

  std::optional<SlotWrite> write = MatchCoveringSlotWrite(while_instr, k);
  if (!write.has_value()) {
    return false;
  }

  // Map the per-iteration slab onto the accumulator.
  SlabMap map{write->dus->shape(), write->dim, write->slice};
  const bool sharded = while_instr->has_sharding();
  if (sharded) {
    map.sharding =
        ElementSharding(while_instr->sharding(), while_instr->shape(), k);
  } else if (add->has_sharding() || acc->has_sharding()) {
    return false;
  }
  // The slab's dimension and size after each op of the chain.
  std::vector<std::pair<int64_t, int64_t>> slabs;
  for (const HloInstruction* op : chain) {
    if (!MapThrough(op, map)) {
      return false;
    }
    slabs.push_back({map.dim, map.slice});
  }
  // Compatible() also requires the element types to match.
  const Shape& acc_shape = add->shape();
  if (!ShapeUtil::Compatible(map.shape, acc_shape) ||
      !ShapeUtil::Compatible(acc->shape(), acc_shape)) {
    return false;
  }
  // The offsets into the accumulator are computed in the loop's offset type.
  HloInstruction* buffer_offset = write->dus->mutable_operand(2 + write->dim);
  if (!HoldsOffsets(buffer_offset->shape().element_type(),
                    acc_shape.dimensions(map.dim))) {
    VLOG(3) << "Offset type cannot index the accumulator: " << add->ToString();
    return false;
  }
  if (sharded) {
    const HloSharding& s = *map.sharding;
    if (s != EffectiveSharding(add) || s != EffectiveSharding(acc) ||
        s.IsManual() || s.IsUnreduced() ||
        (s.IsTiled() && s.dimension(map.dim) != 1)) {
      VLOG(3) << "Accumulator sharding " << EffectiveSharding(acc).ToString()
              << " does not match the loop buffer's " << s.ToString()
              << " or partitions the dynamic dimension.";
      return false;
    }
  }

  VLOG(2) << "Folding " << add->name() << " into " << while_instr->name()
          << " slot " << k;

  // Rewrite the body: buf' = dus(buf, add(ds(buf, o'), f'(update)), o').
  HloComputation* body = while_instr->while_body();
  HloInstruction* gte = write->gte;
  HloInstruction* dus = write->dus;
  HloInstruction* update = dus->mutable_operand(1);
  // Apply the chain to the update: each op's output narrowed to the slab.
  for (int64_t i = 0; i < chain.size(); ++i) {
    const HloInstruction* op = chain[i];
    if (op->opcode() == HloOpcode::kCopy) {
      continue;
    }
    Shape shape = op->shape();
    shape.set_dimensions(slabs[i].first, slabs[i].second);
    std::unique_ptr<HloInstruction> next;
    switch (op->opcode()) {
      case HloOpcode::kConvert:
        next = HloInstruction::CreateConvert(shape, update);
        break;
      case HloOpcode::kTranspose:
        next = HloInstruction::CreateTranspose(shape, update, op->dimensions());
        break;
      case HloOpcode::kReshape:
        next = HloInstruction::CreateReshape(shape, update);
        break;
      default:
        return absl::InternalError("Unexpected op in accumulator chain.");
    }
    if (op->has_sharding()) {
      next->set_sharding(op->sharding());
    }
    update = body->AddInstruction(std::move(next));
  }

  // Offsets into the accumulator: the loop buffer offset is a multiple of the
  // loop buffer's slice, and the slab lands at the same multiple of the
  // accumulator's (see SlabMap). Dividing first keeps every intermediate value
  // at most the final offset, so it cannot overflow.
  HloInstruction* offset = buffer_offset;
  if (map.slice != write->slice) {
    if (write->slice != 1) {
      ABSL_ASSIGN_OR_RETURN(
          offset, MakeScalarIndexOp(HloOpcode::kDivide, offset, write->slice));
    }
    if (map.slice != 1) {
      ABSL_ASSIGN_OR_RETURN(
          offset, MakeScalarIndexOp(HloOpcode::kMultiply, offset, map.slice));
    }
  }
  HloInstruction* zero = body->AddInstruction(HloInstruction::CreateConstant(
      LiteralUtil::Zero(offset->shape().element_type())));
  std::vector<HloInstruction*> offsets(acc_shape.dimensions().size(), zero);
  offsets[map.dim] = offset;

  const Shape slot_shape = acc->shape();
  std::optional<HloSharding> slot_sharding;
  if (sharded) {
    slot_sharding = EffectiveSharding(acc);
  }
  *gte->mutable_shape() = slot_shape;
  if (slot_sharding.has_value()) {
    gte->set_sharding(*slot_sharding);
  }
  SetTupleElement(body->parameter_instruction(0), k, slot_shape, slot_sharding);
  HloInstruction* old = body->AddInstruction(HloInstruction::CreateDynamicSlice(
      update->shape(), gte, offsets, update->shape().dimensions()));
  HloInstruction* sum = body->AddInstruction(HloInstruction::CreateBinary(
      update->shape(), HloOpcode::kAdd, operand_index == 0 ? update : old,
      operand_index == 0 ? old : update));
  HloInstruction* new_dus = body->AddInstruction(
      HloInstruction::CreateDynamicUpdateSlice(slot_shape, gte, sum, offsets));
  if (slot_sharding.has_value()) {
    old->set_sharding(*slot_sharding);
    sum->set_sharding(*slot_sharding);
    new_dus->set_sharding(*slot_sharding);
  }
  HloInstruction* root = body->root_instruction();
  ABSL_RETURN_IF_ERROR(root->ReplaceOperandWithDifferentShape(k, new_dus));
  SetTupleElement(root, k, slot_shape, slot_sharding);
  ABSL_RETURN_IF_ERROR(body->RemoveInstructionAndUnusedOperands(write->tail));

  // Rewrite the condition's parameter, the loop and its init.
  SetTupleElement(while_instr->while_condition()->parameter_instruction(0), k,
                  slot_shape, slot_sharding);
  if (init->user_count() > 1) {
    HloInstruction* clone = comp->AddInstruction(init->Clone());
    ABSL_RETURN_IF_ERROR(while_instr->ReplaceOperandWith(0, clone));
    init = clone;
  }
  ABSL_RETURN_IF_ERROR(init->ReplaceOperandWithDifferentShape(k, acc));
  SetTupleElement(init, k, slot_shape, slot_sharding);
  SetTupleElement(while_instr, k, slot_shape, slot_sharding);
  *out_gte->mutable_shape() = slot_shape;
  if (slot_sharding.has_value()) {
    out_gte->set_sharding(*slot_sharding);
  }
  ABSL_RETURN_IF_ERROR(add->ReplaceAllUsesWith(out_gte));
  ABSL_RETURN_IF_ERROR(comp->RemoveInstructionAndUnusedOperands(add));
  return true;
}

}  // namespace

absl::StatusOr<bool> WhileLoopAccumulatorFolding::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  for (HloComputation* comp :
       module->MakeNonfusionComputations(execution_threads)) {
    std::vector<HloInstruction*> adds;
    for (HloInstruction* instr : comp->instructions()) {
      if (instr->opcode() == HloOpcode::kAdd && instr->shape().IsArray()) {
        adds.push_back(instr);
      }
    }
    // Folding removes only the add and the chain feeding it, so the remaining
    // candidates stay valid.
    for (HloInstruction* add : adds) {
      for (int64_t i = 0; i < 2; ++i) {
        ABSL_ASSIGN_OR_RETURN(bool folded, TryFold(add, i));
        if (folded) {
          changed = true;
          break;
        }
      }
    }
  }
  return changed;
}

}  // namespace xla
