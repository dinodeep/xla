// TODO: license

#include "xla/service/experimental/sharding_strategy.h"

#include "xla/hlo/ir/hlo_opcode.h"

namespace xla {

namespace {

// This function clears all shardings from instructions in the module
void ClearHloShardings(HloModule* module) {

  for (HloComputation* computation : module->computations()) {
    for (HloInstruction* instruction : computation->instructions()) {
      instruction->clear_sharding();
    }
  }

  return;
}

} // namespace

void ShardingStrategy::AddOpSharding(HloSharding sharding) {
  operand_shardings_.push_back(std::make_shared<HloSharding>(sharding));

  // update whether found fully sharded operations
  has_fully_sharded_op_ |= 
    !sharding.IsReplicated() && !sharding.HasPartialReplication();
  return;
}

void ShardingStrategy::set_result_sharding(HloSharding result_sharding) {
  result_sharding_ = std::make_shared<HloSharding>(result_sharding);
}

void ShardingStrategy::ApplyToInstruction(HloInstruction* instr) {
  int num_operands = instr->operand_count();

  // if parameter instruction, then apply result sharding to instruction itself
  if (instr->opcode() == HloOpcode::kParameter) {
    instr->set_sharding(result_sharding());
    return;
  }

  // otherwise, general instruction, apply to it's operands
  assert(num_operands == NumOpShardings());
  for (int i = 0; i < num_operands; i++) {
    instr->mutable_operand(i)->set_sharding(GetOpSharding(i));
  }

  return;
}

// Assumes that provided module is a single-instruction module
void ShardingStrategy::ApplyToModule(HloModule* module) {
  ClearHloShardings(module);
  ApplyToInstruction(module->entry_computation()->root_instruction()); 
  return; 
}

} // xla