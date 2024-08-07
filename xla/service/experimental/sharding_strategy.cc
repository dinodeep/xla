// TODO: license

#include "xla/service/experimental/sharding_strategy.h"

namespace xla {

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

  assert(num_operands == NumOpShardings());
  for (int i = 0; i < num_operands; i++) {
    instr->mutable_operand(i)->set_sharding(GetOpSharding(i));
  }

  return;
}

void ShardingStrategy::ApplyToModule(HloModule* module) {
  ApplyToInstruction(module->entry_computation()->root_instruction()); 
  return; 
}

} // xla