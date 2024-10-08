// TODO: license

#include "xla/service/experimental/instruction_strategies.h"

#include "xla/service/experimental/instruction_to_module.h"
#include "xla/service/experimental/sharding_enumeration.h"
#include "xla/service/experimental/sharding_strategy_evaluator.h"
#include "xla/service/experimental/module_cost_evaluator.h"

#include <stdint.h>

namespace xla {

/*********************************************************/
/* InstructionStrategies Class                         */
/*********************************************************/

InstructionStrategies::InstructionStrategies(HloInstruction* orig_instr) 
  : orig_instr_(orig_instr) {

  // create a single instruction module which will then be used for evaluating
  // all of the sharding strats
  std::unique_ptr<HloModule> single_instr_module = 
    CreateModuleFromInstruction(orig_instr);

  std::vector<ShardingStrategy> potential_strats = 
    EnumerateShardingStrategies(orig_instr);

  // keep only the ones that are valid
  for (int i = 0; i < potential_strats.size(); i++) {
    if (IsValidShardingStrat(single_instr_module.get(), &potential_strats[i])) {
      sharding_strats_.push_back(potential_strats[i]);
    }
  }

  // estimate costs of each sharding strategy
  // TODO: is it weird the way we are creating a pointer to the
  // i'th sharding strategy?
  for (int i = 0; i < sharding_strats_.size(); i++) {
    EvaluateShardingStrat(single_instr_module.get(), &sharding_strats_[i]);
  }

  // TODO: add assert that there is no sharding on the original instruction 
  // estimate the number of FLOPs for an unsharded module
  ModuleCostEvaluator evaluator;
  fully_replicated_flops_ = evaluator.EvaluateFLOPs(single_instr_module.get());

  // TODO: iterate through sharding strats and ignore those that are fully
  // sharded but do not have an inversely proportionate number of FLOPs and
  // decide if this is worth doing

  return;
}

void InstructionStrategies::set_chosen_result_sharding(int idx) {
  assert(0 <= idx && idx < sharding_strats_.size());
  sharding_strats_[idx].ApplyToOnlyInstruction(orig_instr_);
  return;
}

} // xla