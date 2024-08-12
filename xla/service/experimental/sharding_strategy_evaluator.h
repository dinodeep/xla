// TODO: license

#ifndef XLA_SERVICE_EXPERIMENTAL_SHARDING_STRATEGY_EVALUATOR_H_
#define XLA_SERVICE_EXPERIMENTAL_SHARDING_STRATEGY_EVALUATOR_H_

#include "xla/service/experimental/sharding_strategy.h"
#include "xla/hlo/ir/hlo_module.h"

namespace xla {

// This function will determine if a sharding strategy is "valid".
// A strategy is valid if applying it to the instruction within the module
// and running GSPMD on it doesn't reduce it to another sharding strategy
// A reduction from one sharding strategy to another sharding strategy occurs
// in an instruction if there exists an operand of the instruction
// (or parameter in the module) that has only 1 user which is a communication
// primtive that modifies the sharding of the data
bool IsValidShardingStrat(const HloModule* module, ShardingStrategy* strat);

// This function will evaluate the sharding strategy on the 
// single-instruction module by applying the input shardings from the strat
// onto the operands of the module's root instruction, running GSPMD,
// and evaluating the communication costs of the resulting module
// The strat parameter will be updated with this cost and the resulting
// output sharding
void EvaluateShardingStrat(const HloModule* module, ShardingStrategy* strat);

}

#endif // XLA_SERVICE_EXPERIMENTAL_SHARDING_STRATEGY_EVALUATOR_H_