
#include "xla/service/experimental/sharding_strategy_selector.h"

#include "xla/service/experimental/complete_solver_builder.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/errors.h"
#include "xla/service/experimental/fix_log.h"

#define REPLICATED_FLOPS_PROP 0.2

namespace xla {

// Sets the shardings of the HloInstructions based on the best sharding strategy
// selected from the solver
// NOTE: make map an input by reference
bool ShardingStrategySelector::Select(std::unordered_map<HloInstruction*, 
    std::shared_ptr<InstructionStrategies>> strat_map) {

  // initialize a builder
  CompleteSolverBuilder builder(REPLICATED_FLOPS_PROP);

  // create variables, construct their constraints, and add to the objective
  for (auto& [instr, strats] : strat_map) {
    builder.CreateVars(strats);
  }

  for (auto& [instr, strats] : strat_map) {
    builder.AddConstraints(strats);
  }

  std::vector<std::shared_ptr<InstructionStrategies>> all_strats;
  for (auto& [instr, strats] : strat_map) {
    all_strats.push_back(strats);
  }
  builder.AddComputationConstraint(all_strats);

  for (auto& [instr, strats] : strat_map) {
    builder.AddInObjective(strats);
  }

  // solve the problem
  if (!builder.Solve()) {
    return false;
  }

  // success, determine which sharding to load into the instruction
  for (auto& [instr, strats] : strat_map) {
    if (strats->num_sharding_strats() > 0) {
      strats->set_chosen_strat(builder.GetStratIdx(strats));
    }
  }

  return true;
}

} // xla