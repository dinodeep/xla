// TODO: license

#include "xla/service/experimental/simple_solver_builder.h"

#include "tsl/platform/logging.h"
#include "tsl/platform/errors.h"

#include "xla/service/experimental/fix_log.h"

using ::operations_research::LinearExpr;

namespace xla {


SimpleSolverBuilder::SimpleSolverBuilder() :
    solver_(MPSolver::CreateSolver("SCIP")),
    objective_(solver_->MutableObjective()) {

  objective_->SetMinimization();
  return;
}

// setup variables within the solver
void SimpleSolverBuilder::CreateVars(std::shared_ptr<InstructionStrategies> strats) {

  // ignore if instruction strategies already inside and incorporated
  if (var_map_.count(strats) > 0 && var_map_[strats].size() > 0) {
    return;
  }

  // ignore instruction that doesn't have any sharding strategies
  int num_strats = strats->sharding_strats().size();
  if (num_strats == 0) {
    return;
  }

  // create a variable for each strategy
  solver_->MakeBoolVarArray(
    strats->sharding_strats().size(), 
    "", 
    &var_map_[strats]
  );

  return;
}

// setup variable constraints
void SimpleSolverBuilder::AddConstraints(std::shared_ptr<InstructionStrategies> strats) {

  // ignore if no sharding strategies for instruction 
  if (strats->sharding_strats().size() == 0) {
    return;
  }

  // get vars for current strat
  std::vector<MPVariable*>& vars = var_map_[strats];

  // sum of them should be exactly 1
  LinearExpr var_sum;
  for (int i = 0; i < vars.size(); i++) {
    var_sum += vars[i];
  }

  // add as constraint to problem
  solver_->MakeRowConstraint(var_sum == 1);

  return;
}

// setup the objective
void SimpleSolverBuilder::AddInObjective(std::shared_ptr<InstructionStrategies> strats) {

  // ignore if no sharding strategy for instruction
  if (strats->sharding_strats().size() == 0) {
    return;
  }

  // iterate through variables for strategy and incorporate into objective
  std::vector<ShardingStrategy>& sharding_strats = strats->sharding_strats();
  std::vector<MPVariable*> vars = var_map_[strats];

  // each sharding strategy has it's cost as the coefficient
  assert(vars.size() == sharding_strats.size());
  for (int i = 0; i < vars.size(); i++) {
    objective_->SetCoefficient(
      vars[i],
      sharding_strats[i].cost()
    );
  }

  return;
}

// call the solver and return whether found an optimal result
bool SimpleSolverBuilder::Solve() {
  // attempt to solve
  const MPSolver::ResultStatus result_status = solver_->Solve();
  if (result_status != MPSolver::OPTIMAL) {
    return false;
  }

  return true;
}


int SimpleSolverBuilder::GetStratIdx(std::shared_ptr<InstructionStrategies> strats) {

  // ignore if no sharding strategies for instruction
  assert(strats->num_sharding_strats() != 0); 

  // get solved variables and return index of one that was solved
  std::vector<MPVariable*> vars = var_map_[strats];

  for (int i = 0; i < vars.size(); i++) {
    if (vars[i]->solution_value() == 1) {
      return i;
    }
  }

  // should not have reached this point with a valid solution
  assert(0);

  return -1;
}

} // xla