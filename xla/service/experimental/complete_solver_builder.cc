// TODO: license

#include "xla/service/experimental/complete_solver_builder.h"

#include "xla/service/experimental/device_mesh.h"

#include "tsl/platform/logging.h"
#include "tsl/platform/errors.h"

#include "xla/service/experimental/fix_log.h"

namespace xla {

CompleteSolverBuilder::CompleteSolverBuilder(double replicated_flops_prop,
    uint64_t memory_limit) :
    replicated_flops_prop_(replicated_flops_prop),
    memory_limit_(memory_limit),
    solver_(MPSolver::CreateSolver("SCIP")),
    objective_(solver_->MutableObjective()) {

  // NOTE: should check that replicated_flops_prop_ is in a valid range
  double min_prop = ((double) 1) / ((double) (DeviceMesh::DeviceCount()));
  assert(min_prop <= replicated_flops_prop_ && replicated_flops_prop_ <= 1);

  objective_->SetMinimization();
  return;
}

// setup variables within the solver
void CompleteSolverBuilder::CreateVars(std::shared_ptr<InstructionStrategies> strats) {

  if (strats->num_sharding_strats() == 0) {
    return;
  }

  // create variables representing which sharding strategy is chosen
  solver_->MakeBoolVarArray(
    strats->sharding_strats().size(),
    "",
    &var_map_[strats].comp_vars
  );

  // for each user, create a matrix representing the resharding choices
  int num_rows = strats->num_sharding_strats();
  int num_cols;
  for (auto& user_strats : strats->user_strats()) {
    num_cols = user_strats->num_sharding_strats();
    var_map_[strats].resharding_var_matrices.push_back(
      std::make_shared<VariableMatrix>(
        solver_,
        num_rows, num_cols,
        true, 0, 1 /* binary variable specification */
      )
    );
  }

  return;
}

// setup variable constraints
void CompleteSolverBuilder::AddConstraints(std::shared_ptr<InstructionStrategies> strats) {

  // ignore if no sharding strategies for instruction 
  if (strats->num_sharding_strats() == 0) {
    return;
  }

  // access variables representing boolean choice of sharding strategy
  // TODO: rename this to num_strats
  int num_shardings = strats->num_sharding_strats();
  std::vector<MPVariable*>& comp_vars = var_map_[strats].comp_vars;
  std::vector<std::shared_ptr<VariableMatrix>>& user_matrices 
    = var_map_[strats].resharding_var_matrices;
  std::vector<std::shared_ptr<VariableMatrix>> op_matrices
    = GetOpMatrices(strats); 

  assert(num_shardings == comp_vars.size());
  
  // constraints on solely sharding strategy decision:
  // - sums to 1
  LinearExpr expr;
  for (int i = 0; i < num_shardings; i++) {
    expr += comp_vars[i];
  }
  solver_->MakeRowConstraint(expr == 1);

  // constraints on solely resharding decision variable matrices:
  // - sums to 1
  for (std::shared_ptr<VariableMatrix> mat : user_matrices) {
    if (mat->size() > 0) {
      solver_->MakeRowConstraint(mat->Sum() == 1);
    }
  }

  // constraints between sharding strategy decision and decision var matrices
  // - for user resharding matrices, rows of matrix sum to that sharding strat
  for (std::shared_ptr<VariableMatrix> mat : user_matrices) {
    if (mat->size() > 0) {
      assert(num_shardings == user_matrices[user_idx].num_rows());
      for (int r = 0; r < num_shardings; r++) {
        solver_->MakeRowConstraint(mat->SumRow(r) == comp_vars[r]);
      }
    }
  } 

  // - for op resharding matrices, cols of matrix sum to that sharding strat
  for (std::shared_ptr<VariableMatrix> mat : op_matrices) {
    if (mat->size() > 0) {
      assert(num_shardings == user_matrices[user_idx].num_rows());
      for (int c = 0; c < num_shardings; c++) {
        solver_->MakeRowConstraint(mat->SumCol(c) == comp_vars[c]);
      }
    }
  }

  return;
}

void CompleteSolverBuilder::AddComputationConstraint(
    std::vector<std::shared_ptr<InstructionStrategies>> all_strats) {
  
  // constrain computation FLOPs to some proportion of a fully replicated
  // computation to guarantee parallelism to some degree
  LinearExpr total_device_flops;
  LinearExpr fully_replicated_flops;

  // iterate through all instructions that will be sharded
  for (std::shared_ptr<InstructionStrategies> strats : all_strats) {
    
    // unpack boolean variables representing choice of sharding strategies
    int num_shardings = strats->num_sharding_strats();
    std::vector<MPVariable*>& comp_vars = var_map_[strats].comp_vars;
    std::vector<ShardingStrategy>& sharding_strats = strats->sharding_strats();

    // incorporate computation of each sharding strategy of instruction
    // into constraint
    for (int i = 0; i < num_shardings; i++) {
      LinearExpr term(comp_vars[i]);
      term *= sharding_strats[i].flops();
      total_device_flops += term;
    }

    // incorporate fully replicated flops in upper bound
    fully_replicated_flops += strats->fully_replicated_flops();
  }

  // upper bound by a proportion of the number of flops in a fully replicated
  // solution
  fully_replicated_flops *= replicated_flops_prop_;

  // include constraint into solver
  solver_->MakeRowConstraint(total_device_flops <= fully_replicated_flops);

  return;

}

void CompleteSolverBuilder::AddMemoryConstraint(
    std::vector<std::shared_ptr<InstructionStrategies>> all_strats) {

  // Do not create a constraint if there is no memory limit
  if (memory_limit_ == 0) {
    return;
  }

  // iterate through all strategies and add up their memory contributes
  LinearExpr total_memory_usage;

  for (std::shared_ptr<InstructionStrategies> strats : all_strats) {
    // unpack boolean variables representing choice of sharding strategies
    int num_shardings = strats->num_sharding_strats();
    std::vector<MPVariable*>& comp_vars = var_map_[strats].comp_vars;
    std::vector<ShardingStrategy>& sharding_strats = strats->sharding_strats();

    for (int i = 0; i < num_shardings; i++) {
      LinearExpr term(comp_vars[i]);
      term *= sharding_strats[i].memory_bytes();
      total_memory_usage += term;
    }
  }

  // add constraint that memory usage is <= memory limit
  solver_->MakeRowConstraint(total_memory_usage <= memory_limit_);

  return;

}

// setup the objective
void CompleteSolverBuilder::AddInObjective(std::shared_ptr<InstructionStrategies> strats) {

  // ignore if no sharding strategy for instruction
  if (strats->num_sharding_strats() == 0) {
    return;
  }

  // incorporate the comp + comm cost of the sharding strategy into the variable
  int num_strats = strats->num_sharding_strats();
  std::vector<MPVariable*>& comp_vars = var_map_[strats].comp_vars;
  std::vector<ShardingStrategy>& sharding_strats = strats->sharding_strats();
  for (int i = 0; i < num_strats; i++) {
    objective_->SetCoefficient(
      comp_vars[i],
      sharding_strats[i].cost()
    );
  }

  // incorporate the resharding communication costs into each variable matrix
  std::vector<std::shared_ptr<ReshardingCostMatrix>>& cost_matrices 
    = strats->resharding_matrices();
  std::vector<std::shared_ptr<VariableMatrix>>& var_matrices 
    = var_map_[strats].resharding_var_matrices;
  
  assert(var_matrices.size() == cost_matrices.size());
  int num_matrices = var_matrices.size();
  for (int i = 0; i < num_matrices; i++) {
    var_matrices[i]->SetCoefficients(cost_matrices[i]);
  }

  return;
}

// call the solver and return whether found an optimal result
bool CompleteSolverBuilder::Solve() {
  // attempt to solve
  const MPSolver::ResultStatus result_status = solver_->Solve();
  if (result_status != MPSolver::OPTIMAL) {
    return false;
  }

  return true;
}

int CompleteSolverBuilder::GetStratIdx(std::shared_ptr<InstructionStrategies> strats) {

  // ignore if no sharding strategies for instruction
  assert(strats->num_sharding_strats() != 0);

  // get solved variables and return index of one that was solved
  std::vector<MPVariable*> vars = var_map_[strats].comp_vars;

  for (int i = 0; i < vars.size(); i++) {
    if (vars[i]->solution_value() == 1) {
      return i;
    }
  }

  // should not have reached this point with a valid solution
  assert(0);

  return -1;
}

std::vector<std::shared_ptr<VariableMatrix>> CompleteSolverBuilder::GetOpMatrices(
    std::shared_ptr<InstructionStrategies> strats) {

  std::vector<std::shared_ptr<VariableMatrix>> op_matrices;

  for (auto op_strats : strats->operand_strats()) {
    if (op_strats->num_sharding_strats() > 0) {
      int user_idx = op_strats->orig_instr()->UserId(strats->orig_instr());
      op_matrices.push_back(
        var_map_[op_strats].resharding_var_matrices[user_idx]
      );
    }
  }

  return op_matrices;
}

} // xla