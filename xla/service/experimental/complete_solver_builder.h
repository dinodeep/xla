// TODO: license

#ifndef XLA_SERVICE_EXPERIMENTAL_COMPLETE_SOLVER_BUILDER_H_
#define XLA_SERVICE_EXPERIMENTAL_COMPLETE_SOLVER_BUILDER_H_

#include "xla/service/experimental/solver_builder.h"

#include "xla/service/experimental/variable_matrix.h"

#include "ortools/linear_solver/linear_solver.h"

using ::operations_research::MPSolver;
using ::operations_research::MPObjective;
using ::operations_research::MPVariable;
using ::operations_research::LinearExpr;

namespace xla {

// TODO: make this the implementation of ShardingStrategySelector
// no need for the SimpleSolverBuilder

// This solver builder will ignore the resharding costs between 
// instructions and will only perform the naive optimization of choosing a 
// sharding strategy based off of their costs
class CompleteSolverBuilder : SolverBuilder {
public:
  CompleteSolverBuilder(double replicated_flops_prop, uint64_t memory_limit);

  // setup variables within the solver
  void CreateVars(std::shared_ptr<InstructionStrategies> strats) override final;

  // setup variable constraints
  void AddConstraints(std::shared_ptr<InstructionStrategies> strats) override final;

  void AddComputationConstraint(
    std::vector<std::shared_ptr<InstructionStrategies>> all_strats);

  // setup the objective
  void AddInObjective(std::shared_ptr<InstructionStrategies> strats) override final;

  // return the solver built by the solver builder
  bool Solve() override final;

  // get the index of the sharding strategy after solving
  int GetStratIdx(std::shared_ptr<InstructionStrategies> strats) override final;

private:

  // Maximum proportion of FLOPs that are allowed to be fully replicated
  double replicated_flops_prop_;

  // Maximum amount of bytes of memory that a device can handle at a moment
  // of execution
  uint64_t memory_limit_;

  // returns a vector of the resharding matrices of the operand's to the current
  // instruction, if an operand doesn't have any shardings, then it doesn't
  // have a matrix to contribute into result
  // NOTE: as a result, won't return a 1-to-1 correspondence
  std::vector<std::shared_ptr<VariableMatrix>> GetOpMatrices(
    std::shared_ptr<InstructionStrategies> strats);

  struct VarInfo {
    // comp_vars will be a vector of variables representing the chosen strats
    // in the objective, it's coefficient is a cost of the sharding strat itself
    std::vector<MPVariable*> comp_vars;

    // If there are k users of the instruction, resharding_var.size() == k
    // if the user doesn't have any shardings, then it will be a nullptr
    // in the objective, these var's coefficients are resharding costs
    std::vector<std::shared_ptr<VariableMatrix>> resharding_var_matrices;
  };

  // solver that will be built
  std::shared_ptr<MPSolver> solver_;

  // objective to optimization problem
  MPObjective* const objective_;

  // map to hold the variables for resharding costs in an instruction
  std::unordered_map<std::shared_ptr<InstructionStrategies>, VarInfo> var_map_;
};

} // xla

#endif // XLA_SERVICE_EXPERIMENTAL_SIMPLE_SOLVER_BUILDER_H_