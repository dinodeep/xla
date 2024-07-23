// TODO: license

#ifndef XLA_SERVICE_EXPERIMENTAL_SIMPLE_SOLVER_BUILDER_H_
#define XLA_SERVICE_EXPERIMENTAL_SIMPLE_SOLVER_BUILDER_H_

#include "xla/service/experimental/solver_builder.h"

using ::operations_research::MPSolver;
using ::operations_research::MPVariable;

namespace xla {

class SimpleSolverBuilder : SolverBuilder {
public:
  SimpleSolverBuilder();

  // setup variables within the solver
  void CreateVars(std::shared_ptr<InstructionStrategies> strats) override;

  // setup variable constraints
  void AddConstraints(std::shared_ptr<InstructionStrategies> strats) override;

  // setup the objective
  void AddInObjective(std::shared_ptr<InstructionStrategies> strats) override;

  // return the solver built by the solver builder
  bool Solve() override;

  // get the index of the sharding strategy after solving
  int GetStratIdx(std::shared_ptr<InstructionStrategies> strats) override;

private:
  // solver that will be built
  std::unique_ptr<MPSolver> solver_;

  // map to hold the solver variables associated with an instruction
  std::unordered_map<std::shared_ptr<InstructionStrategies>, std::vector<MPVariable*>> var_map_;
  
};

} // xla

#endif // XLA_SERVICE_EXPERIMENTAL_SIMPLE_SOLVER_BUILDER_H_