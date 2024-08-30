// TODO: license

#ifndef XLA_SERVICE_EXPERIMENTAL_INSTRUCTION_STRATEGIES_H_
#define XLA_SERVICE_EXPERIMENTAL_INSTRUCTION_STRATEGIES_H_

#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/experimental/sharding_strategy.h"
#include "xla/service/experimental/resharding_cost_matrix.h"

#include <vector>

namespace xla {

class InstructionStrategies {
public:
  InstructionStrategies(HloInstruction* orig_instr);
  ~InstructionStrategies() = default;
  InstructionStrategies(const InstructionStrategies& info) = default;

  // accessors
  std::vector<ShardingStrategy>& sharding_strats() { 
    return sharding_strats_;
  };

  int num_sharding_strats() {
    return sharding_strats_.size();
  }

  void set_operand_strats(
      std::vector<std::shared_ptr<InstructionStrategies>>& operand_strats) {
    operand_strats_ = operand_strats;
  }

  std::vector<std::shared_ptr<InstructionStrategies>>& operand_strats() {
    return operand_strats_;
  }

  void set_user_strats(
      std::vector<std::shared_ptr<InstructionStrategies>>& user_strats) {
    user_strats_ = user_strats;
  }

  std::vector<std::shared_ptr<InstructionStrategies>>& user_strats() {
    return user_strats_;
  }

  void set_resharding_matrices(
      std::vector<std::shared_ptr<ReshardingCostMatrix>>& resharding_matrices) {
    resharding_matrices_ = resharding_matrices;
  }

  std::vector<std::shared_ptr<ReshardingCostMatrix>>& resharding_matrices() {
    return resharding_matrices_;
  }

  HloInstruction* orig_instr() { return orig_instr_; }

  uint64_t fully_replicated_flops() { return fully_replicated_flops_; }

  // takes the index of sharding_strats_ and sets that strategy's
  // result sharding to the original instruction
  void set_chosen_result_sharding(int idx);

private:

  // Points to the original instruction that will have its
  // sharding strategies enumerated. Eventually, this instruction
  // will be modified with a sharding strategy provided by the solvers
  HloInstruction* orig_instr_;

  // Pointers to strategies of operands of this instruction
  std::vector<std::shared_ptr<InstructionStrategies>> operand_strats_;

  // Pointers to strategies of users of this instruction
  std::vector<std::shared_ptr<InstructionStrategies>> user_strats_;

  // Pointers to resharding matrices for each user of the instruction
  std::vector<std::shared_ptr<ReshardingCostMatrix>> resharding_matrices_;

  // vector of sharding strategies for the given instruction
  std::vector<ShardingStrategy> sharding_strats_;

  // Number of flops of computation this instruction requires when
  // fully replicated on a device
  uint64_t fully_replicated_flops_;

};

} // xla

#endif // XLA_SERVICE_EXPERIMENTAL_INSTRUCTION_STRATEGIES_H_