// TODO: license

#include "xla/service/experimental/sharding_strategy_evaluator.h"

#include "xla/service/experimental/module_cost_evaluator.h"
#include "xla/service/experimental/device_mesh.h"

#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/hlo_pass_interface.h"
#include "xla/service/hlo_pass_pipeline.h"
#include "xla/service/sharding_propagation.h"
#include "xla/service/spmd/stateful_rng_spmd_partitioner.h"

#include "xla/service/experimental/debug.h"

#include "tsl/platform/errors.h"

namespace xla {

namespace {

/*********************************************************/
/* GSPMD Completion                                      */
/*********************************************************/
// sharding_evaluator.{h,cc}

// Major steps prior to evaluating the cost
//  0. clone the original module?
//  1. clear the module of shardings (does GSPMD insert any other metadata?)
//  2. apply the shardings from a strategy
//  3. run GSPMD
//  4. evaluate the cost of the resulting module
//  5. figure out the output sharding of the complete module

// This function runs the sharding propagation pass over an HloModule
void RunShardingPropagation(HloModule* module) {
  // automatically complete the shardings
  HloPassPipeline sharding_pipeline("sharding-propagation");
  sharding_pipeline.AddPass<ShardingPropagation>(
    /* is_spmd */ true,
    /* propagate_metadata */ true,
    /* sharding propagation to output */ absl::Span<const bool>({ true }),
    /* sharding propagation to parameters */ absl::Span<const bool>({ false })
  );

  TF_CHECK_OK(sharding_pipeline.Run(module).status());
}

// This function runs the SpmdPartitioner over an HloModule
void RunSpmdPartitioner(HloModule* module) {
  // fill in communications to produce SPMD program
  HloPassPipeline spmd_pipeline("spmd-partitioner");
  spmd_pipeline.AddPass<spmd::StatefulRngSpmdPartitioner>(
    module->config().num_partitions(),
    module->config().replica_count()
  );

  TF_CHECK_OK(spmd_pipeline.Run(module).status());
}

// This function runs the sharding propagation pipeline pass on the module
HloSharding RunGSPMD(HloModule* module) {

  // TODO: will need to remove manual setting of this eventually
  // TODO: is setting replica_count to 1 okay?
  HloModuleConfig& config = module->mutable_config();
  config.set_num_partitions(DeviceMesh::DeviceCount());
  config.set_replica_count(1);
  config.set_use_spmd_partitioning(true);

  // complete the shardings to the output
  RunShardingPropagation(module);

  // extract the output sharding
  HloInstruction* instr = module->entry_computation()->root_instruction();
  HloSharding out_sharding = instr->sharding();

  // now replace shardings with communication operations
  RunSpmdPartitioner(module);
  return out_sharding;
}

// This function returns the sharding of the entry computation's 
// root instruction
HloSharding GetRootSharding(HloModule* module) {
  HloInstruction* root = module->entry_computation()->root_instruction();
  assert(root->has_sharding());

  return root->sharding();
}

// Returns true if instruction is resharded by exactly 1 communication operation
// Note: this function will return false if there are multiple users that
// perform a resharding because that implies that the sharding for param
// might be used in multiple ways and could intentionally be useful and not
// simply just extra communication
// TODO: could be a bit smarter about this, but could use this function
// to derive the set of shardings strategies that are available in comparison
// to a prior sharding
bool IsInstructionResharded(HloInstruction* param,
    std::shared_ptr<HloSharding> orig_sharding) {

  // if 0 users or more than 1 user, then not just resharded
  if (param->user_count() != 1) {
    return false;
  }

  // get the only unique user of this instruction
  HloInstruction* user = param->users()[0];

  // now determine if it is being resharded
  // Types fo communication primitives
  // - all_gather
  // - all_reduce (but this does some form of summation too)
  // - all_to_all
  // - broadcast
  // - gather
  // - scatter
  // - slice? (this isn't really communication but it could change the sharding)

  // simple implementation
  // just return false if the use is an all-gather along some sharding dimension
  // TODO: if all partitions were taking a slice, would that be a different
  // resharding?
  return user->opcode() == HloOpcode::kAllGather \
      || user->opcode() == HloOpcode::kAllGatherStart;
}

} // namespace


bool IsValidShardingStrat(const HloModule* wrapper_module, 
    ShardingStrategy* strat) {

  // clone the module to avoid modifying it
  std::unique_ptr<HloModule> module = wrapper_module->Clone();

  // apply GSPMD to the module with the sharding strategy
  strat->ApplyToModule(module.get());

  // NOTE: shouldn't really be ignoring this output, poor code design
  // would rather have you separate out the functionality of RunGSPMD
  RunGSPMD(module.get());

  // look through operands and see if it
  HloComputation* entry_computation = module->entry_computation();
  int num_parameters = entry_computation->num_parameters();
  assert(num_parameters == strat->NumOpShardings());

  for (int i = 0; i < num_parameters; i++) {
    HloInstruction* param = entry_computation->parameter_instruction(i);
    std::shared_ptr<HloSharding> orig_sharding = strat->GetOpSharding(i);
    if (IsInstructionResharded(param, orig_sharding)) {
      return false;
    }
  } 


  return true;
}

void EvaluateShardingStrat(const HloModule* module, 
    ShardingStrategy* strat) {

  // clone the module to avoid clobbering future evaluations
  std::unique_ptr<HloModule> eval_module = module->Clone();

  // apply GSPMD to the module with the sharding strategy
  // TODO: should these take in unique pointers or is regulard pointer ok?
  strat->ApplyToModule(eval_module.get());
  strat->set_result_sharding(RunGSPMD(eval_module.get()));

  // now evaluate cost
  ModuleCostEvaluator evaluator;
  strat->set_cost(evaluator.EvaluateCommCost(eval_module.get()));
  strat->set_flops(evaluator.EvaluateFLOPs(eval_module.get()));

  return;
  
}

}