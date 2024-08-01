// TODO: add copyright 

#include "xla/service/experimental/auto_parallel.h"

#include "xla/service/experimental/complete_strategy_graph.h"
#include "xla/service/experimental/sharding_strategy_selector.h"

#include "xla/service/experimental/debug.h"


namespace xla {

namespace {

// TODO: determine if this is a valid approach to determine 
// if the module is the main module for the computations
bool ShardableModule(HloModule* module) {

  std::string name = module->name();
  return name.find("convert_element_type") == std::string::npos &&
    name.find("broadcast_in_dim") == std::string::npos &&
    name.find("_multi_slice") == std::string::npos;
}

}   // namespace

/*********************************************************/
/* AutoParallelizer Pass Implementation                  */
/*********************************************************/

// overriden functions from class
// modifies the sharding specification of the instructions in the module
// TODO: need to ignore the default modules that are not the core computation
//  e.g. jit_convert_element_type, jit_broadcast_in_dim, jit__multi_slice
// otherwise, just too much time spent on these things
absl::StatusOr<bool> AutoParallelizer::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {

  VLOG(5) << "Starting AutoParallelizer Run";

  if (!ShardableModule(module)) {
    VLOG(5) << LOG_HEADER(0) << "module = " 
      << module->name() << " not shardable";
    return false;
  }

  // create a clone of the module, then run off of that 
  VLOG(5) << LOG_HEADER(0) << "module: " << module->name();

  std::unordered_map<HloInstruction*, 
    std::shared_ptr<InstructionStrategies>> info_map;

  // TODO: shouldn't I only be doing this for the main computation?
  // construct relevant sharding information
  for (HloComputation* computation : module->computations()) {
    for (HloInstruction* instr : computation->instructions()) {
      assert(info_map.count(instr) == 0);
      info_map[instr] = std::make_shared<InstructionStrategies>(instr);
    }
  }
  CompleteStrategyGraph(info_map);

  // select optimal sharding strategies
  ShardingStrategySelector selector;
  bool successful = selector.Select(info_map);
  VLOG(5) << "Selector success: " << successful;

  assert(0);
  PrintModuleInfo(module);

  VLOG(5) << "Done AutoParallelizer Run";
  
  return true;
}

}   // namespace xla