// TODO: license

#include "xla/service/experimental/resharding_cost_matrix.h"

#include "xla/service/experimental/resharding_cost_evaluator.h"
#include "xla/hlo/ir/hlo_instruction.h"

namespace xla {

namespace {

std::string UIntToShortString(uint64_t n) {
  uint64_t K = 1000;
  uint64_t M = 1000000;
  uint64_t G = 1000000000;
  uint64_t T = 1000000000000;

  if (n < K) {
    return std::to_string(n);
  } else if (K <= n && n < M) {
    return std::to_string((uint64_t)(n / K)) + "K";
  } else if (M <= n && n < G) {
    return std::to_string((uint64_t)(n / M)) + "M";
  } else if (G < n && n <= T) {
    return std::to_string((uint64_t)(n / G)) + "G";
  } else {
    return std::to_string((uint64_t)(n / T)) + "T";
  }

}

} // namespace

ReshardingCostMatrix::ReshardingCostMatrix(const Shape& shape, 
    std::vector<std::shared_ptr<HloSharding>>& strats1, 
    std::vector<std::shared_ptr<HloSharding>>& strats2) :
      num_rows_(strats1.size()), 
      num_cols_(strats2.size()),
      costs_(num_rows_, std::vector<uint64_t>(num_cols_, 0)) {

  ReshardingCostEvaluator evaluator;
  assert(num_rows_ == strats1.size() && num_cols_ == strats2.size());

  // iterate through each pair and fill the costs_ matrix
  for (int r = 0; r < num_rows_; r++) {
    for (int c = 0; c < num_cols_; c++) {
      costs_[r][c] = evaluator.Evaluate(
        shape, 
        *strats1[r].get(), 
        *strats2[c].get());
    }
  }

  return;

}

uint64_t ReshardingCostMatrix::CostAt(int r, int c) {
  assert(0 <= r && r < num_rows());
  assert(0 <= c && c < num_cols());

  return costs_[r][c];
}

std::string ReshardingCostMatrix::ToString() {

  std::string s = "[" + std::to_string(num_rows_) + "," + std::to_string(num_cols_) + "]" + "\n";
  for (int r = 0; r < num_rows(); r++) {
    for (int c = 0; c < num_cols(); c++) {
      s += UIntToShortString(costs_[r][c]) + " ";
    }
    s += "\n";
  }

  return s;
}

} // xla