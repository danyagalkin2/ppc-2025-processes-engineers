#pragma once

#include "galkin_d_sparse_matrix_mul/common/include/common.hpp"

namespace galkin_d_sparse_matrix_mul {

class GalkinDSparseMatMulSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }

  explicit GalkinDSparseMatMulSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace galkin_d_sparse_matrix_mul
