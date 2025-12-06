#include "galkin_d_ring/seq/include/ops_seq.hpp"

namespace galkin_d_ring {

GalkinDRingSEQ::GalkinDRingSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool GalkinDRingSEQ::ValidationImpl() {
  return true;
}

bool GalkinDRingSEQ::PreProcessingImpl() {
  GetOutput() = 0;
  return true;
}

bool GalkinDRingSEQ::RunImpl() {

  const auto &in = GetInput();

  if (in.count <= 0 || in.src < 0 || in.dest < 0) {
    GetOutput() = 0;
    return true;
  }

  GetOutput() = 1;
  return true;
}

bool GalkinDRingSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace galkin_d_ring
