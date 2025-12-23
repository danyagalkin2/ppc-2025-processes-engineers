#include "galkin_d_ring/seq/include/ops_seq.hpp"

#include "galkin_d_ring/common/include/common.hpp"

namespace galkin_d_ring {

GalkinDRingSEQ::GalkinDRingSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool GalkinDRingSEQ::ValidationImpl() {
  const auto &in = GetInput();

  const bool ok_count = (in.count > 0);
  const bool ok_src = (in.src == 0);
  const bool ok_dest = (in.dest == 0);

  return ok_count && ok_src && ok_dest;
}

bool GalkinDRingSEQ::PreProcessingImpl() {
  GetOutput() = 0;
  return true;
}

bool GalkinDRingSEQ::RunImpl() {
  GetOutput() = 1;
  return true;
}

bool GalkinDRingSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace galkin_d_ring
