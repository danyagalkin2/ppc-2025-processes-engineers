#include "galkin_d_ring/seq/include/ops_seq.hpp"

#include <vector>

namespace galkin_d_ring {

GalkinDRingSEQ::GalkinDRingSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool GalkinDRingSEQ::ValidationImpl() {
  const auto &in = GetInput();

  const bool src_ok = (in.src == 0);
  const bool dest_ok = (in.dest == 0);
  const bool count_ok = (in.count > 0);

  return src_ok && dest_ok && count_ok;
}

bool GalkinDRingSEQ::PreProcessingImpl() {
  GetOutput() = 0;
  return true;
}

bool GalkinDRingSEQ::RunImpl() {
  const auto &in = GetInput();

  const int count = in.count;

  std::vector<int> buffer(count);
  for (int i = 0; i < count; ++i) {
    buffer[i] = i + 1;
  }

  int local_ok = 1;
  for (int i = 0; i < count; ++i) {
    const int expected = i + 1;
    if (buffer[i] != expected) {
      local_ok = 0;
      break;
    }
  }

  GetOutput() = local_ok;
  return true;
}

bool GalkinDRingSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace galkin_d_ring
