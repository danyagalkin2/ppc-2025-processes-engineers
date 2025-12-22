#pragma once

#include "task/include/task.hpp"

namespace galkin_d_ring {

struct Input {
  int src;    // ранг процесса-отправителя
  int dest;   // ранг процесса-получателя
  int count;  // число элементов (int), которые надо передать
};

using InType = Input;
using OutType = int;

using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace galkin_d_ring
