#pragma once

#include <cmath>
#include <tuple>

#include "task/include/task.hpp"

namespace galkin_d_trapezoid_method {

struct Input {
  double a;     // левая граница интегрирования
  double b;     // правая граница
  int n;        // число разбиений (трапеций)
  int func_id;  // id функции для интегрирования
};
using InType = Input;
using OutType = double;
using TestType = std::tuple<int, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

enum class FunctionId : int {
  Linear = 0,     // f(x) = x
  Quadratic = 1,  // f(x) = x^2
  Sin = 2         // f(x) = sin(x)
};

inline double Function(double x, int func_id) {
  switch (static_cast<FunctionId>(func_id)) {
    case FunctionId::Linear:
      return x;  // f(x) = x
    case FunctionId::Quadratic:
      return x * x;  // f(x) = x^2
    case FunctionId::Sin:
      return std::sin(x);  // f(x) = sin(x)
    default:
      return 0.0;
  }
}

inline double GetExactIntegral(const InType &in) {
  const double a = in.a;
  const double b = in.b;

  switch (static_cast<FunctionId>(in.func_id)) {
    case FunctionId::Linear:
      return (b * b - a * a) / 2.0;
    case FunctionId::Quadratic:
      return (b * b * b - a * a * a) / 3.0;
    case FunctionId::Sin:
      return std::cos(a) - std::cos(b);
    default:
      return 0.0;
  }
}

}  // namespace galkin_d_trapezoid_method
