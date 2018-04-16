#include "util/saca.hpp"
#include "saca/example1.hpp"

std::int32_t main(std::int32_t /*argc*/, char const** /*argv*/) {
  sacabench::example1::example1::run_example();
  return 0;
}