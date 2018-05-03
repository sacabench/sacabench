
#include <iostream>
#include <tuple>
#include <array>
#include "induce_sa_dc.hpp"

int main() {
    std::cout << "Running example1" << std::endl;

    std::string input_string = "caabaccaabacaa$";
    std::string t_0 = "cbcba"; //positions i mod 3 = 0 of input_string
    std::array<uint32_t, 10> isa = { 4,8,3,7,2,6,10,5,9,1 };
    std::array<uint32_t, 5> sa0;

    sacabench::util::induce_sa_dc<char>(t_0, isa, sa0);
    
    std::cout << "Finished example1" << std::endl;
}

