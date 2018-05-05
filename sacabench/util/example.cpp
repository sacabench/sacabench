
#include <iostream>
#include <tuple>
#include <array>
#include <util/induce_sa_dc.hpp>
#include <util/container.hpp>
#include <util/string.hpp>

int main() {
    std::cout << "Running example1" << std::endl;

    sacabench::util::string input_string = {'c', 'a', 'a', 'b', 'a', 'c', 'c',
            'a', 'a', 'b', 'a', 'c', 'a', 'a'};
            
    //positions i mod 3 = 0 of input_string
    sacabench::util::string t_0 = {'c', 'b', 'c', 'b', 'a'}; 
    
    //inverse SA of triplets beginning in positions i mod 3 != 0
    auto isa_12 = sacabench::util::container<size_t> { 4,8,3,7,2,6,10,5,9,1 };
    
    //empty SA which should be filled correctly with method induce_sa_dc
    auto sa_0 = sacabench::util::container<size_t> {0,0,0,0,0};
    
    //run method to test it
    sacabench::util::induce_sa_dc<unsigned char>(t_0, isa_12, sa_0);

    sacabench::util::induce_sa_dc<char>(t_0, isa_12, sa_0);
    
    std::cout << "Finished example1" << std::endl;
}

