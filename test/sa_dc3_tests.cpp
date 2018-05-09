/*******************************************************************************
 * Copyright (C) 2018 Johannes Bahne <johannes.bahne@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#include <gtest/gtest.h>
#include <util/induce_sa_dc.hpp>
#include <util/string.hpp>
#include <util/container.hpp>
#include <saca/dc3.hpp>


TEST(DC, induce) {    
    sacabench::util::string input_string = {'c', 'a', 'a', 'b', 'a', 'c', 'c',
            'a', 'a', 'b', 'a', 'c', 'a', 'a'};
            
    
    //run method to test it
    sacabench::saca::determine_triplets<unsigned char>(input_string);
    
}

