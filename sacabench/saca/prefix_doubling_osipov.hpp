#pragma once

#include <util/alphabet.hpp>
#include <util/assertions.hpp>
#include <util/compare.hpp>
#include <util/container.hpp>
#include <util/span.hpp>
#include <util/string.hpp>


namespace sacabench::osipov {
public:
    static constexpr size_t EXTRA_SENTINELS = 1;
    static constexpr char const* NAME = "Osipov_sequential";
    static constexpr char const* DESCRIPTION =
        "Prefix Doubling approach for parallel gpu computation as sequential "
        "approach";

        
    template <typename sa_index>
    static void construct_sa(util::string_span text,
                             util::alphabet const& alphabet,
                             util::span<sa_index> out_sa) {
        // TODO: Fill me with my algorithm
    }
    
private:

    static void mark_singletons(util::span<sa_index> sa, util::span<sa_index> isa) {
        util::container<bool> flags = make_container<flags>(sa.size());
        flags[0] = true;
        // Set flags if predecessor has different rank.
        for(size_t i=1; i < sa.size(); ++i) {
            flags[i] = isa[sa[i-1]] != isa[sa[i]] ? true : false;
        }
        for(size_t i=0; i < sa.size()-1; ++i) {
            // flags corresponding to predecessor having a different rank, i.e. 
            // suffix sa[i] has different rank than its predecessor and successor.
            if(flags[i] && flags[i+1]) { isa[sa[i]] = -isa[sa[i]];}
        }
        // Check for last position - is singleton, if it has a different rank
        // than its predecessor (because of missing successor).
        if(flags[sa.size()-1]) {isa[sa[sa.size()-1]] = -isa[sa[sa.size()-1]];}
    }
    
    template <typename sa_index>
    static void prefix_doubling_sequential(util::string_span text, 
                             util::alphabet const& alphabet, 
                             util::span<sa_index> out_sa) {
        std::cout << "Starting Osipov sequential." << std::endl;
        
        util::container<sa_index> isa_container = util::make_container(out_sa.size());
        util::container<sa_index> sa_container = util::make_container(out_sa.size());
        util::span<sa_index> sa = util::span<sa_index>(sa_container&);
    }
}