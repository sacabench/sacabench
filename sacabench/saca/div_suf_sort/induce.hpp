#pragma once

#include <util/span.hpp>
#include "utils.hpp"
#include <util/string.hpp>
#include <iostream>

namespace sacabench::div_suf_sort {

    template<typename sa_index>
    inline static void induce_s_suffixes(util::string_span input,
                                       util::span<bool> suffix_types,
                                       buckets<sa_index>& buckets,
                                       util::span<sa_index> sa,
                                       const size_t max_character) {
      // bit mask: 1000...000
      constexpr sa_index NEGATIVE_MASK = size_t(1)
                                         << (sizeof(sa_index) * 8 - 1);

      for (size_t c0 = max_character; c0 > '\1'; --c0) {
          // c1 = c0 - 1
          // start at rightmost position of L-bucket of c1
          sa_index interval_start = buckets.l_buckets[c0] - 1;
          // end at RMS-bucket[c1, c1 + 1]
          sa_index interval_end =
              buckets.s_buckets[buckets.get_rms_bucket_index(c0 - 1, c0)];
          std::cout << "____________________________________" << std::endl;
          std::cout << "Currently inducing in interval <" << interval_end
                    << "," << interval_start << ">" << std::endl;
          // induce positions for each suffix in range
          // +1 to allow i reaching 0 (because of unsigned types)
          if(interval_end == 0) { break;}
          for (sa_index i = interval_start; i >= interval_end; --i) {
              // Index 0 found - cannot induce anything -> skip
              if(sa[i] == 0) { continue; }
              
              if ((sa[i] & NEGATIVE_MASK) == 0) {
                  // entry is not negative -> induce predecessor
                  std::cout << "Using index " << sa[i] << " at pos " << i
                            << " for inducing" << std::endl;
                  // insert suffix i-1 at rightmost free index of
                  // associated S-bucket
                  size_t destination_bucket = buckets.get_s_bucket_index(
                      input[sa[i] - 1], input[sa[i]]);
                  std::cout << "Check if index " << sa[i] - 2 << " is l-type"
                            << std::endl;
                  if (sa[i]-1 > 0 && sa_types<sa_index>::is_l_type(sa[i] - 2,
                                                    suffix_types)) {
                      // Check if index is used to induce in current step
                      // (induce s-suffixes)
                      std::cout << "Index " << sa[i] - 2
                                << " is l-type -> negate index of "
                                << sa[i] - 1 << " inserted at pos "
                                << buckets.s_buckets[destination_bucket]
                                << " instead" << std::endl;
                      sa[buckets.s_buckets[destination_bucket]--] =
                          (sa[i] - 1) ^ NEGATIVE_MASK;
                  } else {
                      std::cout << "Inserted index " << sa[i] - 1
                                << " at position "
                                << buckets.s_buckets[destination_bucket]
                                << std::endl;
                      sa[buckets.s_buckets[destination_bucket]--] = sa[i] - 1;
                  }
              }
              std::cout << "Toggled induce flag at pos " << i << std::endl;
              // toggle flag
              sa[i] ^= NEGATIVE_MASK;
          }
      }

      // "$" is the first index
      sa[0] = input.size() - 1;
      std::cout << "Inserted sentinel at front of SA." << std::endl;

      // if predecessor is S-suffix
      if (input[input.size() - 2] < input[input.size() - 1]) {
          sa[0] |= NEGATIVE_MASK;
          std::cout
              << "Negated index of sentinel because predecessor is s-suffix."
              << std::endl;
      }
    }

    template<typename sa_index>
    inline static void induce_l_suffixes(util::string_span input,
                                       buckets<sa_index>& buckets,
                                       util::span<sa_index> sa) {
      // bit mask: 1000...000
      constexpr sa_index NEGATIVE_MASK = size_t(1)
                                         << (sizeof(sa_index) * 8 - 1);

      for (sa_index i = 0; i < sa.size(); ++i) {
          if (sa[i] == 0) {
              std::cout << "Skipped index 0 because it has no predecessor" << std::endl;
              continue;
          }
          if ((sa[i] & NEGATIVE_MASK) > 0) {
              // entry is negative: sa[i]-1 already induced -> remove flag
              sa[i] ^= NEGATIVE_MASK;
              std::cout << "Index " << sa[i] - 1 << " already induced -> skip"
                        << std::endl;
          } else {
              std::cout << "Using index " << sa[i] << " at pos " << i
                        << " for inducing" << std::endl;
              // predecessor has yet to be induced
              sa_index insert_position =
                  buckets.l_buckets[input[sa[i] - 1]]++;
              sa[insert_position] = sa[i] - 1;
              if (sa[i] - 1 > 0 && input[sa[i] - 2] < input[sa[i] - 1]) {
                  std::cout << "Index " << sa[i] - 2
                            << " is s-suffix -> inserted negated index "
                            << sa[i] - 1 << " at pos " << insert_position
                            << std::endl;
                  // predecessor of induced index is S-suffix
                  sa[insert_position] |= NEGATIVE_MASK;
              } else {
                  std::cout << "Inserted index " << sa[i] - 1 << " at pos "
                            << insert_position << std::endl;
              }
          }
      }
    }
}
