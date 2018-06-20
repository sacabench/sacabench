/*******************************************************************************
 * https://github.com/thrill/thrill/blob/master/thrill/common/uint_types.hpp
 *
 * Class representing a 40-bit or 48-bit unsigned integer encoded in five or
 * six bytes.
 *
 * Part of Project Thrill - http://project-thrill.org
 *
 * Copyright (C) 2013 Timo Bingmann <tb@panthema.net>
 *
 * All rights reserved. Published under the BSD-2 license in the LICENSE file.
 ******************************************************************************/

#pragma once

// Includes SB_LIKELY, SB_UNLIKELY and SB_ATTRIBUTE_PACKED.
#include "macros.hpp"

#include <type_traits>

#include <cassert>
#include <limits>
#include <ostream>

#define generate_relop(OP, MACROTYPE)                                          \
    friend constexpr bool operator OP(const MACROTYPE& lhs,                    \
                                      const UIntPair<High>& rhs) {             \
        return lhs OP static_cast<unsigned long long>(rhs);                    \
    }                                                                          \
                                                                               \
    friend constexpr bool operator OP(const UIntPair<High>& lhs,               \
                                      const MACROTYPE& rhs) {                  \
        return static_cast<unsigned long long>(lhs) OP rhs;                    \
    }

#define generate_relop_this(OP)                                                \
    friend constexpr bool operator OP(const UIntPair<High>& lhs,               \
                                      const UIntPair<High>& rhs) {             \
        return static_cast<unsigned long long int>(lhs)                        \
            OP static_cast<unsigned long long int>(rhs);                       \
    }

namespace sacabench::util {

/*!
 * Construct an 40-bit or 48-bit unsigned integer stored in five or six bytes.
 *
 * The purpose of this class is to provide integers with smaller data storage
 * footprints when more than 32-bit, but less than 64-bit indexes are
 * needed. This is commonly the case for storing file offsets and indexes. Here
 * smaller types currently suffice for files < 1 TiB or < 16 TiB.
 *
 * The class combines a 32-bit integer with a HighType (either 8-bit or 16-bit)
 * to get a larger type. Only unsigned values are supported, which fits the
 * general application of file offsets.
 *
 * Calculation in UIntPair are generally done by transforming everything to
 * 64-bit data type, so that 64-bit register arithmetic can be used. The
 * exception here is \b increment and \b decrement, which is done directly on
 * the lower/higher part. Not all arithmetic operations are supported, patches
 * welcome if you really need the operations.
 */
#if defined(_MSC_VER)
#pragma pack(push, 1)
#endif
template <typename High_>
class UIntPair {
public:
    //! lower part type, always 32-bit
    using Low = uint32_t;
    //! higher part type, currently either 8-bit or 16-bit
    using High = High_;

private:
    //! member containing lower significant integer value
    Low low_;
    //! member containing higher significant integer value
    High high_;

    //! return highest value storable in lower part, also used as a mask.
    constexpr static unsigned low_max() {
        return std::numeric_limits<Low>::max();
    }

    //! number of bits in the lower integer part, used a bit shift value.
    constexpr static size_t low_bits = 8 * sizeof(Low);

    //! return highest value storable in higher part, also used as a mask.
    constexpr static unsigned high_max() {
        return std::numeric_limits<High>::max();
    }

    //! number of bits in the higher integer part, used a bit shift value.
    constexpr static size_t high_bits = 8 * sizeof(High);

public:
    //! number of binary digits (bits) in UIntPair
    static constexpr size_t digits = low_bits + high_bits;

    //! number of bytes in UIntPair
    static constexpr size_t bytes = sizeof(Low) + sizeof(High);

    // compile-time assertions about size of Low
    static_assert(8 * sizeof(Low) == 32, "sizeof Low is 32-bit");
    static_assert(digits / 8 == bytes, "digit and bytes ratio is wrong");

    //! empty constructor, does not even initialize to zero!
    constexpr UIntPair() = default;

    //! construct unit pair from lower and higher parts.
    constexpr UIntPair(const Low& l, const High& h) : low_(l), high_(h) {}

    //! copy constructor
    constexpr UIntPair(const UIntPair&) = default;
    //! move constructor
    constexpr UIntPair(UIntPair&&) = default;

    static_assert(sizeof(unsigned int) * 8 == 32,
                  "make sure unsigned int is uint32_t");

    //! const from a simple 32-bit unsigned integer
    constexpr UIntPair(const unsigned int& a) // NOLINT
        : low_(a), high_(0) {}

    //! const from a simple 32-bit signed integer
    constexpr UIntPair(const signed int& a) // NOLINT
        : low_(a), high_(0) {
        if (a >= 0)
            low_ = a;
        else
            low_ = a, high_ = (High)high_max();
    }

    //! construct from an 64-bit unsigned integer
    constexpr UIntPair(const unsigned long long& a) // NOLINT
        : low_((Low)(a & low_max())),
          high_((High)((a >> low_bits) & high_max())) {
        // check for overflow
        assert((a >> (low_bits + high_bits)) == 0);
    }

    //! construct from an 32-bit or 64-bit signed integer
    constexpr UIntPair(const unsigned long& a) // NOLINT
        : UIntPair(static_cast<unsigned long long>(a)) {}

    //! construct from an 32-bit or 64-bit signed integer
    constexpr UIntPair(const signed long& a) // NOLINT
        : UIntPair(static_cast<signed long long>(a)) {}

    //! construct from an 64-bit signed integer
    constexpr UIntPair(const signed long long& a) // NOLINT
        : UIntPair(static_cast<unsigned long long>(a)) {}

    //! copy assignment operator
    constexpr UIntPair& operator=(const UIntPair&) = default;
    //! move assignment operator
    constexpr UIntPair& operator=(UIntPair&&) = default;

private:
    //! return the number as an uint64 (unsigned long long)
    constexpr uint64_t ull() const {
        return ((uint64_t)high_) << low_bits | (uint64_t)low_;
    }

    //! return the number as a uint64_t
    constexpr uint64_t u64() const {
        return ((uint64_t)high_) << low_bits | (uint64_t)low_;
    }

public:
    generate_relop_this(<);
    generate_relop(<, unsigned char);
    generate_relop(<, unsigned short int);
    generate_relop(<, unsigned int);
    generate_relop(<, unsigned long int);
    generate_relop(<, unsigned long long int);

    generate_relop_this(<=);
    generate_relop(<=, unsigned char);
    generate_relop(<=, unsigned short int);
    generate_relop(<=, unsigned int);
    generate_relop(<=, unsigned long int);
    generate_relop(<=, unsigned long long int);

    generate_relop_this(>);
    generate_relop(>, unsigned char);
    generate_relop(>, unsigned short int);
    generate_relop(>, unsigned int);
    generate_relop(>, unsigned long int);
    generate_relop(>, unsigned long long int);

    generate_relop_this(>=);
    generate_relop(>=, unsigned char);
    generate_relop(>=, unsigned short int);
    generate_relop(>=, unsigned int);
    generate_relop(>=, unsigned long int);
    generate_relop(>=, unsigned long long int);

    generate_relop_this(==);
    generate_relop(==, unsigned char);
    generate_relop(==, unsigned short int);
    generate_relop(==, unsigned int);
    generate_relop(==, unsigned long int);
    generate_relop(==, unsigned long long int);

    generate_relop_this(!=);
    generate_relop(!=, unsigned char);
    generate_relop(!=, unsigned short int);
    generate_relop(!=, unsigned int);
    generate_relop(!=, unsigned long int);
    generate_relop(!=, unsigned long long int);

    //! implicit cast to an unsigned long long
    constexpr operator uint64_t() const { return ull(); }

    //! prefix increment operator (directly manipulates the integer parts)
    UIntPair& operator++() {
        if (SB_UNLIKELY(low_ == low_max()))
            ++high_, low_ = 0;
        else
            ++low_;
        return *this;
    }

    //! prefix decrement operator (directly manipulates the integer parts)
    UIntPair& operator--() {
        if (SB_UNLIKELY(low_ == 0))
            --high_, low_ = (Low)low_max();
        else
            --low_;
        return *this;
    }

    //! suffix increment operator (directly manipulates the integer parts)
    UIntPair operator++(int) {
        auto copy = *this;
        ++(*this);
        return copy;
    }

    //! suffix decrement operator (directly manipulates the integer parts)
    UIntPair operator--(int) {
        auto copy = *this;
        --(*this);
        return copy;
    }

    //! addition operator (uses 64-bit arithmetic)
    UIntPair& operator+=(const UIntPair& b) {
        uint64_t add = low_ + uint64_t(b.low_);
        low_ = (Low)(add & low_max());
        high_ = (High)(high_ + b.high_ + ((add >> low_bits) & high_max()));
        return *this;
    }

    //! addition operator (uses 64-bit arithmetic)
    UIntPair operator+(const UIntPair& b) const {
        uint64_t add = low_ + uint64_t(b.low_);
        return UIntPair(
            (Low)(add & low_max()),
            (High)(high_ + b.high_ + ((add >> low_bits) & high_max())));
    }

    //! subtraction operator (uses 64-bit arithmetic)
    UIntPair& operator-=(const UIntPair& b) {
        uint64_t sub = low_ - uint64_t(b.low_);
        low_ = (Low)(sub & low_max());
        high_ = (High)(high_ - b.high_ + ((sub >> low_bits) & high_max()));
        return *this;
    }

    //! subtraction operator (uses 64-bit arithmetic)
    UIntPair operator-(const UIntPair& b) const {
        uint64_t sub = low_ - uint64_t(b.low_);
        return UIntPair(
            (Low)(sub & low_max()),
            (High)(high_ - b.high_ + ((sub >> low_bits) & high_max())));
    }

    //! make a UIntPair outputtable via iostreams, using unsigned long long.
    friend std::ostream& operator<<(std::ostream& os, const UIntPair& a) {
        return os << a.ull();
    }

    //! return an UIntPair instance containing the smallest value possible
    static constexpr UIntPair min() {
        return UIntPair(std::numeric_limits<Low>::min(),
                        std::numeric_limits<High>::min());
    }

    //! return an UIntPair instance containing the largest value possible
    static constexpr UIntPair max() {
        return UIntPair(std::numeric_limits<Low>::max(),
                        std::numeric_limits<High>::max());
    }
} SB_ATTRIBUTE_PACKED;
#if defined(_MSC_VER)
#pragma pack(pop)
#endif

//! Construct a 40-bit unsigned integer stored in five bytes.
using uint40 = UIntPair<uint8_t>;

//! Construct a 48-bit unsigned integer stored in six bytes.
using uint48 = UIntPair<uint16_t>;

// compile-time assertions about size of our data structure, this tests packing
// of structures by the compiler
static_assert(sizeof(uint40) == 5, "sizeof uint40 is wrong");
static_assert(sizeof(uint48) == 6, "sizeof uint48 is wrong");

} // namespace sacabench::util

namespace std {

//! template class providing some numeric_limits fields for UIntPair types.
template <typename HighType>
class numeric_limits<sacabench::util::UIntPair<HighType>> {
public:
    using UIntPair = sacabench::util::UIntPair<HighType>;

    //! yes we have information about UIntPair
    static const bool is_specialized = true;

    //! return an UIntPair instance containing the smallest value possible
    static constexpr UIntPair min() { return UIntPair::min(); }

    //! return an UIntPair instance containing the largest value possible
    static constexpr UIntPair max() { return UIntPair::max(); }

    //! return an UIntPair instance containing the smallest value possible
    static constexpr UIntPair lowest() { return min(); }

    //! unit_pair types are unsigned
    static const bool is_signed = false;

    //! UIntPair types are integers
    static const bool is_integer = true;

    //! unit_pair types contain exact integers
    static const bool is_exact = true;

    //! unit_pair radix is binary
    static const int radix = 2;

    //! number of binary digits (bits) in UIntPair
    static const int digits = UIntPair::digits;

    //! epsilon is zero
    static const UIntPair epsilon() { return UIntPair(0, 0); }

    //! rounding error is zero
    static const UIntPair round_error() { return UIntPair(0, 0); }

    //! no exponent
    static const int min_exponent = 0;

    //! no exponent
    static const int min_exponent10 = 0;

    //! no exponent
    static const int max_exponent = 0;

    //! no exponent
    static const int max_exponent10 = 0;

    //! no infinity
    static const bool has_infinity = false;
};

} // namespace std
