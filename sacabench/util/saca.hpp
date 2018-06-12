/*******************************************************************************
 * Copyright (C) 2018 Florian Kurpicz <florian.kurpicz@tu-dortmund.de>
 * Copyright (C) 2018 Marvin Löbel <loebel.marvin@gmail.com>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#pragma once

#include <functional>
#include <string>
#include <vector>

#include "alphabet.hpp"
#include "container.hpp"
#include "span.hpp"
#include "string.hpp"
#include "uint_types.hpp"

namespace sacabench::util {

/// A wrapper around a suffix array container with extra sentinel values.
///
/// It allows a uniform view of the SA, with entries for extra sentinel
/// characters removed.
template <typename sa_index_t>
class uniform_sa_type {
    size_t m_extra_sentinels;
    container<sa_index_t> m_sa;

public:
    inline uniform_sa_type(size_t extra_sentinels, container<sa_index_t>&& sa)
        : m_extra_sentinels(extra_sentinels), m_sa(std::move(sa)) {}

    /// Return number of sentinel characters
    inline size_t extra_sentinels() { return m_extra_sentinels; }

    /// Return uniform suffix array without sentinel positions.
    inline span<sa_index_t> sa_without_sentinels() {
        return m_sa.slice(extra_sentinels());
    }

    /// Return original suffix array, with potential sentinel positions.
    inline span<sa_index_t> sa_with_sentinels() { return m_sa; }
};

/// A type that represents a input text before any allocation.
struct text_initializer {
    /// Size of the text. In bytes, without any sentinel values.
    size_t text_size = 0;

    /// Initializer function, that writes the text to the passed
    /// span of size `text_size`.
    std::function<void(span<character>)> initializer;
};

/// Creates a `text_initializer` that initializes with a `string_span`.
inline text_initializer text_initializer_from_span(string_span text) {
    return {
        text.size(),
        [=](span<character> s) {
            for (size_t i = 0; i < s.size(); i++) {
                s[i] = text[i];
            }
        },
    };
}

/// Prepares SA and Text containers, and calls the given SACA Algorithm with
/// them.
///
/// \param text_init Initializer for the text.
template <typename Algorithm, typename sa_index_t>
uniform_sa_type<sa_index_t>
prepare_and_construct_sa(text_initializer const& text_init) {
    size_t extra_sentinels = Algorithm::EXTRA_SENTINELS;
    size_t text_size = text_init.text_size;
    auto const& init = text_init.initializer;

    auto output = make_container<sa_index_t>(text_size + extra_sentinels);
    auto text_with_sentinels = string(text_size + extra_sentinels);

    auto text = text_with_sentinels.slice(0, text_size);

    init(text);

    IF_DEBUG({
        for (size_t i = 0; i < text.size(); i++) {
            DCHECK_MSG(text[i] != 0, "Input byte "
                                         << i
                                         << " has value 0, which is reserved "
                                            "for the terminating sentinel!");
        }
    })

    auto alph = apply_effective_alphabet(text);

    {
        span<sa_index_t> out_sa = output;
        string_span readonly_text_with_sentinels = text_with_sentinels;
        Algorithm::construct_sa(readonly_text_with_sentinels, alph, out_sa);
    }

    return uniform_sa_type<sa_index_t>{extra_sentinels, std::move(output)};
}

class saca;

/// List of SACAs in the registry.
class saca_list {
public:
    saca_list(const saca_list&) = delete;
    saca_list(saca_list&&) = delete;
    void operator=(const saca_list&) = delete;
    void operator=(saca_list&&) = delete;

    /// Get a single static instance of the list. (Singleton pattern)
    static saca_list& get() {
        static saca_list list;
        return list;
    }

    /// Register a SACA. This gets called automatically by the
    /// `SACA_REGISTER` macro.
    void register_saca(saca const* algorithm) {
        algorithms_.push_back(algorithm);
    }

    // Iterators

    inline auto begin() { return algorithms_.begin(); }
    inline auto end() { return algorithms_.end(); }
    inline auto cbegin() { return algorithms_.cbegin(); }
    inline auto cend() { return algorithms_.cend(); }

private:
    saca_list() {}

    std::vector<saca const*> algorithms_;
}; // class saca_list

/// Abstract base class for a SACA in the registry.
class saca {
public:
    saca(const std::string& name, const std::string& description)
        : name_(name), description_(description) {
        saca_list::get().register_saca(this);
    }

    /// Run the SACA on a text in memory.
    ///
    /// It selects a suitable `sa_index` type automatically.
    inline void construct_sa(string_span test_input) const {
        construct_sa_64(test_input);
    }

    /// Run the SACA on the example string `"hello world"`.
    inline void run_example() const { construct_sa_32("hello world"_s); }

    /// Get the name of the SACA.
    std::string const& name() const { return name_; }

    /// Get the description of the SACA
    std::string const& description() const { return description_; }

protected:
    /// Runs the SACA with a 32 bit `sa_index` type.
    virtual void construct_sa_32(string_span test_input) const = 0;
    /*
    TODO: Commented out because of compile errors with the uint4X types.
    /// Runs the SACA with a 40 bit `sa_index` type.
    virtual void construct_sa_40(string_span test_input) const = 0;
    /// Runs the SACA with a 48 bit `sa_index` type.
    virtual void construct_sa_48(string_span test_input) const = 0;
    */
    /// Runs the SACA with a 64 bit `sa_index` type.
    virtual void construct_sa_64(string_span test_input) const = 0;

private:
    std::string name_;
    std::string description_;
}; // class saca

/// A concrete SACA in the registry.
template <typename Algorithm>
class concrete_saca : saca {
public:
    concrete_saca(const std::string& name, const std::string& description)
        : saca(name, description) {}

protected:
    virtual void construct_sa_32(string_span test_input) const override {
        construct_sa<uint32_t>(test_input);
    }
    /*
    TODO: Commented out because of compile errors with the uint4X types.
    virtual void construct_sa_40(string_span test_input) const override {
        construct_sa<util::uint40>(test_input);
    }
    virtual void construct_sa_48(string_span test_input) const override {
        construct_sa<util::uint48>(test_input);
    }
    */
    virtual void construct_sa_64(string_span test_input) const override {
        construct_sa<uint64_t>(test_input);
    }

private:
    /// Delegates to the actual SACA runner.
    template <typename sa_index>
    inline void construct_sa(string_span test_input) const {
        prepare_and_construct_sa<Algorithm, sa_index>(
            text_initializer_from_span(test_input));
    }
}; // class concrete_saca

#define SACA_REGISTER(saca_name, saca_description, saca_impl)                  \
    static const auto _saca_algo_##saca_impl##_register =                      \
        ::sacabench::util::concrete_saca<saca_impl>(saca_name,                 \
                                                    saca_description);

} // namespace sacabench::util
/******************************************************************************/
