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
#include "bits.hpp"
#include "container.hpp"
#include "read_text.hpp"
#include "span.hpp"
#include "string.hpp"
#include "uint_types.hpp"

namespace sacabench::util {

/// A wrapper around a suffix array container with extra sentinel values.
///
/// It allows a uniform view of the SA, with entries for extra sentinel
/// characters removed.
template <typename sa_index>
class uniform_sa {
    size_t m_extra_sentinels;
    container<sa_index> m_sa;

public:
    inline uniform_sa() : m_extra_sentinels(0) {}

    inline uniform_sa(size_t extra_sentinels, container<sa_index>&& sa)
        : m_extra_sentinels(extra_sentinels), m_sa(std::move(sa)) {}

    /// Return number of sentinel characters
    inline size_t extra_sentinels() { return m_extra_sentinels; }

    /// Return uniform suffix array without sentinel positions.
    inline span<sa_index> sa_without_sentinels() {
        return m_sa.slice(extra_sentinels());
    }

    /// Return original suffix array, with potential sentinel positions.
    inline span<sa_index> sa_with_sentinels() { return m_sa; }
};

/// A type that represents a input text before any allocation.
struct text_initializer {
    /// Size of the text. In bytes, without any sentinel values.
    virtual size_t text_size() const = 0;

    /// Initializer function, that writes the text to the passed
    /// span of size `text_size`.
    virtual void initializer(span<character>) const = 0;

    /// Declare a virtual destructor, because you have to do that (TM)
    virtual ~text_initializer() = default;
};

/// A `text_initializer` that initializes with a `string_span`.
class text_initializer_from_span : public text_initializer {
    string_span m_text;

public:
    inline text_initializer_from_span(string_span text) : m_text(text) {}

    virtual inline size_t text_size() const override { return m_text.size(); }

    virtual inline void initializer(span<character> s) const override {
        DCHECK_EQ(s.size(), m_text.size());
        for (size_t i = 0; i < s.size(); i++) {
            s[i] = m_text[i];
        }
    }
};

/// A `text_initializer` that initializes with the content of a file.
class text_initializer_from_file : public text_initializer {
    read_text_context m_ctx;

public:
    inline text_initializer_from_file(std::string const& file_path)
        : m_ctx(file_path) {}

    virtual inline size_t text_size() const override { return m_ctx.size; }

    virtual inline void initializer(span<character> s) const override {
        m_ctx.read_text(s);
    }
};

/// Prepares SA and Text containers, and calls the given SACA Algorithm with
/// them.
///
/// \param text_init Initializer for the text.
template <typename Algorithm, typename sa_index, typename text_init_function>
uniform_sa<sa_index> prepare_and_construct_sa(text_initializer const& text_init,
                                                bool WIP_print_stats = false) {
    tdc::StatPhase root("SACA");
    uniform_sa<sa_index> ret;
    {
        size_t extra_sentinels = Algorithm::EXTRA_SENTINELS;
        size_t text_size = text_init.text_size();

        container<sa_index> output;
        string text_with_sentinels;
        alphabet alph;

        {
            tdc::StatPhase init_phase("Allocate SA and Text container");
            output = make_container<sa_index>(text_size + extra_sentinels);
            text_with_sentinels = string(text_size + extra_sentinels);
        }

        // Create a slice to the part of the Text container
        // that contains the original text without sentinels.
        auto text = text_with_sentinels.slice(0, text_size);

        {
            tdc::StatPhase init_phase("Initialize Text");
            text_init.initializer(text);
        }

        IF_DEBUG({
            // Check that we got valid input.
            for (size_t i = 0; i < text.size(); i++) {
                DCHECK_MSG(text[i] != 0,
                           "Input byte " << i
                                         << " has value 0, which is reserved "
                                            "for the terminating sentinel!");
            }
        })

        {
            tdc::StatPhase init_phase("Apply effective Alphabet");
            alph = apply_effective_alphabet(text);
        }

        {
            tdc::StatPhase init_phase("Algorithm");
            span<sa_index> out_sa = output;
            string_span readonly_text_with_sentinels = text_with_sentinels;
            alphabet const& readonly_alphabet = alph;
            Algorithm::construct_sa(readonly_text_with_sentinels,
                                    readonly_alphabet, out_sa);
        }

        ret = uniform_sa<sa_index>{extra_sentinels, std::move(output)};

        root.log("text_size", text.size());
        root.log("extra_sentinels", extra_sentinels);
        root.log("sa_index_size",
                 ceil_log2(std::numeric_limits<sa_index>::max()));
    }

    if (WIP_print_stats) {
        auto j = root.to_json();
        std::cout << j.dump(4) << std::endl;
    }

    return ret;
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
    inline static saca_list& get() {
        static saca_list list;
        return list;
    }

    /// Register a SACA. This gets called automatically by the
    /// `SACA_REGISTER` macro.
    inline void register_saca(saca const* algorithm) {
        algorithms_.push_back(algorithm);
    }

    // Iterators

    inline auto begin() { return algorithms_.begin(); }
    inline auto end() { return algorithms_.end(); }
    inline auto cbegin() { return algorithms_.cbegin(); }
    inline auto cend() { return algorithms_.cend(); }

private:
    inline saca_list() = default;

    std::vector<saca const*> algorithms_;
}; // class saca_list

/// Abstract base class for a SACA in the registry.
class saca {
public:
    saca(const std::string& name, const std::string& description)
        : name_(name), description_(description) {
        saca_list::get().register_saca(this);
    }

    /// Run the SACA on some text.
    ///
    /// It selects a suitable `sa_index` type automatically.
    inline void construct_sa(text_initializer const& text) const {
        // TODO: Select suitable `sa_index` type automatically,
        // or offer an API for selecting it.

        construct_sa_64(text);
    }

    /// Run the SACA on the example string `"hello world"`.
    inline void run_example() const {
        construct_sa_32(text_initializer_from_span("hello world"_s));
    }

    /// Get the name of the SACA.
    std::string const& name() const { return name_; }

    /// Get the description of the SACA
    std::string const& description() const { return description_; }

protected:
    /// Runs the SACA with a 32 bit `sa_index` type.
    virtual void construct_sa_32(text_initializer const& text) const = 0;
    /*
    TODO: Commented out because of compile errors with the uint4X types.
    /// Runs the SACA with a 40 bit `sa_index` type.
    virtual void construct_sa_40(text_initializer const& text) const = 0;
    /// Runs the SACA with a 48 bit `sa_index` type.
    virtual void construct_sa_48(text_initializer const& text) const = 0;
    */
    /// Runs the SACA with a 64 bit `sa_index` type.
    virtual void construct_sa_64(text_initializer const& text) const = 0;

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
    virtual void construct_sa_32(text_initializer const& text) const override {
        prepare_and_construct_sa<Algorithm, uint32_t>(text);
    }
    /*
    TODO: Commented out because of compile errors with the uint4X types.
    virtual void construct_sa_40(text_initializer const& text) const override {
        prepare_and_construct_sa<Algorithm, util::uint40>(text);
    }
    virtual void construct_sa_48(text_initializer const& text) const override {
        prepare_and_construct_sa<Algorithm, util::uint48>(text);
    }
    */
    virtual void construct_sa_64(text_initializer const& text) const override {
        prepare_and_construct_sa<Algorithm, uint64_t>(text);
    }
}; // class concrete_saca

#define SACA_REGISTER(saca_name, saca_description, saca_impl)                  \
    static const auto _saca_algo_##saca_impl##_register =                      \
        ::sacabench::util::concrete_saca<saca_impl>(saca_name,                 \
                                                    saca_description);

} // namespace sacabench::util
/******************************************************************************/
