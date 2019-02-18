/*******************************************************************************
 * Copyright (C) 2018 Florian Kurpicz <florian.kurpicz@tu-dortmund.de>
 * Copyright (C) 2018 Marvin Löbel <loebel.marvin@gmail.com>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#pragma once

#include <memory>
#include <string>
#include <vector>

#include <tudocomp_stat/StatPhase.hpp>

#include "alphabet.hpp"
#include "bits.hpp"
#include "container.hpp"
#include "macros.hpp"
#include "read_text.hpp"
#include "sa_check.hpp"
#include "span.hpp"
#include "string.hpp"
#include "uint_types.hpp"

namespace sacabench::util {

/// Abstract base class for a suffix array with undetermined `sa_index` type.
class abstract_sa {
public:
    virtual ~abstract_sa() = default;

    /// Run the sa checker, and return its result.
    virtual sa_check_result check(string_span text, bool fast) const = 0;

    /// Write the SA to the `ostream` as a JSON array.
    virtual void write_json(std::ostream& out) const = 0;

    /// Write the SA to the `ostream` as a binary array.
    virtual void write_binary(std::ostream& out, uint8_t bits = 0) const = 0;
};

/// A wrapper around a suffix array container with extra sentinel values.
///
/// It allows a uniform view of the SA, with entries for extra sentinel
/// characters removed.
template <typename sa_index>
class uniform_sa : public abstract_sa {
    size_t m_extra_sentinels;
    container<sa_index> m_sa;

public:
    inline uniform_sa() : m_extra_sentinels(0) {}

    inline uniform_sa(size_t extra_sentinels, container<sa_index>&& sa)
        : m_extra_sentinels(extra_sentinels), m_sa(std::move(sa)) {}

    /// Return number of sentinel characters
    inline size_t extra_sentinels() const { return m_extra_sentinels; }

    /// Return uniform suffix array without sentinel positions.
    inline span<sa_index const> sa_without_sentinels() const {
        return m_sa.slice(extra_sentinels());
    }

    /// Return original suffix array, with potential sentinel positions.
    inline span<sa_index const> sa_with_sentinels() const { return m_sa; }

    inline virtual sa_check_result check(string_span text, bool fast) const override {
        return sa_check_dispatch<sa_index const>(sa_without_sentinels(), text, fast);
    }

    inline virtual void write_json(std::ostream& out) const {
        auto sa = sa_without_sentinels();
        out << "[";
        if (sa.size() > 0) {
            out << sa[0];
        }
        for (size_t i = 1; i < sa.size(); i++) {
            out << ", " << sa[i];
        }
        out << "]";
    }

    inline virtual void write_binary(std::ostream& out, uint8_t bits) const {
        auto sa = sa_without_sentinels();
        if (bits == 0) {
            if (sa.size() > 0) {
                bits = ceil_log2(sa.size() - 1);
            }
            out.put(bits);
        }
        // TODO: Allow true by-bit output
        if (bits % 8 != 0) {
            do {
                bits++;
            } while (bits % 8 != 0);
            std::cerr
                << "INFO: Rounding SA bit size up to next power-of-two size "
                << int(bits) << std::endl;
        }
        std::cerr << "INFO: Writing SA elements with bit size " << int(bits)
                  << std::endl;

        for (size_t i = 0; i < sa.size(); i++) {
            // TODO: This only works for power-of-two sizes
            uint64_t v = sa[i];
            uint8_t b = bits;
            while (b) {
                uint8_t out_byte = v;
                out.put(out_byte);
                v >>= 8;
                b -= 8;
            }
        }
    }
};

/// A type that represents a input text before any allocation.
struct text_initializer {
    /// Getter for the prefix size
    virtual size_t prefix_size() const = 0;
    // Getter for the original text size
    virtual size_t original_text_size() const = 0;
    /// Size of the allocated text. In bytes, without any sentinel values.
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
    size_t prefix;

public:
    inline text_initializer_from_span(string_span text, size_t prefix_length = -1) : m_text(text),
        prefix(prefix_length) {}

    virtual inline size_t prefix_size() const { return prefix; }

    virtual inline size_t original_text_size() const { return m_text.size(); }

    virtual inline size_t text_size() const override { return std::min(
        prefix_size(), original_text_size()); }

    virtual inline void initializer(span<character> s) const override {
        //DCHECK_EQ(s.size(), m_text.size());
        for (size_t i = 0; i < s.size(); i++) {
            s[i] = m_text[i];
        }
    }
};

/// A `text_initializer` that initializes with the content of a file.
class text_initializer_from_file : public text_initializer {
    read_text_context m_ctx;
    size_t prefix;

public:
    inline text_initializer_from_file(std::string const& file_path, size_t prefix_length = -1)
        : m_ctx(file_path), prefix(prefix_length) {}

    virtual inline size_t prefix_size() const { return prefix; }

    virtual inline size_t original_text_size() const { return m_ctx.size; };

    virtual inline size_t text_size() const override { return std::min(
        prefix_size(), original_text_size()); }

    virtual inline void initializer(span<character> s) const override {
        m_ctx.read_text(s);
    }
};

/// Prepares SA and Text containers, and calls the given SACA Algorithm with
/// them.
///
/// \param text_init Initializer for the text.
template <typename Algorithm, typename sa_index>
uniform_sa<sa_index>
prepare_and_construct_sa(text_initializer const& text_init) {
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
        root.log("sa_index_bit_size",
                 ceil_log2(std::numeric_limits<sa_index>::max()));
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

    using abstract_sa_ptr = std::unique_ptr<abstract_sa>;

    /// Run the SACA on some text.
    ///
    /// It selects a suitable `sa_index` type automatically.
    inline abstract_sa_ptr
    construct_sa(text_initializer const& text,
                 size_t minimum_sa_bit_width) const {
        if (minimum_sa_bit_width <= 32) {
            return construct_sa_32(text);
        } else if (minimum_sa_bit_width <= 40) {
            return construct_sa_40(text);
        } else if (minimum_sa_bit_width <= 48) {
            return construct_sa_48(text);
        } else {
            return construct_sa_64(text);
        }
    }

    /// Run the SACA on the example string `"hello world"`.
    inline abstract_sa_ptr run_example() const {
        return construct_sa_32(text_initializer_from_span("hello world"_s));
    }

    /// Get the name of the SACA.
    std::string const& name() const { return name_; }

    /// Get the description of the SACA
    std::string const& description() const { return description_; }

protected:
    /// Runs the SACA with a 32 bit `sa_index` type.
    virtual abstract_sa_ptr
    construct_sa_32(text_initializer const& text) const = 0;

    /// Runs the SACA with a 40 bit `sa_index` type.
    virtual abstract_sa_ptr
    construct_sa_40(text_initializer const& text) const = 0;

    /// Runs the SACA with a 48 bit `sa_index` type.
    virtual abstract_sa_ptr
    construct_sa_48(text_initializer const& text) const = 0;

    /// Runs the SACA with a 64 bit `sa_index` type.
    virtual abstract_sa_ptr
    construct_sa_64(text_initializer const& text) const = 0;

private:
    std::string name_;
    std::string description_;
}; // class saca

/// A concrete SACA in the registry.
template <typename Algorithm>
class concrete_saca : saca {
public:
    concrete_saca() : saca(Algorithm::NAME, Algorithm::DESCRIPTION) {}

protected:
    virtual abstract_sa_ptr
    construct_sa_32(text_initializer const& text) const override {
        return construct_sa_helper<uint32_t>(text);
    }

    virtual abstract_sa_ptr
    construct_sa_40(text_initializer const& text) const override {
        return construct_sa_helper<util::uint40>(text);
    }
    virtual abstract_sa_ptr
    construct_sa_48(text_initializer const& text) const override {
        return construct_sa_helper<util::uint48>(text);
    }

    virtual abstract_sa_ptr
    construct_sa_64(text_initializer const& text) const override {
        return construct_sa_helper<uint64_t>(text);
    }

private:
    template <typename sa_index>
    abstract_sa_ptr construct_sa_helper(text_initializer const& text) const {
        return std::make_unique<uniform_sa<sa_index>>(
            prepare_and_construct_sa<Algorithm, sa_index>(text));
    }
}; // class concrete_saca

#define SACA_REGISTER(...)                                                     \
    static const auto GENSYM(_saca_algo_register_) =                           \
        ::sacabench::util::concrete_saca<__VA_ARGS__>();

} // namespace sacabench::util
/******************************************************************************/
