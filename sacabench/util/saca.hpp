/*******************************************************************************
 * Copyright (C) 2018 Florian Kurpicz <florian.kurpicz@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#pragma once

#include <string>
#include <vector>

#include "alphabet.hpp"
#include "container.hpp"
#include "span.hpp"
#include "string.hpp"

namespace sacabench::util {

template <typename Algorithm, typename sa_index_t, typename text_init_function>
container<sa_index_t> prepare_and_construct_sa(size_t text_size,
                                               text_init_function init) {
    auto output = make_container<sa_index_t>(text_size);

    size_t extra_sentinels = Algorithm::EXTRA_SENTINELS;

    auto text_with_sentinels = string(text_size + extra_sentinels);
    auto text = text_with_sentinels.slice(0, text_size);

    init(text);

    /* TODO: Uncommend and write test-for if IF_DEBUG is merged
    IF_DEBUG({
        for (size_t i = 0; i < text.size(); i++) {
            DCHECK_MSG(text[i] != 0, "Input byte " << i << " has value 0, which
    is reserved for the terminating sentinel!")
        }
    })
    */

    auto alph = apply_effective_alphabet(text);
    auto const alph_info = alphabet_info(alph.real_size - 1, true);

    {
        span<sa_index_t> out_sa = output;
        string_span readonly_text_with_sentinels = text_with_sentinels;
        Algorithm::construct_sa(readonly_text_with_sentinels, alph_info,
                                out_sa);
    }

    return output;
}

class saca;

class saca_list {
public:
    saca_list(const saca_list&) = delete;
    saca_list(saca_list&&) = delete;
    void operator=(const saca_list&) = delete;
    void operator=(saca_list&&) = delete;

    static saca_list& get() {
        static saca_list list;
        return list;
    }

    void register_saca(saca const* algorithm) {
        algorithms_.push_back(algorithm);
    }

    inline auto begin() { return algorithms_.begin(); }
    inline auto end() { return algorithms_.end(); }
    inline auto cbegin() { return algorithms_.cbegin(); }
    inline auto cend() { return algorithms_.cend(); }

private:
    saca_list() {}

    std::vector<saca const*> algorithms_;
}; // class saca_list

class saca {
public:
    saca(const std::string& name, const std::string& description)
        : name_(name), description_(description) {
        saca_list::get().register_saca(this);
    }

    virtual void run_example() const = 0;
    virtual void construct_sa(string_span test_input) const = 0;

    std::string const& name() const { return name_; }
    std::string const& description() const { return description_; }

private:
    std::string name_;
    std::string description_;
}; // class saca

template <typename Algorithm>
class concrete_saca : saca {
public:
    concrete_saca(const std::string& name, const std::string& description)
        : saca(name, description) {}

    virtual void construct_sa(string_span test_input) const override {
        prepare_and_construct_sa<Algorithm, size_t>(
            test_input.size(), [&](auto s) {
                for (size_t i = 0; i < s.size(); i++) {
                    s[i] = test_input[i];
                }
            });
    }
    virtual void run_example() const override {
        using sa_index_t = uint32_t;
        string_span test_input = "hello world"_s;

        prepare_and_construct_sa<Algorithm, sa_index_t>(
            test_input.size(), [&](auto s) {
                for (size_t i = 0; i < s.size(); i++) {
                    s[i] = test_input[i];
                }
            });
    }
}; // class concrete_saca

#define SACA_REGISTER(saca_name, saca_description, saca_impl)                  \
    static const auto _saca_algo_##saca_impl##_register =                      \
        ::sacabench::util::concrete_saca<saca_impl>(saca_name,                 \
                                                    saca_description);

} // namespace sacabench::util
/******************************************************************************/
