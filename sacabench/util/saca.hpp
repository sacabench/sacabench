/*******************************************************************************
 * Copyright (C) 2018 Florian Kurpicz <florian.kurpicz@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#pragma once

#include <string>
#include <vector>

class saca;

class saca_list {
    public:
        saca_list(const saca_list&) = delete;
        saca_list(saca_list&&) = delete;
        void operator =(const saca_list&) = delete;
        void operator =(saca_list&&) = delete;

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
        saca_list() { }

        std::vector<saca const*> algorithms_;
}; // class saca_list

class saca {
    public:
        saca(const std::string& name, const std::string& description)
            : name_(name), description_(description) {
                saca_list::get().register_saca(this);
            }

        virtual void run_example() const = 0;

        std::string name() const { return name_; }
        std::string description() const { return description_; }

    private:
        std::string name_;
        std::string description_;
}; // class saca

template <typename Algorithm>
class concrete_saca : saca {
    public:
        concrete_saca(const std::string& name, const std::string& description)
            : saca(name, description) { }

        void run_example() const override {
            Algorithm::run_example();
        }
}; // class concrete_saca

#define SACA_REGISTER(saca_name, saca_description, saca_impl) \
    static const auto _saca_algo_ ## saca_impl ## _register     \
    = concrete_saca<saca_impl>(saca_name, saca_description);

/******************************************************************************/
