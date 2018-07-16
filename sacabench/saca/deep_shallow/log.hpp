#pragma once

// Uncomment to activate logging
// #define SB_DSLOG

// Include headers
#ifdef SB_DSLOG
#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>
#include <util/macros.hpp>
#endif

namespace sacabench::deep_shallow {

#ifdef SB_DSLOG

// #############################################################################

class logger {
private:
    std::stringstream ss;
    bool flushed = false;

    size_t elements_induction_sorted;
    size_t elements_blind_sorted;
    size_t elements_quick_sorted;

    size_t ns_induction_sorted;
    size_t ns_induction_testing;
    size_t ns_blind_sorted;
    size_t ns_quick_sorted;

    inline logger() : ss() {
        std::cout << "Using Deep-Shallow Logger" << std::endl;
    }

    inline ~logger() {
        if (!flushed) {
            ss << "Aborting!!\n";
            flush();
        }
    }

public:
    inline static logger& get() {
        static logger singleton;
        return singleton;
    }

    template <typename T>
    inline logger& operator<<(const T& c) {
        ss << c;
        return *this;
    }

    inline void sorted_elements_blind(const size_t n) {
        elements_blind_sorted += n;
    }

    inline void sorted_elements_quick(const size_t n) {
        elements_quick_sorted += n;
    }

    inline void sorted_elements_induction(const size_t n) {
        elements_induction_sorted += n;
    }

    inline void time_spent_blind(const size_t n) { ns_blind_sorted += n; }

    inline void time_spent_quick(const size_t n) { ns_quick_sorted += n; }

    inline void time_spent_induction_sorting(const size_t n) {
        ns_induction_sorted += n;
    }

    inline void time_spent_induction_testing(const size_t n) {
        ns_induction_testing += n;
    }

    inline void flush() {
        std::cout << "Writing Deep-Shallow logfile." << std::endl;
        std::ofstream myfile;
        myfile.open("most_recent.log", std::fstream::out | std::fstream::trunc);
        myfile << "####################################################\n";
        myfile << "#                    STATISTICS                    #\n";
        myfile << "# Sorted by Blind-Sort: " << elements_blind_sorted << "\n";
        myfile << "# Sorted by Quicksort: " << elements_quick_sorted << "\n";
        myfile << "# Sorted by Induction: " << elements_induction_sorted
               << "\n\n";
        myfile << "# Time spent by Blind Sort: " << ns_blind_sorted << "\n";
        myfile << "# Time spent by Quick Sort: " << ns_quick_sorted << "\n";
        myfile << "# Time spent testing for Induced Sorting: "
               << ns_induction_testing << "\n";
        myfile << "# Time spent by Induced Sorting: " << ns_induction_sorted
               << "\n";
        myfile << "#                                                  #\n";
        myfile << "# Time spent by Blind Sort per suffix: "
               << (ns_blind_sorted * 1.0 / elements_blind_sorted) << "ns\n";
        myfile << "# Time spent by Quick Sort per suffix: "
               << (ns_quick_sorted * 1.0 / elements_quick_sorted) << "ns\n";
        myfile << "# Time spent by Induced Sorting per suffix: "
               << (ns_induction_sorted * 1.0 / elements_induction_sorted)
               << "ns\n";
        myfile << "#                                                  #\n";
        myfile << "####################################################\n\n";
        myfile << ss.rdbuf();
        myfile.close();
        flushed = true;
    }
};

template <typename Fn>
SB_NO_INLINE double duration(Fn fn) {
    const auto start = std::chrono::steady_clock::now();
    fn();
    const auto end = std::chrono::steady_clock::now();
    const auto dur = end - start;
    return std::chrono::duration_cast<std::chrono::nanoseconds>(dur).count();
}

// #############################################################################

#else
class logger {
public:
    inline static logger& get() {
        static logger singleton;
        return singleton;
    }

    template <typename T>
    inline logger& operator<<(const T&) {
        return *this;
    }

    inline void flush() {}

    inline void sorted_elements_blind(const size_t) {}
    inline void sorted_elements_quick(const size_t) {}
    inline void sorted_elements_induction(const size_t) {}
    inline void time_spent_blind(const size_t) {}
    inline void time_spent_quick(const size_t) {}
    inline void time_spent_induction_sorting(const size_t) {}
    inline void time_spent_induction_testing(const size_t) {}
};

template <typename Fn>
inline double duration(Fn fn) {
    fn();
    return 0;
}

#endif
} // namespace sacabench::deep_shallow
