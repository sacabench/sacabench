#pragma once

// Uncomment to activate logging
// #define SB_DSLOG

// Run logging in debug mode
#ifdef DEBUG
#define SB_DSLOG
#endif

// Include headesr
#ifdef SB_DSLOG
#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>
#endif

namespace sacabench::deep_shallow {

#ifdef SB_DSLOG

// #############################################################################

class logger {
private:
    std::stringstream ss;

    inline logger() : ss() {
        std::cout << "Using Deep-Shallow Logger" << std::endl;
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

    inline void flush() {
        std::cout << "Writing Deep-Shallow logfile." << std::endl;
        std::ofstream myfile;
        myfile.open("most_recent.log", std::fstream::out | std::fstream::trunc);
        myfile << ss.rdbuf();
        myfile.close();
    }
};

template <typename Fn>
inline double duration(Fn fn) {
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
    inline logger& operator<<(const T& c) {
        return *this;
    }

    inline void flush() {}
};

template <typename Fn>
inline double duration(Fn fn) { fn(); return 0; }

#endif
} // namespace sacabench::deep_shallow
