/*
 * Copyright (c) 2009-2017, Farooq Mela
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.

 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
 FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#pragma once

#include <string>

#define MYTRACE() std::cerr << "TRACE @" << __FILE__ << ":" << __LINE__ << "\n";

struct lspan {
    static constexpr size_t npos = ~0ull;

    char const* ptr;
    size_t size;

    inline lspan(char const* _ptr, size_t _size) : ptr(_ptr), size(_size) {}
    inline lspan(char const* _ptr) : lspan(_ptr, strlen(_ptr)) {}
    inline lspan() : lspan("") {}

    inline auto begin() const { return ptr; }
    inline auto end() const { return ptr + size; }
    inline lspan slice(size_t from = 0, size_t to = npos) {
        if (to == npos) {
            to = size;
        }
        if (from > size) {
            MYTRACE();
            abort();
        }
        if (to > size) {
            MYTRACE();
            abort();
        }
        if (from > to) {
            MYTRACE();
            abort();
        }
        return lspan{ptr + from, to - from};
    }
    inline bool starts_with(lspan other) {
        if (other.size > size)
            return false;
        for (size_t i = 0; i < other.size; i++) {
            if (ptr[i] != other.ptr[i])
                return false;
        }
        return true;
    }
};

struct result {
    lspan return_ty;
    lspan function_name;
    lspan function_namespace;
    lspan function_namespace_and_name;
};

struct noop {
    inline bool operator()(result) { return true; }
};


#if defined(__linux__)

#include <iomanip>
#include <iostream>

#include <cxxabi.h>   // for __cxa_demangle
#include <dlfcn.h>    // for dladdr
#include <execinfo.h> // for backtrace

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>

namespace sacabench::backtrace {

inline std::ostream& operator<<(std::ostream& out, lspan span) {
    return out.write(span.ptr, span.size);
}

inline result parse_symbol(lspan name) {
    result r;

    int nesting = 0;

    size_t i;
    for (i = 0; i < name.size; i++) {
        auto n = name.slice(i);

        auto kw = [&](lspan m) {
            if (n.starts_with(m)) {
                i += (m.size - 1);
                return true;
            }
            return false;
        };

        if (kw("operator<<=") || kw("operator<=>") || kw("operator<<") ||
            kw("operator<=") || kw("operator<") || kw("operator>>=") ||
            kw("operator>>") || kw("operator>=") || kw("operator>") ||
            kw("operator->*") || kw("operator->")) {
            continue;
        } else if (n.starts_with("<") || n.starts_with("(")) {
            nesting++;
        } else if (n.starts_with(">") || n.starts_with(")")) {
            nesting--;
        } else if (n.starts_with(" ") && nesting == 0) {
            if (n.starts_with(" const")) {
                continue;
            }
            break;
        }
    }

    if (i < name.size) {
        r.return_ty = name.slice(0, i);
        r.function_name = name.slice(i + 1);
    } else {
        r.return_ty = "void";
        r.function_name = name;
    }

    size_t j;
    size_t k = 0;
    nesting = 0;
    for (j = 0; j < r.function_name.size; j++) {
        auto n = r.function_name.slice(j);

        if (n.starts_with("<") || n.starts_with("(")) {
            nesting++;
        } else if (n.starts_with(">") || n.starts_with(")")) {
            nesting--;
        } else if (n.starts_with("::") && nesting == 0) {
            k = j;
        }
    }

    r.function_namespace_and_name = r.function_name;
    r.function_namespace = r.function_name.slice(0, k);
    r.function_name = r.function_name.slice(k);
    if (r.function_name.starts_with("::")) {
        r.function_name = r.function_name.slice(2);
    }

    return r;
}

// This function produces a stack backtrace with demangled function & method
// names.
template <typename callback = noop>
inline std::string Backtrace(int skip = 1, bool nice = true,
                             callback cb = callback()) {
    int offset = 0;
    if (nice) {
        offset = skip;
    }

    void* callstack[128];
    const int nMaxFrames = sizeof(callstack) / sizeof(callstack[0]);
    int nFrames = ::backtrace(callstack, nMaxFrames);
    char** symbols = ::backtrace_symbols(callstack, nFrames);

    std::ostringstream trace_buf;
    trace_buf << "Stack Trace:\n";
    for (int i = skip; i < nFrames; i++) {
        // printf("%s\n", symbols[i]);

        trace_buf << std::setw(3) << (i - offset) << "    ";
        if (!nice) {
            trace_buf << callstack[i] << " ";
        }

        Dl_info info;
        if (dladdr(callstack[i], &info) && info.dli_sname) {
            char* demangled = nullptr;
            int status = -1;
            if (info.dli_sname[0] == '_') {
                demangled =
                    abi::__cxa_demangle(info.dli_sname, NULL, 0, &status);
            }
            char const* name =
                (status == 0
                     ? demangled
                     : info.dli_sname == 0 ? symbols[i] : info.dli_sname);

            trace_buf << name;

            auto p = parse_symbol(name);
            cb(p);

            /*
            trace_buf << "\n";
            trace_buf << "namespace: " << p.function_namespace << "\n";
            trace_buf << "function:  " << p.function_name << "\n";
            trace_buf << "retty:     " << p.return_ty << "\n";
            */

            if (!nice) {
                trace_buf << " + ";
                trace_buf << ((char*)callstack[i] - (char*)info.dli_saddr);
            }

            const auto marker = "_Test::TestBody()";
            bool marker_found = false;
            if (std::strlen(name) >= std::strlen(marker)) {
                name = name + std::strlen(name) - std::strlen(marker);
                marker_found =
                    std::strncmp(name, marker, std::strlen(marker)) == 0;
            }

            std::free(demangled);

            if (nice && marker_found) {
                trace_buf << "\n";
                break;
            }
        } else {
            trace_buf << symbols[i];
        }
        trace_buf << "\n";
    }
    free(symbols);
    if (nFrames == nMaxFrames)
        trace_buf << "[truncated]\n";
    trace_buf << "\nFor more details, use a debugger like gdb.\n\n";
    return trace_buf.str();
}
} // namespace sacabench::backtrace

#else

namespace sacabench::backtrace {

template <typename callback = noop>
inline std::string Backtrace(int = 1, bool = true,
                             callback = callback()) {
    return "<no backtrace on this platform>";
}

} // namespace sacabench::backtrace

#endif
