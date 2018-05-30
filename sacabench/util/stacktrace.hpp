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

#include <string>

#ifdef __linux__

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

// This function produces a stack backtrace with demangled function & method
// names.
inline std::string Backtrace(int skip = 1) {
    bool nice = true;
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

            if (marker_found) {
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
inline std::string Backtrace(int skip = 1) {
    return "<no backtrace on this platform>";
}
} // namespace sacabench::backtrace

#endif
