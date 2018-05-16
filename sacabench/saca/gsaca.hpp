/*******************************************************************************
 * Copyright (C) 2018 David Piper <david.piper@tu-dortmund.de
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#pragma once

namespace sacabench::gsaca {

    class gsaca {
    public:
        template<typename sa_index>
        static void construct_sa(util::string_span text,
                                 size_t /*alphabet_size*/,
                                 util::span <sa_index> out_sa) {

        }
    }; // class gsaca
} // namespace sacabench::gsaca
