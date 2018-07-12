/*******************************************************************************
 * Copyright (C) 2018 David Piper <david.piper@tu-dortmund.de
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#pragma once

template<typename sa_index>
class GSIZE {
private:
    sacabench::util::container<sa_index> private_GSIZE;
    sa_index group_start_marker = 0;

public:
    inline void setup(size_t number_of_chars) {
        private_GSIZE = sacabench::util::make_container<sa_index>(number_of_chars);
    }

    inline sa_index get_value_at_index(size_t index) {
        return private_GSIZE[index];
    }

    inline void set_value_at_index(size_t index, sa_index value) {
        private_GSIZE[index] = value;
    }

    inline bool is_marked_as_group_start(size_t index) {
        return private_GSIZE[index] == group_start_marker;
    }

    inline void set_group_start_marker_at_index(size_t index) {
        private_GSIZE[index] = group_start_marker;
    }
};

template<typename sa_index>
class GSIZE_BOOL {
private:
    sacabench::util::container<bool> private_GSIZE_BOOL;
    sa_index group_start_marker = false;

    inline sa_index calculate_value(size_t index) {
        sa_index size = 1;
        for (sa_index loop_index = 1; is_member_of_group(index + size); loop_index++) {
            auto old_size = size;
            size++;
            if (is_marked_as_end_of_group(index + old_size)) {
                break;
            }
        }
        return size;
    }

    inline bool is_member_of_group(size_t index) {
        return private_GSIZE_BOOL[2 * index] == false;
    }

    inline bool is_marked_as_end_of_group(size_t index) {
        return private_GSIZE_BOOL[2 * index + 1] == true;
    }

public:
    inline void setup(size_t number_of_chars) {
        private_GSIZE_BOOL = sacabench::util::make_container<bool>(2 * number_of_chars);
    }

    inline sa_index get_value_at_index(size_t index) {
        if (private_GSIZE_BOOL[2 * index] == false) {                   // Check if first bit is 0.
            return 0;
        }

        if (private_GSIZE_BOOL[2 * index + 1] == true) {                // Check if the first group member is marked as end.
            return 1;
        }

        return calculate_value(index);                                  // Calculate the value.
    }

    inline void set_value_at_index(size_t index, sa_index value) {

        if (value == static_cast<sa_index>(0)) {
            private_GSIZE_BOOL[2 * index] = false;                      // Mark group size as 0.
            private_GSIZE_BOOL[2 * index + 1] = false;                  // Mark end of group sequence.
            return;
        }

        private_GSIZE_BOOL[2 * index] = true;                           // Mark begin of group.
        if (value > static_cast<sa_index>(1)) {
            private_GSIZE_BOOL[2 * index + 1] = false;                  // Mark begin of group not as end if there are more members.
        }

        sa_index loop_index = 1;
        while (loop_index < value) {                                    // Iterate over all indices which are to be set to 0.
            private_GSIZE_BOOL[2 * (index + loop_index)] = false;       // Mark member of group as 0.
            private_GSIZE_BOOL[2 * (index + loop_index) + 1] = false;   // Mark member of group as not end of group.
            loop_index += 1;
        }

        private_GSIZE_BOOL[2 * (index + value - 1) + 1] = true;         // Mark end of group sequence.
    }

    inline bool is_marked_as_group_start(size_t index) {
        return private_GSIZE_BOOL[2 * index] == false;
    }

    inline void set_group_start_marker_at_index(size_t index) {
        private_GSIZE_BOOL[2 * index] = false;
    }
};

// BI-SIMULATION
template<typename sa_index>
class GSIZE_Compare {
private:
    GSIZE_BOOL<sa_index> gsize_bool;
    GSIZE<sa_index> gsize;

public:
    bool logging = false;

    inline void setup(size_t number_of_chars) {
        gsize_bool.setup(number_of_chars);
        gsize.setup(number_of_chars);
    }

    inline sa_index get_value_at_index(size_t index) {

        if (logging) { std::cout << "Getting value at index " << index << std::endl; }

        auto gsize_bool_result = gsize_bool.get_value_at_index(index);
        auto gsize_result = gsize.get_value_at_index(index);
        if (gsize_bool_result != gsize_result) {
            std::cout << "ERROR ON GSIZE AT POSITION " << index << std::endl;
        }
        return gsize_result;
    }

    inline void set_value_at_index(size_t index, sa_index value) {

        if (logging) { std::cout << "Setting value at index " << index << " to " << value << std::endl; }

        gsize_bool.set_value_at_index(index, value);
        gsize.set_value_at_index(index, value);
    }

    inline bool is_marked_as_group_start(size_t index) {

        if (logging) { std::cout << "Checking group_start_marker at index " << index << std::endl; }

        auto gsize_bool_result = gsize_bool.is_marked_as_group_start(index);
        auto gsize_result = gsize.is_marked_as_group_start(index);
        if (gsize_bool_result != gsize_result) {
            std::cout << "ERROR ON GSIZE GROUPSTART-MARKER AT POSITION " << index << std::endl;
        }
        return gsize_result;
    }

    inline void set_group_start_marker_at_index(size_t index) {

        if (logging) { std::cout << "Setting group_start_marker at index " << index << std::endl; }

        gsize_bool.set_group_start_marker_at_index(index);
        gsize.set_group_start_marker_at_index(index);
    }
};