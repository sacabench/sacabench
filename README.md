# SACABench
## What is it?
TBD

## How to get it?
First clone this repository, then build all executables.
```sh
git clone git@flint-v3.cs.tu-dortmund.de:sacabench/sacabench.git
cd sacabench
mkdir build
cd build
cmake ..

# Run tests:
make check

# Run examples:
make
```

#TODO: Short demonstration of the SACABench commandline tool.

## Including your own SACA
If you want to use our benchmark tool for your own SACA implementation, you
first need to implement our SACA interface within the sacabench/saca directory.

```cpp
// Either class or struct work
class your_saca {
    // The amount of sentinels (null byte) your SACA needs at the end of the input text.
    static constexpr EXTRA_SENTINELS = 0;
    // Your SACA's name
    static constexpr NAME = "Your_saca";
    // The Description for your SACA
    static constexpr DESCRIPTION = "Your SACA's description.";

    /**
        Member function to call your SACA from within our framework.

        @param text The input text (with sentinels)
        @param alphabet The input's alphabet (without sentinel)
        @param out_sa The span containing your computed sa.
    */
    template <typename sa_index>
    static void construct_sa(util::string_span text,
                             util::alphabet const& alphabet,
                             util::span<sa_index> out_sa) {
        /*
            Your SACAs implementation will be called from here.
        */
    }
};
```

After you implement this interface, you still need to register your saca in the
register.cmake file (within the same directory):

```cmake
SACA_REGISTER("saca/path_to_your_saca_header/saca.hpp"
    sacabench::your_namespace)
```
It is recommended to begin your namespace with ```sacabench```.

If you wish to implement a GPGPU-SACA, you are advised to add your SACA within
the ```IF(CUDA)``` environment.
