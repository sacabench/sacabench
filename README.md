# SACABench
## What is it?
TBD

## Dependencies

On a fresh Ubuntu 18.04 LTS installation, the following dependencies need to be installed:

```sh
# REQUIRED: git, the gcc compiler, the cmake build system
sudo apt install git build-essential cmake

# REQUIRED for some dependencies: the autoconf build system
sudo apt install autoconf libtool

# RECOMMENDED for faster rebuilds during development: ccache
sudo apt install ccache

# OPTIONAL for some support scripts: python 3
sudo apt install python3
```

## How to build

```sh
# Clone this repo
git clone https://github.com/sacabench/sacabench.git

# Create build directory
cd sacabench
mkdir build
cd build

# Configure the build system for a optimized "Release" build
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build the sacabench tool
make

# Build & Run unit tests:
make check
```

There are also a few cmake options and build system targets that are useful for active development:

- `cmake .. -DCMAKE_BUILD_TYPE=Release` configure for optimized release builds.
- `cmake .. -DCMAKE_BUILD_TYPE=Debug` configure for unoptimized debug builds with assertions.
- `make build_check` only build the unit tests, don't run them.
- `cmake .. -DSACA_WERROR=ON` turn gcc warnings into hard errors.
- `cmake .. -DSACA_RELEASE_DEBUGINFO=ON` activate debug information (`-g`) in release builds.
- `make datasets` download a few 200MiB datasets from different sources.

## Using the cli

After you've build the `sacabench` tool via `make`, it can be found at `<build directory>/sacabench/sacabench`. It offers the following CLI options:

- `sacabench list` lists all available suffix array construction algorithms contained in this project.
- `sacabench demo` tests the correct functionality on the current system.
- `sacabench construct` executes a single SACA.
- `sacabench batch` allows to execute and compare multiple SACAs.
- `sacabench plot` generates plots from a previously measurement.

For more information about the commands and available options, use `-h` or `--help`, e.g. `sacabench list --help`

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
