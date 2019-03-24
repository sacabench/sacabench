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

## Using the cli
```sacabench list``` lists all available suffix array construction algorithms contained in this project.

Usage: ```sacabench/sacabench list [OPTIONS]```

Available options are:

  -h,--help: Print this help message and exit<br>
  -n,--no-description: Don't show a description for each algorithm.<br>
  -j,--json: Output list as an json array

---

```sacabench demo``` tests the correct functionality on the current system.

Usage: ```sacabench/sacabench demo [OPTIONS]```

Available options are:

  -h,--help: Print this help message and exit<br>
  
---

```sacabench construct``` executes a single SACA.

Usage: ```sacabench/sacabench construct [OPTIONS] algorithm input```

Positionals:

  algorithm TEXT REQUIRED: Which algorithm to run.<br>
  input TEXT REQUIRED: Path to input file, or - for STDIN.

Available options are:
  -h,--help:                   Print this help message and exit<br>
  --config TEXT:               Read an config file for CLI args<br>
  -c,--check:                  Check the constructed SA.<br>
  -q,--fastcheck:              Check the constructed SA with a faster, parallel algorithm.<br>
  -b,--benchmark TEXT:         Record benchmark and output as JSON. Takes path to output file, or - for STDOUT<br>
  -J,--json TEXT:              Output SA as JSON array. Takes path to output file, or - for STDOUT.<br>
  -B,--binary TEXT:            Output SA as binary array of unsigned integers, with a 1 Byte header describing the number of bits used for each integer. Takes path to output file, or - for STDOUT.<br>
  -F,--fixed UINT Needs --binary: Elide the header, and output a fixed number of bits per SA entry<br>
  -p,--prefix TEXT:            Calculate SA of prefix of size TEXT.<br>
  -f,--force:                  Overwrite existing files instead of raising an error.<br>
  -m,--minimum_sa_bits UINT=32: The lower bound of bits to use per SA entry during construction<br>
  -r,--repetitions UINT=1: The value indicates the number of times the SACA(s) will run. A larger number will possibly yield more accurate results<br>
  -z,--rplot Needs --benchmark: Plots measurements with R.<br>
  --latexplot Needs --benchmark: Plots measurements with LaTex and SqlPlotTools.<br>
  -s,--sysinfo Needs --benchmark: Add system information to benchmark output.
  
---

```sacabench batch``` allows to execute and compare multiple SACAs.

Usage: ```sacabench/sacabench batch [OPTIONS] input```
  
---

```sacabench plot``` generates plots from a previously mesurement.

Usage: ```sacabench/sacabench plot [OPTIONS] benchmark_file```

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
