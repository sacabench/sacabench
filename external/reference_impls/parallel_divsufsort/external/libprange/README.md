Description
==========

Parallel-Range-Lite is a parallel suffix array construction algorithm for integer alphabets.
It is a lightweight implementation based on the range algorithm from the Problem Based Benchmark Suite [PBBS](http://www.cs.cmu.edu/~pbbs/).
A rough description of the algorithm can be found in the following work.
> Julian Labeit, Julian Shun, and Guy E. Blelloch. Parallel Lightweight Wavelet Tree, Suffix Array and FM-Index Construction. DCC 2015.

Installation
==========
The following steps have been tested on Ubuntu 14.04 with gcc 5.3.0 and cmake 2.8.12.
```shell
git clone https://github.com/jlabeit/parallel-range-lite.git
cd parallel-range-lite
mkdir build
cd build
cmake ..
make
make install
```
Note that in the default version the cilkplus implementation by gcc is used for parallelization.
To change this setting edit parallelization settings in the CMakeLists.txt file.

Getting Started
=========
An example application can be found in demo/main.cpp.
The library provides two basic functions to build the suffix array over a text.

```c++
// 32 bit version.
void parallelrangelite(int32_t* ISA, int32_t* SA, int32_t n);
// 64 bit version.
void parallelrangelite(int64_t* ISA, int64_t* SA, int64_t n);
```

To use the library include the header `parallel-range.h`, link against the library `libprange`.

