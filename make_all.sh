rm -rf build_debug build_release

mkdir build_debug
mkdir build_release

cd build_debug
cmake -DCMAKE_BUILD_TYPE=Debug ..

cd ../build_release
cmake -DCMAKE_BUILD_TYPE=Release ..

cd ../build_debug
make -j${OMP_NUM_THREADS}
make check

cd ../build_release
make -j${OMP_NUM_THREADS}
make check