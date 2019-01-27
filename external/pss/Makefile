COMPILE = icc -O2 -std=c++11 -I..

all: tbb-lowlevel.x tbb-highlevel.x cilkplus.x openmp.x gomp.x

run: all
	./tbb-highlevel.x
	./tbb-lowlevel.x
	./cilkplus.x
	./gomp.x
	./openmp.x

tbb-lowlevel.x: ../tbb-lowlevel/parallel_stable_sort.h
	$(COMPILE) -o $@ -DUSE_TBB_LOWLEVEL ../test.cpp -ltbb

tbb-highlevel.x: ../tbb-highlevel/parallel_stable_sort.h
	$(COMPILE) -o $@ -DUSE_TBB_HIGHLEVEL ../test.cpp -ltbb

cilkplus.x: ../cilkplus/parallel_stable_sort.h
	$(COMPILE) -o $@ -DUSE_CILKPLUS ../test.cpp -ltbb

openmp.x: ../openmp/parallel_stable_sort.h
	$(COMPILE) -o $@ -DUSE_OPENMP -openmp ../test.cpp 

gomp.x: ../openmp/parallel_stable_sort.h
	g++ -O0 -g -std=c++11 -I.. -o $@  -DUSE_OPENMP -fopenmp ../test.cpp 

*.x: ../test.cpp

clean:
	rm -f *.x
