TEX = latexmk
BENCHMARK = plots.json
CONFIG = plotconfig.json

.PHONY: all clean run

all: main

sqlplottools:
	if [ ! -d ../sqlplot/sqlplottools ] ; then git clone https://github.com/bingmann/sqlplot-tools.git ../sqlplot/sqlplottools; fi
	mkdir -p ../sqlplot/sqlplottools/build
	cd ../sqlplot/sqlplottools/build && cmake .. && $(MAKE)

json:
	mkdir -p result
	python ../sqlplot/json_to_result_converter.py $(CONFIG) $(BENCHMARK) result scaling

sql: json sqlplottools
	for file in $(shell ls *.tex); do \
		../sqlplot/sqlplottools/build/src/sqlplot-tools $$file; \
	done;

main: sql
	for file in $(shell ls *.tex); do \
		$(TEX) -pdf $$file; \
	done;

run: main
	for file in $(shell ls *.tex); do \
		$(TEX) -pdf -pv $$file; \
	done;

clean:
	$(TEX) -C
	rm *.tex
