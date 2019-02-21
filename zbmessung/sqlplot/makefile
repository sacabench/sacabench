TEX = latexmk
BENCHMARK = plots.json

.PHONY: all clean run

all: main

sqlplottools:
	if [ ! -d sqlplottools ] ; then git clone https://github.com/bingmann/sqlplot-tools.git sqlplottools; fi
	mkdir -p sqlplottools/build
	cd sqlplottools/build && cmake .. && $(MAKE)

json:
	mkdir -p result
	python json_to_result_converter.py plotconfig.json $(BENCHMARK) result

sql: json sqlplottools
	for file in $(shell ls *.tex); do \
		./sqlplottools/build/src/sp-process $$file; \
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
