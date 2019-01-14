TEX = latexmk
FILE = main
BENCHMARK = plots.json
EXTENSION = .tex

.PHONY: all clean run

all: main

sqlplottools:
	git clone https://github.com/bingmann/sqlplot-tools.git sqlplottools
	mkdir -p sqlplottools/build
	cd sqlplottools/build && cmake .. && $(MAKE)

json:
	mkdir -p result
	python json_to_result_converter.py $(BENCHMARK) result

sql: json sqlplottools
	./sqlplottools/build/src/sp-process $(FILE)$(EXTENSION)

main: sql
	$(TEX) -pdf $(FILE)

run: main
	$(TEX) -pdf -pv $(FILE)

clean:
	$(TEX) -C
