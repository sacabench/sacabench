TEX = latexmk
FILE = main2
EXTENSION = .tex

.PHONY: all clean run

all: main

sqlplottools:
	git clone https://github.com/bingmann/sqlplot-tools.git sqlplottools
	mkdir -p sqlplottools/build
	cd sqlplottools/build && cmake .. && $(MAKE)

sql: sqlplottools
	./sqlplottools/build/src/sp-process $(FILE)$(EXTENSION)

main: sql
	$(TEX) -pdf $(FILE)

run: main
	$(TEX) -pdf -pv $(FILE)

clean:
	$(TEX) -C
