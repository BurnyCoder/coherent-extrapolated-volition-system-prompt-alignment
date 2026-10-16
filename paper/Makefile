# Build the paper with latexmk (pdfLaTeX + BibTeX); intermediates go to build/ (git-ignored).
# Pattern adapted from the author's fact-teaching/paper/Makefile.
LATEXMK ?= latexmk
BUILD_DIR := build
FINAL_PDF := cev-system-prompt-alignment.pdf
SOURCES := main.tex references.bib $(wildcard sections/*.tex) $(wildcard appendices/*.tex) $(wildcard generated/*.tex)

.PHONY: all clean

all: $(FINAL_PDF)

$(FINAL_PDF): $(BUILD_DIR)/main.pdf
	cp $(BUILD_DIR)/main.pdf $(FINAL_PDF)

$(BUILD_DIR)/main.pdf: $(SOURCES)
	mkdir -p $(BUILD_DIR)
	$(LATEXMK) -pdf -bibtex -interaction=nonstopmode -halt-on-error -file-line-error -outdir=$(BUILD_DIR) main.tex

clean:
	$(LATEXMK) -C -outdir=$(BUILD_DIR) main.tex
	rm -rf $(BUILD_DIR)
