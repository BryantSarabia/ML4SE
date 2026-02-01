# Makefile for LaTeX Report Compilation
# Toxic Comment Classification - ML4SE Project

MAIN = main
PDF = $(MAIN).pdf

.PHONY: all clean cleanall quick

# Default target: full compilation with bibliography
all: $(PDF)

$(PDF): $(MAIN).tex references.bib
	pdflatex -interaction=nonstopmode $(MAIN)
	bibtex $(MAIN)
	pdflatex -interaction=nonstopmode $(MAIN)
	pdflatex -interaction=nonstopmode $(MAIN)
	@echo "PDF generated successfully: $(PDF)"

# Quick compilation (no bibliography update)
quick:
	pdflatex -interaction=nonstopmode $(MAIN)
	@echo "Quick PDF generated: $(PDF)"

# Clean auxiliary files
clean:
	rm -f *.aux *.log *.bbl *.blg *.out *.toc *.synctex.gz

# Clean everything including PDF
cleanall: clean
	rm -f $(PDF)

# Help
help:
	@echo "Available targets:"
	@echo "  make        - Full compilation (default)"
	@echo "  make quick  - Quick compilation without bibliography"
	@echo "  make clean  - Remove auxiliary files"
	@echo "  make cleanall - Remove all generated files"
