name=presentation

report:
	pdflatex $(name)
	bibtex $(name)
	pdflatex $(name)
	pdflatex $(name)
clean:
	rm -f $(name).out $(name).aux $(name).log $(name).bbl $(name).blg $(name).out $(name).toc $(name).nav $(name).snm
