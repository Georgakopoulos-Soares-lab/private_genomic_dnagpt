# latexmk configuration for the manuscript.
$pdf_mode = 1;              # pdflatex; switch to 5 for xelatex if a font demands it
$bibtex_use = 2;            # run bibtex, and clean the .bbl on -C
$out_dir = '.';
$clean_ext = 'bbl nav snm run.xml synctex.gz fdb_latexmk fls';
$pdflatex = 'pdflatex -interaction=nonstopmode -halt-on-error -file-line-error %O %S';
