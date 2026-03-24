# Bibliography is inline (thebibliography environment), no .bib file.
# Use pdflatex only — skip bibtex/biber entirely.
$pdf_mode = 1;
$bibtex_use = 0;
$max_repeat = 5;
$warnings_as_errors = 0;
$pdflatex = 'pdflatex -interaction=nonstopmode -synctex=1 %O %S';
