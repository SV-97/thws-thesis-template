# THWS latex template

Latex template based on my bachelor's thesis at THWS Würzburg-Schweinfurt. Note that this is no official template.

## Building

Run these commands in succession

1. `xelatex`
2. `biber`
3. `xelatex`
4. `xelatex`

[FiraCode](https://github.com/tonsky/FiraCode) can be enabled in prelude line 74. By removing stuff like minted it's also be possible to build using pdflatex.

## Basic structure

* images go into `img`
* code for listings into `listings`
* code for image generation etc. (should you need it) into `scripts`
* sections of the document into `sections` (to be included in `main.tex`)
* references into `sources.bib`
* `misc` contains files that usually don't need to be touched as often (`prelude` etc.)
