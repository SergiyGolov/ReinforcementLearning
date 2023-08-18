all: compile-scss generate

generate:
	pandoc -t revealjs -s -o presentation.html presentation.md  -V revealjs-url=./reveal.js -V css=slides.css

compile-scss:
	sass ./sass/app.scss ./slides.css