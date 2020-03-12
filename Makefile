all: run notebook

.PHONY: run
run:
	python3 model.py

.PHONY: notebook
notebook:
	jupyter notebook

