all: venv run notebook

.PHONY: venv
venv:
	source bin/activate

.PHONY: run
run:
	python3 model.py

.PHONY: notebook
notebook:
	jupyter notebook

