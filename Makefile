clean.notebooks:
	@find . -name "*.ipynb" -exec nbstripout {} \;

test:
	pytest -s .
