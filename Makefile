# Makefile with some convenient quick ways to do common things

PROJECT = iop4

help:
	@echo ''
	@echo '     help               Print this help message (the default)'
	@echo ''
	@echo '     test               Run pytest'
	@echo '     test-cov           Run pytest with coverage'
	@echo ''
	@echo '     docs-sphinx        Build docs (Sphinx only)'
	@echo '     docs-show          Open local HTML docs in browser'
	@echo ''


test:
	python -m pytest -vx tests/

test-cov:
	python -m pytest -vx --cov=iop4lib tests/

docs-sphinx:
	#cd docs && make clean && make html
	cd docs && python -m sphinx . _build/html -b html

docs-show:
	open docs/_build/html/index.html