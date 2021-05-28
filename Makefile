help:
	@echo "Commands:"
	@echo ""
	@echo "  install        install in editable mode"
	@echo "  dev-install    install in editable mode with dev requirements"
	@echo "  pytest         run the test suite and report coverage"
	@echo "  flake8         style check with flake8"
	@echo "  html           build docs (update existing)"
	@echo "  html-noplot    as above, without gallery"
	@echo "  example FILE=  build particular example"
	@echo "  html-clean     build docs (new, removing any existing)"
	@echo "  preview        renders docs in Browser"
	@echo "  linkcheck      check all links in docs"
	@echo "  linkcheck-noplot"
	@echo "  clean          clean up all generated files"
	@echo ""

install:
	pip install -e .

dev-install:
	pip install -r requirements-dev.txt && pip install -e .

pytest:
	rm -rf .coverage htmlcov/ .pytest_cache/ && pytest --cov=emg3d --flake8 && coverage html

flake8:
	flake8 docs/ setup.py emg3d/ tests/ examples/

html:
	cd docs && make html

html-noplot:
	cd docs && make htm-noplot

html-clean:
	cd docs && rm -rf api/emg3d* && rm -rf _build/ && rm -rf gallery/*/ && make html

example:
	cd docs && sphinx-build -D sphinx_gallery_conf.filename_pattern=$(FILE) -b html -d _build/doctrees . _build/html

preview:
	xdg-open docs/_build/html/index.html

linkcheck:
	cd docs && make linkcheck

linkcheck-noplot:
	cd docs && make linkcheck-noplot

clean:
	rm -rf build/ dist/ .eggs/ emg3d.egg-info/ emg3d/version.py  # build
	rm -rf */__pycache__/ */*/__pycache__/      # python cache
	rm -rf .coverage htmlcov/ .pytest_cache/    # tests and coverage
	rm -rf docs/api/emg3d* docs/_build/ docs/savefig/ # docs
	rm -rf docs/gallery/*/ docs/gallery/*.zip   # gallery
