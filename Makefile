help:
	@echo "Commands:"
	@echo ""
	@echo "  install      install in editable mode"
	@echo "  dev-install  install in editable mode with dev requirements"
	@echo "  pytest       run the test suite and report coverage"
	@echo "  flake8       style check with flake8"
	@echo "  docs         build docs (new, removing any existing)"
	@echo "  docs-update  build docs (update existing)"
	@echo "  linkcheck    check all links in docs"
	@echo "  clean        clean up all generated files"
	@echo ""

install:
	pip install -e .

dev-install:
	pip install -r requirements-dev.txt && pip install -e .

pytest:
	pytest --cov=emg3d tests/ --flake && coverage html

flake8:
	flake8 docs/conf.py setup.py emg3d/ tests/

docs:
	cd docs && rm -rf api/ && rm -rf _build/ && make html && cd ..

docs-update:
	cd docs && make html && cd ..

linkcheck:
	cd docs && make html -b linkcheck && cd ..

clean:
	rm -rf build/ dist/ .eggs/ emg3d.egg-info/
	rm -rf emg3d/__pycache__/ emg3d/cli/__pycache__/ tests/__pycache__/
	rm -rf .coverage pytest_cache htmlcov/ .pytest_cache/
	rm -rf docs/api/ docs/_build/
