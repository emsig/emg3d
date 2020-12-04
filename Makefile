help:
	@echo "Commands:"
	@echo ""
	@echo "  install      install in editable mode"
	@echo "  dev-install  install in editable mode with dev requirements"
	@echo "  pytest       run the test suite and report coverage"
	@echo "  flake8       style check with flake8"
	@echo "  doc          build docs (new, removing any existing)"
	@echo "  doc-update   build docs (update existing)"
	@echo "  linkcheck    check all links in docs"
	@echo "  clean        clean up all generated files"
	@echo ""

install:
	pip install -e .

dev-install:
	pip install -r requirements-dev.txt && pip install -e .

pytest:
	pytest --cov=emg3d tests/ --flake8 && coverage html

flake8:
	flake8 docs/conf.py setup.py emg3d/ tests/

doc:
	cd docs && rm -rf api/ && rm -rf _build/ && make html && cd ..

doc-update:
	cd docs && make html && cd ..

linkcheck:
	cd docs && make html -b linkcheck && cd ..

clean:
	rm -rf build/ dist/ .eggs/ emg3d.egg-info/ emg3d/version.py  # build
	rm -rf */__pycache__/ */*/__pycache__/      # python cache
	rm -rf .coverage htmlcov/ .pytest_cache/    # tests and coverage
	rm -rf docs/api/ docs/_build/               # docs
