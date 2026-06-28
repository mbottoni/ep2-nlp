.PHONY: install install-dev lint test baselines regression classification clean

install:           ## install the package and runtime deps
	pip install -e .

install-dev:       ## install with dev tools (pytest, ruff)
	pip install -e ".[dev]"

lint:              ## run ruff
	ruff check src tests

test:              ## run the unit tests (no GPU needed)
	pytest -q

baselines:         ## evaluate the naive density baselines
	bertimbau-probing baselines

regression:        ## fine-tune the regression probe
	bertimbau-probing regression

classification:    ## fine-tune the classification probe (balanced classes)
	bertimbau-probing classification --balanced

clean:
	rm -rf artifacts *.pth .pytest_cache **/__pycache__ src/*.egg-info
