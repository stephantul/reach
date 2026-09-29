VERBOSITY=

venv:
	uv venv

install:
	uv sync --all-extras
	uv run pre-commit install

fix:
	uv run pre-commit run --all-files

test:
	uv run pytest --cov=reach --cov-report=term-missing

test-verbose:
	make test VERBOSITY="-vvv"
