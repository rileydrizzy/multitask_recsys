.DEFAULT_GOAL := help

help:
	@echo "    prepare              desc of the command prepare"
	@echo "    install              desc of the command install"

precommit:
	@echo "Running precommit on all files"
	pre-commit run --all-files

export:
	@echo "Exporting dependencies to requirements file"
	python -m pip freeze > requirements.txt

## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf .pytest_cache