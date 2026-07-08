ifeq ($(OS),Windows_NT)
BOOTSTRAP_PYTHON := py -3
VENV_PYTHON := .venv/Scripts/python.exe
ACTIVATE_HINT := .venv\Scripts\Activate.ps1
else
BOOTSTRAP_PYTHON := python3
VENV_PYTHON := .venv/bin/python
ACTIVATE_HINT := source .venv/bin/activate
endif

INPUT ?= examples/tasks.json
OUTPUT ?= output/results.json
IMAGE ?= gemmaclip:latest

.PHONY: help venv install-dev test run-local docker-build docker-run-example clean

help:
	@echo "Targets:"
	@echo "  make venv               Create .venv if missing"
	@echo "  make install-dev        Install the project and dev dependencies into .venv"
	@echo "  make test              Run pytest in .venv"
	@echo "  make run-local         Run the pipeline with INPUT=$(INPUT) OUTPUT=$(OUTPUT)"
	@echo "  make docker-build      Build the linux/amd64 Docker image"
	@echo "  make docker-run-example Run the container with ./input and ./output mounts"
	@echo "  make clean             Remove output artifacts"
	@echo ""
	@echo "Activation hint: $(ACTIVATE_HINT)"

venv:
	$(BOOTSTRAP_PYTHON) -c "from pathlib import Path; import venv; path = Path('.venv'); path.exists() or venv.create(path, with_pip=True)"

install-dev: venv
	"$(VENV_PYTHON)" -m pip install -e .[dev]

test:
	"$(VENV_PYTHON)" -m pytest

run-local:
	"$(VENV_PYTHON)" -c "from pathlib import Path; Path('$(OUTPUT)').parent.mkdir(parents=True, exist_ok=True)"
	"$(VENV_PYTHON)" -m gemmaclip.main --input "$(INPUT)" --output "$(OUTPUT)"

docker-build:
	docker buildx build --platform linux/amd64 --tag "$(IMAGE)" .

docker-run-example:
	docker run --rm -v "${PWD}/input:/input" -v "${PWD}/output:/output" "$(IMAGE)"

clean:
	"$(VENV_PYTHON)" -c "from pathlib import Path; [path.unlink() for path in Path('output').glob('*') if path.is_file()]"
