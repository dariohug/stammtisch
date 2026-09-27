PY    ?= .venv/bin/python
BUILD ?= build

.PHONY: all engine venv data validate test sim monitor bench report clean train arena nightly

all: engine

engine:
	cmake -S engine -B $(BUILD) -DCMAKE_BUILD_TYPE=Release >/dev/null
	cmake --build $(BUILD) -j

venv:
	python3 -m venv .venv
	$(PY) -m pip install -q -e '.[dev]'
	$(PY) -m pip install -q torch --index-url https://download.pytorch.org/whl/cpu

data:            ## download Swisslos logs (see data/README.md) and convert to npz
	./scripts/fetch_data.sh
	$(PY) -m stammtisch ingest

validate: engine
	$(PY) -m stammtisch validate

test: engine
	$(BUILD)/engine-test
	$(PY) -m pytest -q tests

train:           ## imitation networks on the Swisslos logs (~1 h on a laptop CPU)
	$(PY) -m stammtisch train card --steps 60000
	$(PY) -m stammtisch train trump --epochs 6

arena: engine
	$(PY) -m stammtisch arena

nightly:
	./scripts/nightly.sh

monitor: engine  ## live dashboard, e.g. make monitor ARGS="--seconds 60 --a heuristic --b random"
	$(PY) -m stammtisch monitor $(ARGS)

bench: engine
	$(PY) -m stammtisch bench --seconds 3

report:          ## regenerate numbers + figures data (quiet machine!), then compile docs/report.pdf
	$(PY) -m stammtisch report-data
	$(PY) -m stammtisch monitor --seconds 4
	cp "$$(ls -t runs/*.txt | head -1)" docs/data/monitor.txt
	cd docs && ../.tools/tectonic -X compile report.tex

clean:
	rm -rf $(BUILD)
