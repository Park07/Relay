# Relay — developer entry points (DESIGN.md §11).
# The headline target is `bench`: it produces the Pareto frontier locally, for
# free, with no GPU. `test` runs the fast dependency-free unit suite.

PY ?= python
export PYTHONPATH := $(CURDIR)

.PHONY: help install install-dev test bench bench-quick calibrate proto \
        compose-up compose-down helm-install helm-uninstall lint fmt clean

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

install: ## Install core deps (enough for the benchmark)
	$(PY) -m pip install -e ".[bench]"

install-dev: ## Install everything for local dev (all services + tooling)
	$(PY) -m pip install -e ".[gateway,scheduler,worker,bench,dev]"

test: ## Run the fast unit-test suite
	$(PY) -m pytest tests/unit

bench: ## Run the full prefix-routing sweep → bench/results/{frontier.csv,png,RESULTS.md}
	$(PY) bench/run.py

bench-quick: ## Smaller, faster sweep for a quick look
	$(PY) bench/run.py --quick

calibrate: ## Fit MockEngine alpha/beta (synthetic if no Ollama present)
	$(PY) bench/calibrate.py

proto: ## Generate gRPC stubs from proto/relay/v1/worker.proto into services/_gen
	cd proto && buf generate

compose-up: ## Bring up the full local stack (redis/pg/scheduler/gateway/4 workers/prom/grafana)
	docker compose -f deploy/compose/docker-compose.yml up --build

compose-down: ## Tear the local stack down
	docker compose -f deploy/compose/docker-compose.yml down -v

helm-install: ## Install the chart to the current kube-context (k3d)
	helm install relay deploy/helm/relay

helm-uninstall:
	helm uninstall relay

lint: ## Ruff lint
	$(PY) -m ruff check .

fmt: ## Ruff format
	$(PY) -m ruff format .

clean:
	rm -rf .pytest_cache **/__pycache__ services/_gen
