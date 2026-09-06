.DEFAULT_GOAL := help
SHELL := /bin/bash

VENV       := .venv
PY         := $(VENV)/bin/python
PIP        := $(VENV)/bin/pip
PYTEST     := $(VENV)/bin/pytest
RUFF       := $(VENV)/bin/ruff
MYPY       := $(VENV)/bin/mypy
ALEMBIC    := $(VENV)/bin/alembic
UVICORN    := $(VENV)/bin/uvicorn
COMPOSE    := docker compose

.PHONY: help
help: ## Show this help
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
.PHONY: venv
venv: ## Create the virtual environment
	python3.12 -m venv $(VENV) || python3 -m venv $(VENV)
	$(PIP) install --upgrade pip setuptools wheel

.PHONY: install
install: venv ## Install runtime dependencies
	$(PIP) install -r requirements.txt

.PHONY: install-dev
install-dev: venv ## Install runtime + development dependencies
	$(PIP) install -r requirements-dev.txt

.PHONY: lock
lock: ## Freeze the resolved dependency set into requirements.lock.txt
	$(PIP) freeze --exclude-editable > requirements.lock.txt
	@echo "wrote requirements.lock.txt"

.PHONY: env
env: ## Create .env from the template if it does not exist
	@test -f .env && echo ".env already exists -- leaving it alone" \
		|| (cp .env.example .env && echo "created .env -- now fill in your credentials")

.PHONY: setup
setup: install-dev env ## One-shot local setup
	@echo "setup complete -- next: make db-up && make migrate"

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
# No --log-config flag: backend.main installs our dictConfig at import time,
# which runs after uvicorn's default and therefore wins.
.PHONY: run
run: ## Run the API with hot reload
	$(UVICORN) backend.main:app --reload --host 0.0.0.0 --port 8000

.PHONY: shell
shell: ## Python REPL with the app importable
	$(PY)

# ---------------------------------------------------------------------------
# Racing API / ingestion
# ---------------------------------------------------------------------------
.PHONY: api-check
api-check: ## Verify Racing API credentials and subscription state
	$(PY) -m scripts.racing_cli check

.PHONY: ingest-today
ingest-today: ## Import today's racecards
	$(PY) -m scripts.racing_cli racecards

.PHONY: ingest-racecards
ingest-racecards: ## Import racecards:  make ingest-racecards date=2026-08-10 tier=pro
	$(PY) -m scripts.racing_cli racecards $(if $(date),--date $(date)) $(if $(tier),--tier $(tier))

.PHONY: ingest-results
ingest-results: ## Import results:  make ingest-results start=2026-08-01 end=2026-08-07
	@test -n "$(start)" -a -n "$(end)" || (echo 'usage: make ingest-results start=YYYY-MM-DD end=YYYY-MM-DD' && exit 1)
	$(PY) -m scripts.racing_cli results --start $(start) --end $(end)

.PHONY: backfill
backfill: ## Backfill day by day:  make backfill start=2025-08-01 end=2026-08-01
	@test -n "$(start)" -a -n "$(end)" || (echo 'usage: make backfill start=YYYY-MM-DD end=YYYY-MM-DD' && exit 1)
	$(PY) -m scripts.racing_cli backfill --start $(start) --end $(end)

# ---------------------------------------------------------------------------
# Research (Phase 3)
# ---------------------------------------------------------------------------
.PHONY: quality
quality: ## Data quality report over stored racing data
	$(PY) -m scripts.research_cli quality

.PHONY: features
features: ## Build point-in-time features:  make features store=1 manifest=1
	$(PY) -m scripts.research_cli features $(if $(store),--store) $(if $(manifest),--manifest)

.PHONY: dataset
dataset: ## Build the research dataset:  make dataset out=data/processed/train.parquet
	$(PY) -m scripts.research_cli dataset --out $(or $(out),data/processed/dataset.parquet)

.PHONY: backtest
backtest: ## Backtest a baseline:  make backtest predictor=all
	$(PY) -m scripts.research_cli backtest --predictor $(or $(predictor),market)

.PHONY: synthetic
synthetic: ## DEV ONLY: fill the database with synthetic races
	$(PY) -m scripts.research_cli synthetic --days $(or $(days),180)

# ---------------------------------------------------------------------------
# Modelling (Phase 4)
# ---------------------------------------------------------------------------
.PHONY: train
train: ## Train, calibrate, evaluate and register:  make train detail=1
	$(PY) -m scripts.ml_cli train $(if $(detail),--detail) $(if $(adaptive),--adaptive)

.PHONY: train-no-market
train-no-market: ## Market-free control: can the other features predict anything?
	$(PY) -m scripts.ml_cli train --no-market $(if $(adaptive),--adaptive)

.PHONY: compare
compare: ## Model comparison table from the registry
	$(PY) -m scripts.ml_cli compare

.PHONY: importance
importance: ## Feature importance for the champion model
	$(PY) -m scripts.ml_cli importance

.PHONY: predict
predict: ## Score a race or a day:  make predict race=rac_123  |  make predict day=2026-08-10
	@test -n "$(race)" -o -n "$(day)" || (echo 'usage: make predict race=<id> | day=YYYY-MM-DD' && exit 1)
	$(PY) -m scripts.ml_cli predict $(if $(race),--race-id $(race)) $(if $(day),--date $(day))

# ---------------------------------------------------------------------------
# Value betting (Phase 5)
# ---------------------------------------------------------------------------
.PHONY: signals
signals: ## Walk-forward predictions (expensive, cached):  make signals nomarket=1
	$(PY) -m scripts.strategy_cli predictions $(if $(nomarket),--no-market)

.PHONY: strategy-backtest
strategy-backtest: ## Backtest one strategy:  make strategy-backtest strategy=A_ev5
	$(PY) -m scripts.strategy_cli backtest --strategy $(or $(strategy),A_ev5)

.PHONY: strategy-compare
strategy-compare: ## Compare every strategy over identical predictions
	$(PY) -m scripts.strategy_cli compare

.PHONY: market-dependency
market-dependency: ## Does the model add anything the market does not know?
	$(PY) -m scripts.strategy_cli market-dependency

.PHONY: sensitivity
sensitivity: ## How fast does the edge vanish under realistic costs?
	$(PY) -m scripts.strategy_cli sensitivity

# ---------------------------------------------------------------------------
# Real data validation (Phase 6)
#
# Run in this order. Each step refuses to proceed if the one before it has not
# genuinely succeeded, which is the point: the gates are the deliverable.
# ---------------------------------------------------------------------------
.PHONY: protocol
protocol: ## Print the pre-registered analysis plan (frozen before data exists)
	$(PY) -m scripts.phase6_cli protocol

.PHONY: api-status
api-status: ## Auth, subscription and per-endpoint access. Everything below is gated on this
	$(PY) -m scripts.phase6_cli api-status

.PHONY: history
history: ## Resumable backfill:  make history start=2018-01-01 end=2026-06-30
	$(PY) -m scripts.phase6_cli ingest --start $(or $(start),2018-01-01) $(if $(end),--end $(end))

.PHONY: history-progress
history-progress: ## How far the backfill got
	$(PY) -m scripts.phase6_cli progress

.PHONY: history-runs
history-runs: ## The backfill audit trail from the backfill_runs table
	$(PY) -m scripts.phase6_cli runs

.PHONY: audit
audit: ## Data quality audit -> reports/REAL_DATA_AUDIT.md
	$(PY) -m scripts.phase6_cli audit

.PHONY: phase6-preflight
phase6-preflight: ## READY FOR VALIDATION, or BLOCKED with the one reason why
	$(PY) -m scripts.phase6_cli phase6-preflight

.PHONY: phase6-run
phase6-run: ## THE ONE BUTTON: ingest -> audit -> features -> train -> predict -> backtest -> report
	$(PY) -m scripts.phase6_cli phase6-run

.PHONY: phase6-steps
phase6-steps: ## How far the last workflow run got
	$(PY) -m scripts.phase6_cli steps

.PHONY: experiments
experiments: ## The research run ledger — every validation, and whether it is reproducible
	$(PY) -m scripts.phase6_cli experiments

.PHONY: validate
validate: ## Validation only, on data already downloaded
	$(PY) -m scripts.phase6_cli phase6-validate

.PHONY: validate-harness
validate-harness: ## Exercise the whole workflow on synthetic data. NOT a validation.
	$(PY) -m scripts.phase6_cli phase6-run --skip-ingest --allow-unready --no-pin-protocol

# ---------------------------------------------------------------------------
# The product (Phase 7)
#
# train once, then run `predictions` every morning.
# ---------------------------------------------------------------------------
.PHONY: train-production
train-production: ## Train the production model from history and promote the best
	$(PY) -m scripts.predict_cli train

.PHONY: predictions
predictions: ## THE MORNING JOB: fetch today's card, score it, store the signals
	$(PY) -m scripts.predict_cli daily

.PHONY: today
today: ## Score a day without storing:  make today day=2026-08-10
	$(PY) -m scripts.predict_cli today $(if $(day),--race-date $(day)) --bets-only

.PHONY: model-status
model-status: ## Which model is live, and the betting rules in force
	$(PY) -m scripts.predict_cli status

.PHONY: results
results: ## How the recommendations actually did, on settled bets only
	$(PY) -m scripts.predict_cli performance

.PHONY: replay
replay: ## Does live scoring match the backtest?  make replay day=2026-06-15
	@test -n "$(day)" || (echo 'usage: make replay day=YYYY-MM-DD' && exit 1)
	$(PY) -m scripts.predict_cli replay $(day)

.PHONY: dashboard
dashboard: ## Serve the API and dashboard at http://localhost:8000/dashboard/
	@echo "dashboard -> http://localhost:8000/dashboard/"
	$(UVICORN) backend.main:app --host 0.0.0.0 --port 8000

# ---------------------------------------------------------------------------
# Quality
# ---------------------------------------------------------------------------
.PHONY: test
test: ## Run the full test suite with coverage, in parallel
	$(PYTEST) -n auto --dist loadfile --cov=backend --cov-report=term-missing

.PHONY: test-fast
test-fast: ## Unit tests only, in parallel, no coverage — the inner-loop check
	$(PYTEST) tests/unit -n auto --dist loadfile -q

.PHONY: test-unit
test-unit: ## Run unit tests only (no database required)
	$(PYTEST) -m "not integration and not network and not slow"

.PHONY: test-integration
test-integration: ## Run tests that need PostgreSQL
	$(PYTEST) -m integration

.PHONY: lint
lint: ## Lint with ruff
	$(RUFF) check backend tests scripts

.PHONY: format
format: ## Auto-format and fix imports
	$(RUFF) format backend tests scripts
	$(RUFF) check --fix backend tests scripts

.PHONY: typecheck
typecheck: ## Static type checking
	$(MYPY) backend

.PHONY: check
check: lint typecheck test ## Everything CI runs

# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------
.PHONY: db-up
db-up: ## Start PostgreSQL + Redis
	$(COMPOSE) up -d postgres redis

.PHONY: db-down
db-down: ## Stop PostgreSQL + Redis
	$(COMPOSE) stop postgres redis

.PHONY: db-shell
db-shell: ## psql into the running database
	$(COMPOSE) exec postgres psql -U $${POSTGRES_USER:-horse_quant} -d $${POSTGRES_DB:-horse_quant}

.PHONY: migrate
migrate: ## Apply all migrations
	$(ALEMBIC) upgrade head

.PHONY: migration
migration: ## Autogenerate a migration:  make migration m="add races table"
	@test -n "$(m)" || (echo 'usage: make migration m="describe the change"' && exit 1)
	$(ALEMBIC) revision --autogenerate -m "$(m)"

.PHONY: downgrade
downgrade: ## Roll back one migration
	$(ALEMBIC) downgrade -1

.PHONY: db-reset
db-reset: ## DESTRUCTIVE: drop the postgres volume and recreate an empty database
	@read -p "This deletes all local racing data. Type 'yes' to continue: " ok; \
	 [ "$$ok" = "yes" ] || (echo "aborted" && exit 1)
	$(COMPOSE) rm -sf postgres
	docker volume rm hqp_postgres_data || true
	$(MAKE) db-up

# ---------------------------------------------------------------------------
# Docker
# ---------------------------------------------------------------------------
.PHONY: build
build: ## Build the API image
	$(COMPOSE) build api

.PHONY: up
up: ## Start the full stack
	$(COMPOSE) up -d

.PHONY: down
down: ## Stop the stack (volumes preserved)
	$(COMPOSE) down

.PHONY: logs
logs: ## Tail container logs
	$(COMPOSE) logs -f --tail=100

.PHONY: ps
ps: ## Show container status
	$(COMPOSE) ps

# ---------------------------------------------------------------------------
# Housekeeping
# ---------------------------------------------------------------------------
.PHONY: clean
clean: ## Remove caches and build artefacts
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage coverage.xml build dist *.egg-info

.PHONY: clean-logs
clean-logs: ## Truncate log files
	find logs -name "*.log*" -delete
