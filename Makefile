# Evidex local runtime entry point.
#
# `make up` builds and starts the full personal stack: doctor preflight,
# PostgreSQL role reconciliation, revision-labelled images, Alembic
# migrations, then API/Worker/frontend with health gating. See `make help`.

SHELL := /bin/sh
.SHELLFLAGS := -eu -c

COMPOSE_PROJECT := rag
MANIFEST := .env.local
COMPOSE := docker compose --env-file $(MANIFEST) --project-name $(COMPOSE_PROJECT)
FRONTEND_DIR := apps/web-chat
LOG_DIRECTORY := .runtime/logs
PYTHON := $(shell if [ -x .venv/bin/python ]; then echo .venv/bin/python; else echo python3; fi)
SERVICES ?= api worker

export RAG_KB_BUILD_REVISION := $(shell git rev-parse HEAD 2>/dev/null || echo unknown)

# Build-time wiring only matters for targets that build images; keeping the
# probes behind MAKECMDGOALS keeps `make help`/`make down` instant.
ifneq ($(filter up build restart frontend-dist,$(MAKECMDGOALS)),)

# Reuse the verified model artifacts from the current local image when
# present; otherwise compose.yaml falls back to the pinned Python base image.
ifneq ($(shell docker image inspect rag-kb-app:local >/dev/null 2>&1 && echo yes),)
export RAG_KB_BUILD_MODEL_ASSET_CONTEXT := docker-image://rag-kb-app:local
endif

# A verified host Node toolchain builds the frontend once on the host and the
# image just packages dist/ (prebuilt-runtime); otherwise it builds in Docker.
HOST_FRONTEND_OK := $(shell if command -v node >/dev/null 2>&1 && command -v npm >/dev/null 2>&1 && [ -d $(FRONTEND_DIR)/node_modules ] && npm --prefix $(FRONTEND_DIR) ls --all >/dev/null 2>&1; then echo yes; fi)
ifeq ($(HOST_FRONTEND_OK),yes)
export RAG_KB_BUILD_FRONTEND_TARGET := prebuilt-runtime
export RAG_KB_BUILD_FRONTEND_DIST_CONTEXT := $(FRONTEND_DIR)/dist
endif

endif

# Read one key from the local manifest without evaluating it.
manifest_value = $(shell awk -v key="$(1)" 'index($$0, key "=") == 1 { print substr($$0, length(key) + 2); exit }' $(MANIFEST) 2>/dev/null)
API_PORT := $(or $(call manifest_value,RAG_KB_API_PORT),8000)
FRONTEND_PORT := $(or $(call manifest_value,RAG_KB_FRONTEND_PORT),3000)

.PHONY: help preflight doctor prepare frontend-dist up down restart ps logs build migrate smoke clean

help:
	@printf '%s\n' \
		"Evidex local runtime (Compose project: $(COMPOSE_PROJECT), manifest: $(MANIFEST))" \
		"" \
		"make up       Preflight, build, migrate, and start the full stack" \
		"make down     Stop the stack (volumes are retained)" \
		"make restart  down + up" \
		"make ps       Show service status" \
		"make logs     Follow API/Worker logs (override with SERVICES='api')" \
		"make build    Rebuild the application and frontend images" \
		"make migrate  Apply Alembic migrations" \
		"make doctor   Run the content-safe runtime preflight only" \
		"make smoke    Run the local smoke check ($(PYTHON))" \
		"make clean    down --remove-orphans; data reset stays with tools/reset_local.py"

preflight:
	@command -v git >/dev/null 2>&1 || { printf '%s\n' "Local startup failed: Git is not installed or not on PATH" >&2; exit 1; }
	@command -v python3 >/dev/null 2>&1 || { printf '%s\n' "Local startup failed: Python 3 is not installed or not on PATH" >&2; exit 1; }
	@command -v docker >/dev/null 2>&1 || { printf '%s\n' "Local startup failed: Docker is not installed or not on PATH" >&2; exit 1; }
	@docker info >/dev/null 2>&1 || { printf '%s\n' "Local startup failed: Docker is not running" >&2; exit 1; }
	@[ -f $(MANIFEST) ] || { printf '%s\n' "Local startup failed: $(MANIFEST) is missing; copy .env.example and replace every placeholder" >&2; exit 1; }
	@[ -z "$${RAG_KB_LOCAL_COMPOSE_ENV_FILE:-}" ] || { printf '%s\n' "Local startup failed: RAG_KB_LOCAL_COMPOSE_ENV_FILE is retired; use $(MANIFEST)" >&2; exit 1; }
	@[ -z "$${RAG_KB_LOCAL_APP_ENV_FILE:-}" ] || { printf '%s\n' "Local startup failed: RAG_KB_LOCAL_APP_ENV_FILE is retired; use $(MANIFEST)" >&2; exit 1; }
	@[ "$${COMPOSE_PROJECT_NAME:-$(COMPOSE_PROJECT)}" = "$(COMPOSE_PROJECT)" ] || { printf '%s\n' "Local startup failed: the personal Compose project is fixed to $(COMPOSE_PROJECT)" >&2; exit 1; }
	@common_directory=$$(git rev-parse --path-format=absolute --git-common-dir 2>/dev/null) || { printf '%s\n' "Local startup failed: cannot resolve the primary checkout" >&2; exit 1; }; \
	case "$$common_directory" in \
		*/.git) primary_checkout=$$(dirname "$$common_directory") ;; \
		*) printf '%s\n' "Local startup failed: Git common directory is not a primary checkout" >&2; exit 1 ;; \
	esac; \
	[ "$$(pwd)" = "$$primary_checkout" ] || { printf '%s\n' "Local startup failed: run make from the primary checkout; linked worktrees are read-only for the personal runtime" >&2; exit 1; }

doctor: preflight
	python3 tools/local_runtime.py doctor

prepare:
	@[ ! -L .runtime ] || { printf '%s\n' "Local startup failed: runtime directory must not be a symbolic link" >&2; exit 1; }
	@[ ! -L $(LOG_DIRECTORY) ] || { printf '%s\n' "Local startup failed: log directory must not be a symbolic link" >&2; exit 1; }
	@[ ! -e .runtime ] || [ -d .runtime ] || { printf '%s\n' "Local startup failed: runtime path is not a directory" >&2; exit 1; }
	@[ ! -e $(LOG_DIRECTORY) ] || [ -d $(LOG_DIRECTORY) ] || { printf '%s\n' "Local startup failed: log path is not a directory" >&2; exit 1; }
	@mkdir -p $(LOG_DIRECTORY)
	@chmod 700 .runtime $(LOG_DIRECTORY)

frontend-dist:
ifeq ($(HOST_FRONTEND_OK),yes)
	@printf '%s\n' "Building frontend with the verified host dependency graph..."
	npm --prefix $(FRONTEND_DIR) run build
	@[ ! -L $(FRONTEND_DIR)/dist/runtime-config.json ] || { printf '%s\n' "Local startup failed: generated frontend runtime config must not be a symbolic link" >&2; exit 1; }
	@rm -f -- $(FRONTEND_DIR)/dist/runtime-config.json
	@[ -f $(FRONTEND_DIR)/dist/index.html ] || { printf '%s\n' "Local startup failed: host frontend build did not produce dist/index.html" >&2; exit 1; }
else
	@:
endif

up: doctor prepare frontend-dist
	@printf '%s\n' "Using canonical Compose project $(COMPOSE_PROJECT) and owner-only $(MANIFEST)."
	@printf 'Building Git revision %s.\n' "$(RAG_KB_BUILD_REVISION)"
	$(COMPOSE) up -d --wait postgres
	$(COMPOSE) exec -T postgres /docker-entrypoint-initdb.d/10-init-runtime.sh
	$(COMPOSE) build api frontend
	$(COMPOSE) up storage-init
	$(COMPOSE) --profile tools run --rm migrate
	$(COMPOSE) up -d --wait api worker frontend
	@$(COMPOSE) ps
	@printf '\nRAG KB is ready.\n'
	@printf 'User Chat: http://127.0.0.1:%s\n' "$(FRONTEND_PORT)"
	@printf 'API docs: http://127.0.0.1:%s/api/v1/docs\n' "$(API_PORT)"

down:
	$(COMPOSE) down

restart: down up

ps:
	@$(COMPOSE) ps

logs:
	$(COMPOSE) logs --no-color -f $(SERVICES)

build: frontend-dist
	$(COMPOSE) build api frontend

migrate:
	$(COMPOSE) --profile tools run --rm migrate

smoke:
	PYTHONPATH=src:. $(PYTHON) tools/smoke_local.py

clean:
	$(COMPOSE) down --remove-orphans
	@printf '%s\n' "Volumes are retained; destructive data reset stays with tools/reset_local.py (run with --inspect-only first)."
