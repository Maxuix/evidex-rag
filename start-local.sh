#!/bin/sh

set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
MANIFEST="$ROOT/.env.local"
REQUESTED_COMPOSE_PROJECT=${COMPOSE_PROJECT_NAME:-rag}
COMPOSE_PROJECT_NAME=rag
export COMPOSE_PROJECT_NAME
LOG_DIRECTORY="$ROOT/.runtime/logs"

fail() {
  printf 'Local startup failed: %s\n' "$1" >&2
  exit 1
}

manifest_value() {
  key=$1
  awk -v key="$key" '
    index($0, key "=") == 1 {
      print substr($0, length(key) + 2)
      exit
    }
  ' "$MANIFEST"
}

compose() {
  docker compose \
    --env-file "$MANIFEST" \
    --project-name "$COMPOSE_PROJECT_NAME" \
    "$@"
}

run_compose() {
  label=$1
  shift
  printf '%s\n' "$label"
  if ! compose "$@"; then
    printf 'Inspect logs with:\n  docker compose --env-file %s --project-name rag logs --no-color api worker migrate\n' "$MANIFEST" >&2
    exit 1
  fi
}

command -v git >/dev/null 2>&1 || fail "Git is not installed or not on PATH"
common_directory=$(
  git -C "$ROOT" rev-parse --path-format=absolute --git-common-dir 2>/dev/null
) || fail "cannot resolve the primary checkout"
case "$common_directory" in
  */.git) primary_checkout=$(dirname -- "$common_directory") ;;
  *) fail "Git common directory is not a primary checkout" ;;
esac
[ "$ROOT" = "$primary_checkout" ] ||
  fail "run this command from the primary checkout; linked worktrees are read-only for the personal runtime"

[ -z "${RAG_KB_LOCAL_COMPOSE_ENV_FILE:-}" ] ||
  fail "RAG_KB_LOCAL_COMPOSE_ENV_FILE is retired; use .env.local"
[ -z "${RAG_KB_LOCAL_APP_ENV_FILE:-}" ] ||
  fail "RAG_KB_LOCAL_APP_ENV_FILE is retired; use .env.local"
[ "$REQUESTED_COMPOSE_PROJECT" = "rag" ] ||
  fail "the personal Compose project is fixed to rag"

command -v python3 >/dev/null 2>&1 || fail "Python 3 is not installed or not on PATH"
command -v docker >/dev/null 2>&1 || fail "Docker is not installed or not on PATH"
docker info >/dev/null 2>&1 || fail "Docker is not running"

if ! python3 "$ROOT/tools/local_runtime.py" doctor; then
  fail "local runtime doctor found blocking drift"
fi

[ ! -L "$ROOT/.runtime" ] || fail "runtime directory must not be a symbolic link"
[ ! -L "$LOG_DIRECTORY" ] || fail "log directory must not be a symbolic link"
[ ! -e "$ROOT/.runtime" ] || [ -d "$ROOT/.runtime" ] ||
  fail "runtime path is not a directory"
[ ! -e "$LOG_DIRECTORY" ] || [ -d "$LOG_DIRECTORY" ] ||
  fail "log path is not a directory"
mkdir -p "$LOG_DIRECTORY"
chmod 700 "$ROOT/.runtime" "$LOG_DIRECTORY"

RAG_KB_BUILD_REVISION=$(git -C "$ROOT" rev-parse HEAD 2>/dev/null) ||
  fail "cannot resolve the build revision"
export RAG_KB_BUILD_REVISION

printf 'Using canonical Compose project rag and owner-only .env.local.\n'
printf 'Building Git revision %s.\n' "$RAG_KB_BUILD_REVISION"
run_compose "Starting PostgreSQL..." up -d --wait postgres
run_compose "Reconciling PostgreSQL roles..." \
  exec -T postgres /docker-entrypoint-initdb.d/10-init-runtime.sh
run_compose "Building application and frontend images..." build api frontend
run_compose "Preparing local source storage..." up storage-init
run_compose "Applying database migrations..." --profile tools run --rm migrate
run_compose "Starting API, Worker, and frontend..." \
  up -d --wait api worker frontend

compose ps

api_port=$(manifest_value RAG_KB_API_PORT)
frontend_port=$(manifest_value RAG_KB_FRONTEND_PORT)

printf '\nRAG KB is ready.\n'
printf 'User Chat: http://127.0.0.1:%s\n' "$frontend_port"
printf 'API docs: http://127.0.0.1:%s/api/v1/docs\n' "$api_port"
