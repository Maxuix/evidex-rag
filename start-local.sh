#!/bin/sh

set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
STATE_FILE=${RAG_KB_LOCAL_COMPOSE_ENV_FILE:-"$ROOT/.env.local"}
APP_ENV_FILE=${RAG_KB_LOCAL_APP_ENV_FILE:-"$ROOT/.env"}

fail() {
  printf 'Local startup failed: %s\n' "$1" >&2
  exit 1
}

state_value() {
  key=$1
  awk -v key="$key" '
    index($0, key "=") == 1 {
      print substr($0, length(key) + 2)
      exit
    }
  ' "$STATE_FILE"
}

container_value() {
  key=$1
  printf '%s\n' "$CONTAINER_ENVIRONMENT" | awk -v key="$key" '
    index($0, key "=") == 1 {
      print substr($0, length(key) + 2)
      exit
    }
  '
}

validate_password() {
  label=$1
  value=$2
  case "$value" in
    ""|*[!A-Za-z0-9._~-]*)
      fail "$label must use only letters, digits, dot, underscore, tilde, or hyphen"
      ;;
  esac
}

validate_credentials() {
  validate_password "POSTGRES_ADMIN_PASSWORD" "$ADMIN_PASSWORD"
  validate_password "RAG_KB_MIGRATION_PASSWORD" "$MIGRATION_PASSWORD"
  validate_password "RAG_KB_RUNTIME_PASSWORD" "$RUNTIME_PASSWORD"
}

random_password() {
  if command -v openssl >/dev/null 2>&1; then
    openssl rand -hex 24
  else
    od -An -N24 -tx1 /dev/urandom | tr -d ' \n'
  fi
}

write_state_file() {
  state_directory=$(dirname -- "$STATE_FILE")
  [ -d "$state_directory" ] ||
    fail "state directory does not exist: $state_directory"
  temporary="$STATE_FILE.tmp.$$"
  umask 077
  trap 'rm -f "$temporary"' EXIT HUP INT TERM
  {
    printf 'RAG_KB_ENV_FILE=%s\n' "$APP_ENV_FILE"
    printf 'POSTGRES_ADMIN_PASSWORD=%s\n' "$ADMIN_PASSWORD"
    printf 'RAG_KB_MIGRATION_PASSWORD=%s\n' "$MIGRATION_PASSWORD"
    printf 'RAG_KB_RUNTIME_PASSWORD=%s\n' "$RUNTIME_PASSWORD"
  } >"$temporary"
  chmod 600 "$temporary"
  mv "$temporary" "$STATE_FILE"
  trap - EXIT HUP INT TERM
}

compose() {
  docker compose --env-file "$STATE_FILE" "$@"
}

run_compose() {
  label=$1
  shift
  printf '%s\n' "$label"
  if ! compose "$@"; then
    printf 'Inspect logs with:\n  docker compose --env-file %s logs --no-color api worker migrate\n' "$STATE_FILE" >&2
    exit 1
  fi
}

command -v docker >/dev/null 2>&1 || fail "Docker is not installed or not on PATH"
docker info >/dev/null 2>&1 || fail "Docker is not running"
[ -f "$APP_ENV_FILE" ] ||
  fail "missing $APP_ENV_FILE; copy .env.example to .env and configure the model providers"

credential_source=
if [ -f "$STATE_FILE" ]; then
  chmod 600 "$STATE_FILE"
  ADMIN_PASSWORD=$(state_value POSTGRES_ADMIN_PASSWORD)
  MIGRATION_PASSWORD=$(state_value RAG_KB_MIGRATION_PASSWORD)
  RUNTIME_PASSWORD=$(state_value RAG_KB_RUNTIME_PASSWORD)
  credential_source="saved local credentials"
else
  provided=0
  [ -n "${POSTGRES_ADMIN_PASSWORD:-}" ] && provided=$((provided + 1))
  [ -n "${RAG_KB_MIGRATION_PASSWORD:-}" ] && provided=$((provided + 1))
  [ -n "${RAG_KB_RUNTIME_PASSWORD:-}" ] && provided=$((provided + 1))
  if [ -n "${RAG_KB_LOCAL_DATABASE_PASSWORD:-}" ] && [ "$provided" -ne 0 ]; then
    fail "provide RAG_KB_LOCAL_DATABASE_PASSWORD alone, or provide all three role password variables"
  fi
  if [ "$provided" -ne 0 ] && [ "$provided" -ne 3 ]; then
    fail "provide all three database password variables together, or unset all three"
  fi

  if [ -n "${RAG_KB_LOCAL_DATABASE_PASSWORD:-}" ]; then
    ADMIN_PASSWORD=$RAG_KB_LOCAL_DATABASE_PASSWORD
    MIGRATION_PASSWORD=$RAG_KB_LOCAL_DATABASE_PASSWORD
    RUNTIME_PASSWORD=$RAG_KB_LOCAL_DATABASE_PASSWORD
    credential_source="current shell"
  elif [ "$provided" -eq 3 ]; then
    ADMIN_PASSWORD=$POSTGRES_ADMIN_PASSWORD
    MIGRATION_PASSWORD=$RAG_KB_MIGRATION_PASSWORD
    RUNTIME_PASSWORD=$RAG_KB_RUNTIME_PASSWORD
    credential_source="current shell"
  else
    container_id=$(docker compose ps -aq postgres 2>/dev/null || true)
    if [ -n "$container_id" ]; then
      CONTAINER_ENVIRONMENT=$(docker inspect --format '{{range .Config.Env}}{{println .}}{{end}}' "$container_id")
      ADMIN_PASSWORD=$(container_value POSTGRES_PASSWORD)
      MIGRATION_PASSWORD=$(container_value RAG_KB_MIGRATION_PASSWORD)
      RUNTIME_PASSWORD=$(container_value RAG_KB_RUNTIME_PASSWORD)
      credential_source="existing PostgreSQL container"
    else
      project_name=${COMPOSE_PROJECT_NAME:-$(basename "$ROOT" | tr '[:upper:]' '[:lower:]')}
      if docker volume inspect "${project_name}_postgres-data" >/dev/null 2>&1; then
        fail "an existing database volume has no recoverable credentials; export the original three passwords once and rerun"
      fi
      ADMIN_PASSWORD=$(random_password)
      MIGRATION_PASSWORD=$ADMIN_PASSWORD
      RUNTIME_PASSWORD=$ADMIN_PASSWORD
      credential_source="new random local credentials"
    fi
  fi
  validate_credentials
  write_state_file
fi

validate_credentials
unset POSTGRES_ADMIN_PASSWORD RAG_KB_MIGRATION_PASSWORD RAG_KB_RUNTIME_PASSWORD
unset RAG_KB_LOCAL_DATABASE_PASSWORD

printf 'Using %s from %s (mode 600; values are not printed).\n' "$credential_source" "$STATE_FILE"
run_compose "Starting PostgreSQL..." up -d --wait postgres
run_compose "Building the shared application and frontend images..." \
  build api frontend frontend-diagnostic
run_compose "Preparing local source storage..." up storage-init
run_compose "Applying database migrations..." --profile tools run --rm migrate
run_compose "Starting API, Worker, and frontends..." up -d --wait api worker frontend frontend-diagnostic

compose ps

api_port=${RAG_KB_API_PORT:-$(state_value RAG_KB_API_PORT)}
frontend_port=${RAG_KB_FRONTEND_PORT:-$(state_value RAG_KB_FRONTEND_PORT)}
diagnostic_frontend_port=${RAG_KB_DIAGNOSTIC_FRONTEND_PORT:-$(state_value RAG_KB_DIAGNOSTIC_FRONTEND_PORT)}
[ -n "$api_port" ] || api_port=8000
[ -n "$frontend_port" ] || frontend_port=3000
[ -n "$diagnostic_frontend_port" ] || diagnostic_frontend_port=3001

printf '\nRAG KB is ready.\n'
printf 'User Chat: http://127.0.0.1:%s\n' "$frontend_port"
printf 'Diagnostic UI: http://127.0.0.1:%s\n' "$diagnostic_frontend_port"
printf 'API docs: http://127.0.0.1:%s/api/v1/docs\n' "$api_port"
