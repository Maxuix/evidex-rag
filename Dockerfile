FROM docker.io/library/python:3.12.13-slim-bookworm@sha256:8a7e7cc04fd3e2bd787f7f24e22d5d119aa590d429b50c95dfe12b3abe52f48b

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app/src:/app

WORKDIR /app

RUN groupadd --gid 10001 rag-kb \
    && useradd --uid 10001 --gid rag-kb --no-create-home --shell /usr/sbin/nologin rag-kb

COPY requirements.lock /app/requirements.lock
RUN python -m pip install --no-cache-dir --require-hashes -r /app/requirements.lock

COPY alembic.ini /app/alembic.ini
COPY apps /app/apps
COPY src /app/src

USER 10001:10001

CMD ["python", "-m", "apps.api.main", "--container-listen"]
