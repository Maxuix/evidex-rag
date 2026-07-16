FROM docker.io/library/python:3.12.13-slim-bookworm@sha256:8a7e7cc04fd3e2bd787f7f24e22d5d119aa590d429b50c95dfe12b3abe52f48b

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN groupadd --gid 10001 rag-kb-e2e \
    && useradd --uid 10001 --gid rag-kb-e2e --no-create-home --shell /usr/sbin/nologin rag-kb-e2e

COPY model_provider_stub.py /app/model_provider_stub.py

USER 10001:10001

CMD ["python", "/app/model_provider_stub.py"]
