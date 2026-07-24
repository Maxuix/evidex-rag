FROM docker.io/library/python:3.12.13-slim-bookworm@sha256:8a7e7cc04fd3e2bd787f7f24e22d5d119aa590d429b50c95dfe12b3abe52f48b

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app/src:/app

WORKDIR /app

RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libmagic1 \
        poppler-utils \
        tesseract-ocr \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10001 rag-kb \
    && useradd --uid 10001 --gid rag-kb --no-create-home --shell /usr/sbin/nologin rag-kb

COPY requirements.lock /app/requirements.lock
RUN python -m pip install --no-cache-dir --require-hashes -r /app/requirements.lock

COPY config/docling-artifacts-v1.json /app/config/docling-artifacts-v1.json
COPY tools/prepare_docling_artifacts.py /app/tools/prepare_docling_artifacts.py
COPY src/rag_kb/adapters/parser/docling/artifacts.py /app/tools/docling_artifacts_verifier.py
RUN python /app/tools/prepare_docling_artifacts.py \
        --download \
        --artifacts-path /opt/rag-kb/docling-artifacts \
        --manifest /app/config/docling-artifacts-v1.json \
    && chmod -R a-w /opt/rag-kb/docling-artifacts

ENV HOME=/tmp/rag-kb-home \
    MPLCONFIGDIR=/tmp/rag-kb-home/.config/matplotlib \
    NUMBA_CACHE_DIR=/tmp/rag-kb-home/.cache/numba \
    HF_HOME=/var/lib/rag-kb/model-cache/huggingface

RUN mkdir -p "${MPLCONFIGDIR}" "${NUMBA_CACHE_DIR}" \
    && chown -R rag-kb:rag-kb "${HOME}"

COPY alembic.ini /app/alembic.ini
COPY apps /app/apps
COPY src /app/src

USER 10001:10001

CMD ["python", "-m", "apps.api.main", "--container-listen"]
