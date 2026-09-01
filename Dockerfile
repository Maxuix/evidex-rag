FROM docker.io/library/python:3.12.13-slim-bookworm@sha256:8a7e7cc04fd3e2bd787f7f24e22d5d119aa590d429b50c95dfe12b3abe52f48b AS python-dependencies

ARG RAG_KB_BUILD_DEBIAN_MIRROR=https://mirrors.tuna.tsinghua.edu.cn/debian
ARG RAG_KB_BUILD_DEBIAN_SECURITY_MIRROR=https://mirrors.tuna.tsinghua.edu.cn/debian-security
ARG RAG_KB_BUILD_PYPI_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=120 \
    PIP_RETRIES=10 \
    PYTHONPATH=/app/src:/app

WORKDIR /app

RUN sed -i \
        "s|URIs: http://deb.debian.org/debian$|URIs: ${RAG_KB_BUILD_DEBIAN_MIRROR}|" \
        /etc/apt/sources.list.d/debian.sources \
    && sed -i \
        "s|URIs: http://deb.debian.org/debian-security$|URIs: ${RAG_KB_BUILD_DEBIAN_SECURITY_MIRROR}|" \
        /etc/apt/sources.list.d/debian.sources \
    && grep -F "URIs: ${RAG_KB_BUILD_DEBIAN_MIRROR}" \
        /etc/apt/sources.list.d/debian.sources \
    && grep -F "URIs: ${RAG_KB_BUILD_DEBIAN_SECURITY_MIRROR}" \
        /etc/apt/sources.list.d/debian.sources \
    && apt-get update \
    && apt-get install --yes --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10001 rag-kb \
    && useradd --uid 10001 --gid rag-kb --no-create-home --shell /usr/sbin/nologin rag-kb

COPY requirements.lock /app/requirements.lock
RUN --mount=type=cache,id=rag-kb-pip-v1,target=/root/.cache/pip,sharing=locked \
    PIP_INDEX_URL="${RAG_KB_BUILD_PYPI_INDEX_URL}" \
    python -m pip install --require-hashes -r /app/requirements.lock

# Package installation is a build-time capability. The application never
# installs packages at runtime, so remove pip from the stage inherited by the
# final image. Verify the installed graph first, then fail the build if pip's
# module remains importable after removal.
RUN python -m pip check \
    && python -m pip uninstall --yes pip \
    && python -c "import importlib.util; assert importlib.util.find_spec('pip') is None"

FROM python-dependencies AS runtime

ARG RAG_KB_BUILD_REVISION=unknown
ARG RAG_KB_BUILD_HF_ENDPOINT=https://hf-mirror.com
LABEL org.opencontainers.image.revision="${RAG_KB_BUILD_REVISION}"

COPY config/docling-artifacts-v1.json /app/config/docling-artifacts-v1.json
COPY tools/artifact_download.py /app/tools/artifact_download.py
COPY tools/prepare_docling_artifacts.py /app/tools/prepare_docling_artifacts.py
COPY src/rag_kb/adapters/parser/docling/artifacts.py /app/tools/docling_artifacts_verifier.py
RUN --mount=type=bind,from=model-assets,source=/,target=/mnt/rag-kb-model-assets,ro \
    --mount=type=cache,id=rag-kb-build-models-v1,target=/var/cache/rag-kb-build-models,sharing=locked \
    if [ -d /mnt/rag-kb-model-assets/opt/rag-kb/docling-artifacts ] \
        && python /app/tools/prepare_docling_artifacts.py \
            --artifacts-path /mnt/rag-kb-model-assets/opt/rag-kb/docling-artifacts \
            --manifest /app/config/docling-artifacts-v1.json; then \
        mkdir -p /opt/rag-kb/docling-artifacts \
        && cp -a /mnt/rag-kb-model-assets/opt/rag-kb/docling-artifacts/. /opt/rag-kb/docling-artifacts/; \
    else \
        HF_ENDPOINT="${RAG_KB_BUILD_HF_ENDPOINT}" \
        python /app/tools/prepare_docling_artifacts.py \
            --download \
            --cache-path /var/cache/rag-kb-build-models \
            --artifacts-path /opt/rag-kb/docling-artifacts \
            --manifest /app/config/docling-artifacts-v1.json; \
    fi \
    && chmod -R a-w /opt/rag-kb/docling-artifacts

COPY config/local-reranker-artifacts-v1.json /app/config/local-reranker-artifacts-v1.json
COPY tools/prepare_local_reranker_artifacts.py /app/tools/prepare_local_reranker_artifacts.py
COPY src/rag_kb/adapters/local_reranker_artifacts.py /app/tools/local_reranker_artifacts_verifier.py
RUN --mount=type=bind,from=model-assets,source=/,target=/mnt/rag-kb-model-assets,ro \
    --mount=type=cache,id=rag-kb-build-models-v1,target=/var/cache/rag-kb-build-models,sharing=locked \
    if [ -d /mnt/rag-kb-model-assets/opt/rag-kb/local-reranker ] \
        && python /app/tools/prepare_local_reranker_artifacts.py \
            --artifacts-path /mnt/rag-kb-model-assets/opt/rag-kb/local-reranker \
            --manifest /app/config/local-reranker-artifacts-v1.json; then \
        mkdir -p /opt/rag-kb/local-reranker \
        && cp -a /mnt/rag-kb-model-assets/opt/rag-kb/local-reranker/. /opt/rag-kb/local-reranker/; \
    else \
        HF_ENDPOINT="${RAG_KB_BUILD_HF_ENDPOINT}" \
        python /app/tools/prepare_local_reranker_artifacts.py \
            --download \
            --cache-path /var/cache/rag-kb-build-models \
            --artifacts-path /opt/rag-kb/local-reranker \
            --manifest /app/config/local-reranker-artifacts-v1.json; \
    fi \
    && chmod -R a-w /opt/rag-kb/local-reranker

ENV HOME=/tmp/rag-kb-home \
    MPLCONFIGDIR=/tmp/rag-kb-home/.config/matplotlib \
    NUMBA_CACHE_DIR=/tmp/rag-kb-home/.cache/numba \
    HF_HOME=/var/lib/rag-kb/model-cache/huggingface

RUN mkdir -p "${MPLCONFIGDIR}" "${NUMBA_CACHE_DIR}" \
    && chown -R rag-kb:rag-kb "${HOME}"

COPY alembic.ini /app/alembic.ini
COPY apps /app/apps
COPY src /app/src
RUN chmod a+r /app/alembic.ini \
    && chmod -R a+rX /app/apps /app/src
RUN python -c "from rag_kb.tokenizer import preflight_tokenizer; preflight_tokenizer()"

USER 10001:10001

CMD ["python", "-m", "apps.api.main", "--container-listen"]
