FROM docker.io/library/node:24.18.0-bookworm-slim@sha256:6f7b03f7c2c8e2e784dcf9295400527b9b1270fd37b7e9a7285cf83b6951452d AS build

ARG RAG_KB_BUILD_NPM_REGISTRY=https://registry.npmmirror.com

WORKDIR /web

COPY package.json package-lock.json ./
RUN --mount=type=cache,id=rag-kb-npm-v1,target=/root/.npm,sharing=locked \
    npm ci \
        --ignore-scripts \
        --no-audit \
        --no-fund \
        --prefer-offline \
        --replace-registry-host=always \
        --registry="${RAG_KB_BUILD_NPM_REGISTRY}" \
        --fetch-retries=6 \
        --fetch-retry-factor=2 \
        --fetch-retry-mintimeout=2000 \
        --fetch-retry-maxtimeout=30000 \
        --fetch-timeout=120000

COPY index.html tsconfig.json vite.config.ts ./
COPY public ./public
COPY src ./src
RUN npm run build && rm -f dist/runtime-config.json

FROM docker.io/library/python:3.12.13-slim-bookworm@sha256:8a7e7cc04fd3e2bd787f7f24e22d5d119aa590d429b50c95dfe12b3abe52f48b AS runtime-base

ARG RAG_KB_BUILD_REVISION=unknown
LABEL org.opencontainers.image.revision="${RAG_KB_BUILD_REVISION}"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN groupadd --gid 10001 rag-kb \
    && useradd --uid 10001 --gid rag-kb --no-create-home --shell /usr/sbin/nologin rag-kb

COPY server.py /app/server.py

EXPOSE 3000

CMD ["python", "/app/server.py"]

FROM runtime-base AS runtime

COPY --from=build /web/dist /app/dist

# Host/build umasks must not make the compiled shell unreadable to the server.
RUN chmod -R a+rX /app
USER 10001:10001

FROM runtime-base AS prebuilt-runtime

COPY --from=frontend-dist / /app/dist

RUN chmod -R a+rX /app
USER 10001:10001
