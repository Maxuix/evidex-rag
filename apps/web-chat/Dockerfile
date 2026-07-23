FROM docker.io/library/node:24.18.0-bookworm-slim@sha256:6f7b03f7c2c8e2e784dcf9295400527b9b1270fd37b7e9a7285cf83b6951452d AS build

WORKDIR /web

COPY package.json package-lock.json ./
RUN npm ci --ignore-scripts --no-audit --no-fund

COPY index.html tsconfig.json vite.config.ts ./
COPY public ./public
COPY src ./src
RUN npm run build && rm -f dist/runtime-config.json

FROM docker.io/library/python:3.12.13-slim-bookworm@sha256:8a7e7cc04fd3e2bd787f7f24e22d5d119aa590d429b50c95dfe12b3abe52f48b

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN groupadd --gid 10001 rag-kb \
    && useradd --uid 10001 --gid rag-kb --no-create-home --shell /usr/sbin/nologin rag-kb

COPY server.py /app/server.py
COPY --from=build /web/dist /app/dist

USER 10001:10001

EXPOSE 3000

CMD ["python", "/app/server.py"]
