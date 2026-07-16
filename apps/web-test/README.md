# Test Frontend

This local Vite/React observation frontend implements the Stage 06 Documents,
Chat, and Retrieval Debug views using only the public `/api/v1` API. It contains
no login, identity/workspace switcher, arbitrary authorization header, provider
credential, direct model access, or private backend call.

## Local development

The frozen toolchain is Node `24.18.0` and npm `11.16.0`:

```bash
npm ci
npm ls --all
npm audit --package-lock-only
npm audit signatures
npm run typecheck
npm test
npm run build
```

Run those host commands only with the exact declared engines. The reproducible
path copies the read-only source into the pinned build container without writing
container-owned files into the checkout:

```bash
docker run --rm \
  --mount type=bind,src="$PWD/apps/web-test",dst=/src,readonly \
  docker.io/library/node:24.18.0-bookworm-slim@sha256:6f7b03f7c2c8e2e784dcf9295400527b9b1270fd37b7e9a7285cf83b6951452d \
  sh -c 'mkdir /tmp/web && cp /src/package.json /src/package-lock.json /src/index.html /src/tsconfig.json /src/vite.config.ts /tmp/web/ && cp -R /src/public /src/src /tmp/web/ && cd /tmp/web && npm ci --ignore-scripts --no-audit --no-fund && npm ls --all && npm audit --package-lock-only && npm audit signatures && npm run typecheck && npm test && npm run build'
```

`npm run dev` serves source for local UI iteration. It still requires
`/runtime-config.json`; the checked-in development value points to
`http://127.0.0.1:8000/api/v1`. The production build removes that development
file; the static server serves the canonical route dynamically from its
validated loopback-only command argument and exposes only compiled output.

Uploads are raw `.txt`/`.md` bodies. Because the public filename headers do not
define an encoding and browser headers are byte strings, this release rejects
non-printable-ASCII filenames and display names before sending a request.

See [the observation-boundary design](../../docs/architecture/test-frontend.md)
for endpoint use, state recovery, SSE fallback, security constraints, and known
limitations.
