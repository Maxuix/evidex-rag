# Provider Capability Verification

The repository stores provider declarations, fingerprints, and non-sensitive reports only. Base URLs and API keys are supplied through environment variables during a manual smoke run and must not appear in declarations, reports, logs, Git history, or command output.

## Current state

`provider-declarations.template.json` defines the required chat, embedding, optional-rerank, and P1A embedding-space fields. It is intentionally incomplete until candidate endpoints are selected and tested.

The current declaration is `provider-declarations-deepseek-qwen-v1.0.json`, backed by the secret-free `verification-report-v1.0.json`. It selects DeepSeek `deepseek-v4-flash` for chat and Alibaba Cloud Model Studio Beijing `text-embedding-v4` at 1024 dimensions for embeddings. Any material change requires a new verified version and an updated implementation plan.

Validate the template without exposing environment values:

```bash
python3 tools/provider_fingerprint.py \
  verification/providers/provider-declarations.template.json \
  --allow-incomplete
```

## Secure local configuration

Place live values in `/tmp/rag-provider.env` with mode `0600`, or export them into the process environment. Do not create a repository `.env` file. Required variable names are:

- `RAG_CHAT_BASE_URL`
- `RAG_CHAT_API_KEY`
- `RAG_EMBEDDING_BASE_URL`
- `RAG_EMBEDDING_API_KEY`

Model identifiers and static non-sensitive capabilities belong in a copied declaration file under `verification/providers/`; secrets and raw internal URLs do not. A report may store a SHA-256 of normalized endpoint configuration but not the URL itself.

## Bounded real-endpoint verification

The manual verifier uses direct OpenAI-compatible HTTPS calls and writes only summarized evidence:

```bash
python3 tools/verify_provider_capabilities.py
```

The verifier requires a working TLS CA source. On the current macOS Python installation it uses the installed `certifi` bundle because the framework Python default CA path is empty; TLS verification remains enabled. Never use an insecure TLS flag. The script makes one model-list request, three short chat calls (one structured plus two concurrent), and one ten-item short embedding request.
