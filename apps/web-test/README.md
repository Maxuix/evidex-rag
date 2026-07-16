# Local Frontend

The React frontend provides Documents, Chat, and Retrieval Debug views through
the public `/api/v1` API.

For direct frontend development:

```bash
npm ci
npm run dev
```

The only routine static checks are:

```bash
npm run typecheck
npm run build
```

Normal application use should start the complete stack with
`./start-local.sh` from the repository root.
