# Enterprise Knowledge Base Architecture Design

Date: 2026-07-10
Revision date: 2026-07-13
Status: Revised after architecture review; approved for detailed P0/P1A implementation planning

## 1. Purpose

This document defines the initial architecture for a locally developed enterprise knowledge-base prototype built with retrieval-augmented generation (RAG) and a LangGraph-compatible workflow boundary. The first delivery must provide a small but complete vertical slice while preserving stable interfaces for a richer frontend, stronger parsing, authentication, external connectors, and a possible future department-scale deployment. The current project is **only for local development validation** and does not claim enterprise authentication, high availability, formal backup/recovery, compliance audit, or production multi-tenant isolation. LangGraph persistence is introduced only when measured workflow recovery or branching needs justify it.

The system is an internal document question-answering assistant. Documents are uploaded locally in the first release. All model access, including access to locally hosted models, goes through HTTP APIs.

## 2. Scope

### 2.1 Initial scope

- Deliver the initial work in three explicit milestones: `P1A usable closed loop`, `P1B delivery reliability`, and `P1C index lifecycle`. P1A is the implementation-planning target; P1B and P1C preserve independently testable paths for later hardening after the local flow is usable and measured.
- Create and manage knowledge bases.
- Upload local `.txt` and `.md` files through a temporary test parser.
- Index documents asynchronously with one worker and durable job state in P1A.
- Retrieve relevant index chunks with vector search.
- Answer with an evidence-only grounding policy and independently selectable response style and insufficiency behavior.
- Return a durable final answer and citations through polling or terminal-result SSE. P1A may optionally use **validated progressive delivery** to present an already committed answer in display-sized pieces; it is not provider-token streaming.
- Preserve session history without long-term user memory.
- Provide standardized, versioned backend APIs.
- Provide a small test frontend for document, chat, and retrieval inspection.
- Run locally with Docker Compose.
- Preserve configuration and interface boundaries for a possible later transition beyond local validation; such a transition requires a separate security and operations review.
- Include lightweight retrieval and answer evaluation.

### 2.2 Explicitly deferred capabilities

The following choices are intentionally deferred and do not block the initial framework:

- Production document parsing technology.
- OCR technology and image understanding.
- External document connectors.
- Long-term memory implementation.
- JWT/OIDC, user groups, and document ACLs.
- Production object storage provider.
- Dedicated vector database selection.
- Production hybrid search and reranking implementations.
- P1B delivery-reliability features: Outbox dispatch, multi-worker takeover, execution leases and epochs, optional LangGraph checkpoint recovery, and retained SSE replay.
- P1C index-lifecycle features: source snapshot/catch-up, full index-revision rebuild, atomic activation, rollback, and cleanup.

Each deferred capability has an interface or extension point defined in this document. Deferral is a design decision, not an unspecified requirement.

## 3. Architecture Decision

The system uses a modular monolith with separately deployable API and worker processes. This balances a small initial operational footprint with explicit module boundaries.

The initial foundation is:

- FastAPI
- Pydantic v2
- SQLAlchemy 2.x and Alembic
- PostgreSQL with pgvector
- A single asynchronous worker using a PostgreSQL job-table poller; P1A correctness relies on durable task state rather than Redis Pub/Sub, a replay stream, or a second queue system
- Local filesystem storage
- A direct application-service chat pipeline, with an optional minimal `GraphRunner` implementation behind a stable workflow interface
- Vite and React for the test frontend

API DTOs, domain models, and persistence models remain separate.

### 3.1 Verified compatibility and locking policy

P0 must verify and record one reproducible component set before schema or retrieval implementation begins. The target validation baseline is:

| Component | Supported line | Exact-lock requirement |
| --- | --- | --- |
| Python | 3.12 | Exact patch in `.python-version` and CI image |
| FastAPI | `>=0.135.0,<0.136` when native `fastapi.sse` is used | Exact patch in the Python lockfile |
| SQLAlchemy | `>=2.0,<2.1` | One integration-tested patch with no floating transitive dependencies |
| asyncpg | `>=0.30,<1.0` | Exact patch in the Python lockfile |
| PostgreSQL | 18 | Exact minor container tag and immutable image digest |
| pgvector | `>=0.8.0,<0.9` | Exact extension patch and immutable image digest; iterative scan must be capability-tested |

The lockfile and Compose image digests, rather than this table alone, are the release source of truth. `latest` tags are forbidden. Dependency upgrades use an explicit dependency-update change, execute Alembic migration tests and real PostgreSQL/pgvector integration tests, and run the retrieval regression suite before merge. P0 records the verified patch versions and digests in a checked-in compatibility manifest.

### 3.2 Asynchronous data-access policy

API handlers, application services, repositories, the Unit of Work, and the pgvector adapter use SQLAlchemy `AsyncSession` with `asyncpg`. A business service must not mix synchronous and asynchronous sessions, and an async route, pipeline step, or optional graph node must never call a synchronous repository or session directly.

Transactions are short-lived: read or persist the necessary database state, commit or roll back, then await the model, parser, queue, file store, or other external operation. CPU-intensive parsing or chunking runs in the worker process (or a separately configured process executor), never on the API event loop. This is the only supported initial data-access strategy; a future synchronous implementation would require a separately designed, thread-pool-isolated service path.

## 4. Architectural Boundaries

```text
API / Worker Entrypoints
          |
Application Services
          |
Business Capabilities -------------------- Optional Workflow Runner
  |          |          |                         |
Indexing  Retrieval  Answering             invokes capabilities
          |
Repositories + Unit of Work
          |
Database

Application Services also invoke external-system Adapters:
Model API, Vector Store, File Store, Parser, and Job Queue
```

### 4.1 Workflow runner and LangGraph

`GraphRunner` is a workflow interface, not a business capability. P1A's fixed linear chat flow is executed directly by an application service:

```text
retrieve -> assess -> generate/refuse -> validate -> persist
```

This direct pipeline is the default P1A implementation. It uses durable business run state and idempotent persistence, but does not require LangGraph, checkpoint storage, leases, epochs, or resume rules.

LangGraph is an optional implementation of `GraphRunner`; it becomes the default only when the workflow needs material conditional branching, human approval, multiple tools, or interruption and checkpoint recovery. Those production recovery capabilities are P1B work.

When enabled, a workflow runner is responsible for:

- Chat state transitions.
- Conditional routing.
- Node sequencing.
- Optional presentation orchestration.
- Checkpoint and recovery integration only when the P1B capability is enabled.

It is not responsible for:

- Parsing documents.
- Selecting chunking algorithms.
- Writing embeddings.
- Implementing retrieval algorithms.
- Implementing answer policies.
- Accessing SQL, vector stores, or file stores directly.

`IndexingPipeline` is driven directly by the worker and application service. Indexing may be wrapped in a workflow later without changing its business interface.

### 4.2 Application services

Services implement use cases and coordinate business capabilities, repositories, adapters, and workflows. They own operation ordering and cross-resource consistency decisions. They do not contain SQL or external SDK details.

### 4.3 Repositories

Repositories encapsulate relational persistence:

- CRUD operations.
- SQL queries.
- Pagination, filtering, and sorting.
- Persistence model mapping.
- State transitions stored in PostgreSQL.

Repository methods are asynchronous and receive the request- or command-scoped `AsyncSession` from the Unit of Work. They do not create sessions, call synchronous database APIs, or perform external I/O.

Repositories do not call model APIs, file storage, queues, or vector stores.

### 4.4 Unit of Work

The asynchronous Unit of Work owns PostgreSQL transaction boundaries, exposes repositories, and provides `commit` and `rollback` behavior. File storage, queues, and external vector stores are not falsely treated as participants in a relational transaction.

### 4.5 Adapters

Adapters encapsulate external protocols and clients:

- Chat, embedding, and rerank model APIs.
- Vector stores.
- File stores.
- Parsers.
- Job queues.

Ownership is semantic rather than physical. The pgvector adapter may use the same PostgreSQL instance as repositories, but vector retrieval remains behind the `VectorStore` contract.

## 5. Recommended Repository Structure

```text
RAG/
  apps/
    api/
      main.py
      dependencies.py
      routers/
        knowledge_bases.py
        documents.py
        chat.py
        retrieval.py
        indexing.py
        evals.py
        health.py
    worker/
      main.py
      tasks.py
    web-test/
      package.json
      src/

  src/
    rag_kb/
      config/
        settings.py
        profiles.py
      schemas/
        knowledge_bases.py
        documents.py
        chat.py
        retrieval.py
        indexing.py
        evals.py
        common.py
      domain/
        workspaces.py
        knowledge_bases.py
        documents.py
        indexing.py
        index_chunks.py
        citations.py
        conversations.py
        jobs.py
        policies.py
      services/
        knowledge_base_service.py
        document_service.py
        chat_service.py
        retrieval_service.py
        indexing_service.py
        eval_service.py
      workflows/
        runner.py                 # stable GraphRunner interface
        chat_pipeline.py          # P1A direct linear application workflow
        langgraph_runner.py       # optional; checkpoint/recovery features are P1B
      indexing/
        parser_contracts.py
        chunker.py
        metadata.py
        pipeline.py
      retrieval/
        retriever.py
        filters.py
        reranker.py
        evidence.py
        debug.py
      answering/
        policies.py
        policy_resolver.py
        evidence_assessor.py
        prompt_builder.py
        generator.py
        validators.py
      auth/
        context.py
        access_policy.py
      memory/
        conversation_memory.py
        long_term_memory.py
      repositories/
        workspaces.py
        knowledge_bases.py
        documents.py
        index_revisions.py
        indexed_document_versions.py
        index_chunks.py
        indexing_jobs.py
        outbox.py                 # P1B durable-delivery repository
        chat_sessions.py
        chat_runs.py
        chat_messages.py
        evals.py
      uow/
        unit_of_work.py
      db/
        models.py
        session.py
        migrations/
      adapters/
        event_stream/
          base.py                 # P1B retained/replayable transport adapter
          redis.py                # P1B implementation
        model_api/
          chat.py
          embeddings.py
          rerank.py
          openai_compatible.py
        vector_store/
          base.py
          pgvector.py
        file_store/
          base.py
          local.py
          s3.py
        parser/
          base.py
          plain_text_test.py
        job_queue/
          base.py
          postgres_poller.py      # P1A durable job-table poller
          redis.py                # P1B optional durable queue adapter
          inline.py
      observability/
        logging.py
        tracing.py

  evals/
    datasets/
    runners/
    reports/
  tests/
    unit/
    contract/
    integration/
    e2e/
  deploy/
    docker-compose.yml
    Dockerfile.api
    Dockerfile.worker
  docs/
    architecture/
    api/
    decisions/
  pyproject.toml
  README.md
  .env.example
```

P1A deliberately has one worker entrypoint rather than separate chat executor, recovery, and event-publisher processes. It claims durable PostgreSQL job rows through `postgres_poller.py`; Redis is not a P1A dependency. The workflow and event-stream files retain extension interfaces, but the P1A worker calls `chat_pipeline.py` directly and does not require checkpoint storage or retained event delivery. Placeholder modules such as `long_term_memory.py` and `s3.py` define contracts or explicit unsupported implementations. They must not silently simulate production behavior.

## 6. Core Data Model

### 6.1 Main entities

| Aggregate | Entities | Responsibility |
| --- | --- | --- |
| Workspace | `Workspace` | Future tenant isolation boundary; one default workspace initially |
| Knowledge base | `KnowledgeBase`, `SourceChange` | Document corpus, retrieval defaults, active-revision selector, and an ordered source-change ledger |
| Document | `Document`, `DocumentVersion` | Stable document identity and immutable source versions |
| Embedding | `EmbeddingSpace` | Immutable, physically compatible embedding definition: dimension, metric, data type, provider model, and model version |
| Index | `IndexRevision`, `IndexedDocumentVersion`, `IndexChunk` | Immutable configuration/source snapshot, per-revision document state, and retrievable content |
| Vector store | `VectorRecord` | Metadata and physical-vector reference for an index chunk, owned through the vector store adapter |
| Job | `IndexingJob`, `OutboxEvent` | Durable asynchronous processing; `OutboxEvent` is introduced for P1B reliable delivery |
| Conversation | `ChatSession`, `ChatRun`, `ChatMessage`, `Citation` | Session history, reproducible runs, answer status, and evidence links |
| Evaluation | `EvalDataset`, `EvalCase`, `EvalRun`, `EvalResult` | Reproducible evaluation records |

```text
Workspace
  |-- EmbeddingSpace
  `-- KnowledgeBase
        |-- SourceChange
        |-- Document
        |     `-- DocumentVersion
        |-- IndexRevision
        |     `-- references EmbeddingSpace
        |-- IndexedDocumentVersion
        |     |-- references DocumentVersion
        |     |-- references IndexRevision
        |     `-- IndexChunk
        |           `-- VectorRecord
        `-- ChatSession
              |-- ChatRun
              `-- ChatMessage
                    `-- Citation --> IndexChunk
```

The central cardinality is:

```text
DocumentVersion 1 -- N IndexedDocumentVersion N -- 1 IndexRevision
IndexedDocumentVersion 1 -- N IndexChunk
IndexChunk 1 -- 0..1 VectorRecord     # P1
```

### 6.2 Modeling rules

- `Document` is a stable business identity. An upload creates an immutable `DocumentVersion`, and `Document.current_version_id` identifies the latest source version visible to document management APIs.
- `DocumentVersion` owns source identity, checksum, storage location, and source availability only. It does not own parsing, chunking, embedding, or index readiness state.
- `EmbeddingSpace` is immutable and explicitly records `dimension`, `distance_metric`, `vector_data_type`, embedding provider/model identifier, requested model name, resolved deployment/model revision when the provider exposes one, endpoint logical identity, vector-normalization convention, and embedding/tokenizer configuration fingerprints. Its compatibility fingerprint is unique. An `IndexRevision` references exactly one `embedding_space_id`; it never infers compatibility from a logical revision namespace. If a provider exposes only a mutable model alias, the deployment configuration fingerprint is an operator-controlled release input and the alias must not be replaced without creating a new space.
- P1A provisions exactly one fixed `EmbeddingSpace` (float32 + cosine in the local deployment) and locks its model/version for the release. A model, dimension, metric, or vector-data-type change creates a new space and a new index revision; it is a P1C rebuild/migration, not an in-place setting change.
- `IndexRevision` stores immutable parser and chunking configuration, its `embedding_space_id`, and a frozen `source_snapshot_seq`. It is either the one active revision receiving ordinary incremental content or a separately built revision. A configuration change always creates a new revision.
- `SourceChange` is an immutable, per-knowledge-base ledger. Every document create, new version, availability change, and delete that becomes visible to the knowledge base commits with a strictly monotonic `source_change_seq`; provisional staging records do not become source changes until their source version is available and visible. The mutation transaction allocates the sequence atomically by incrementing `KnowledgeBase.source_change_seq` with `UPDATE ... RETURNING`, then inserts the ledger row with `UNIQUE (kb_id, source_change_seq)` in the same transaction. `MAX(seq) + 1` is forbidden. Each entry identifies the affected document, its resulting document version when applicable, and whether it is upserted or deleted. This ledger makes a snapshot and its catch-up set explicit, including deletes that have no new document version.
- `IndexedDocumentVersion` is the unique `(document_version_id, index_revision_id)` association and also stores its denormalized `document_id` and last applied `source_change_seq`. It owns per-revision processing state, serving state, error detail, and indexing job association.
- `IndexedDocumentVersion.build_status` uses `queued`, `processing`, `ready`, or `failed`. Its independent `serving_status` uses `candidate`, `serving`, or `retired`.
- `IndexingJob` targets one `IndexedDocumentVersion`; retries never change that target. P1A jobs store `claimed_by`, `claimed_at`, `heartbeat_at`, `attempt`, and `next_attempt_at`; the PostgreSQL poller uses these fields for bounded retry and stale-work reconciliation. Chunk/vector writes are idempotent upserts keyed by `(indexed_document_version_id, ordinal)` and the stable vector-record key; only a fully validated target can transition to `ready`.
- A new version can be prepared while the previous indexed version remains serving. Its promotion is a conditional transaction: the target must be `ready + candidate`, `Document.current_version_id` must still equal the target `document_version_id`, and no later upsert or delete `SourceChange` may exist for that document. Only then does the transaction retire the previous serving association and promote the new one for the same document and index revision. A target that has been superseded by a later version or delete is retired and cleaned up, never promoted. A delete transaction immediately retires/cancels all associations for that document in the active revision; physical vectors may be cleaned up later but are no longer eligible to serve.
- `IndexChunk` is a derived retrieval unit produced for exactly one `IndexedDocumentVersion`. It is not an original or revision-independent split of a source document.
- Index chunks are unique by `(indexed_document_version_id, ordinal)` in P1. A content hash may support diagnostics and future reuse but does not define cross-revision identity.
- P1 deliberately creates separate index chunks for separate index revisions even when their text happens to match. This duplicates some text during rebuilds but keeps revision activation, rollback, cleanup, and citation provenance unambiguous.
- `VectorRecord` is a separate vector-store record keyed to `index_chunk_id` and its `embedding_space_id`; an index chunk has at most one vector record. It is metadata/pointer state, not a second business truth for whether content serves: `IndexedDocumentVersion` remains the serving authority.
- Each `EmbeddingSpace` has an independent physical vector collection. In P1A, the one fixed pgvector table, its typed embedding column, and any indexes are created only by Alembic using a compile-time constant table name. The API and worker never construct a table name from request or database data and never execute runtime DDL. A later space receives a separately reviewed migration with the matching vector type, dimension, and operator class (`vector_cosine_ops`, `vector_l2_ops`, or `vector_ip_ops`). Different dimensions, metrics, data types, or model versions are never mixed in one table or ANN index.
- P1A starts with exact vector search. HNSW is enabled only when measured exact-search latency misses the agreed target and the candidate HNSW configuration passes unfiltered and filtered Recall@k gates on representative data. HNSW build parameters and query recall settings are recorded as deployment configuration, not confused with an embedding-space identity.
- Index chunk text, location, hierarchy, and source metadata remain in PostgreSQL even if vectors move to another store.
- Stable `index_chunk_id`, `indexed_document_version_id`, and `index_revision_id` values link relational metadata and vector records.
- Citations reference the retrieved `IndexChunk` while it is retained and always store document version, quoted text, and source location snapshots. Retired index cleanup may set the chunk reference to null without breaking historical citations.
- `ChatRun` stores run status, requested and effective answer policies, model configuration references, index revision, timing, usage, error state, retry count, idempotency key and request hash. P1A additionally records `claimed_by`, `claimed_at`, `heartbeat_at`, and `next_attempt_at` for its PostgreSQL worker claim protocol. P1B adds execution-lease fields and a monotonically increasing `execution_epoch`; `workflow_thread_id` is added only when optional checkpoint recovery is enabled. It references the associated user and assistant messages.
- Chat run status uses `queued`, `running`, `completed`, `failed`, or `cancelled`. A non-terminal `running` attempt may return to `queued` only through P1A's bounded automatic retry transaction, which increments its attempt count and applies `next_attempt_at`; a terminal failed run returns to `queued` only through an explicit retry operation.
- `(principal_id, client_id, endpoint, idempotency_key)` is unique. Reusing the key with the same request hash returns the existing run; reusing it with a different hash fails with `IDEMPOTENCY_KEY_REUSED`.
- When P1B checkpoint recovery is enabled, `workflow_thread_id` equals `chat_run_id`; an optional latest checkpoint identifier may be stored for diagnostics, but it is not product state.
- User messages are unique by the client request id within a session. Assistant messages are unique by `chat_run_id`, and citations are unique by `(assistant_message_id, ordinal)`.
- Assistant messages use `generating`, `completed`, or `failed` status.
- Document versions use source lifecycle states `available`, `unavailable`, or `deleted`; index processing state never appears on `DocumentVersion`.
- Index revisions use `building`, `ready`, `active`, `retired`, or `failed` status.
- Database constraints enforce the index-selection invariants through executable partial unique indexes and same-KB composite foreign keys, as specified below. Before activation, every non-deleted source document at the final source sequence must have exactly one `ready + candidate` association in the building revision. The activation transaction promotes those candidates to serving, retires the old revision's serving associations, moves the selector, and then commits the invariant of exactly one ready serving association in the new active revision.

Answer behavior is represented by a versioned, immutable `EffectiveAnswerPolicy` rather than a single mode. The requested policy, effective policy, retrieval strategy, model configuration, and index revision are recorded on each chat run.

### 6.3 Executable PostgreSQL constraints and roles

Alembic migrations, not ORM-only declarations or runtime startup code, create the schema. Partial uniqueness is expressed with `CREATE UNIQUE INDEX ... WHERE ...` (and SQLAlchemy `Index(..., unique=True, postgresql_where=...)`), never as a table-level `UniqueConstraint`:

```sql
CREATE UNIQUE INDEX uq_one_active_revision_per_kb
    ON index_revision (kb_id)
    WHERE status = 'active';

CREATE UNIQUE INDEX uq_one_serving_version_per_document_revision
    ON indexed_document_version (document_id, index_revision_id)
    WHERE build_status = 'ready' AND serving_status = 'serving';

ALTER TABLE indexed_document_version
    ADD CONSTRAINT ck_serving_requires_ready
    CHECK (serving_status <> 'serving' OR build_status = 'ready');

ALTER TABLE source_change
    ADD CONSTRAINT uq_source_change_sequence UNIQUE (kb_id, source_change_seq);

ALTER TABLE index_revision
    ADD CONSTRAINT uq_index_revision_kb_id_id UNIQUE (kb_id, id);

ALTER TABLE knowledge_base
    ADD CONSTRAINT fk_active_revision_same_kb
    FOREIGN KEY (id, active_index_revision_id)
    REFERENCES index_revision (kb_id, id);
```

Sequence allocation and ledger insertion occur in one transaction:

```sql
UPDATE knowledge_base
   SET source_change_seq = source_change_seq + 1
 WHERE id = :kb_id
 RETURNING source_change_seq;
-- insert SourceChange(kb_id, returned source_change_seq, ...) before commit
```

The composite foreign key prevents a selector from referencing another knowledge base. A deferred constraint trigger is used only for the remaining cross-row status rule: at commit, a provisioned knowledge base's selector must reference its sole `active` revision. P0 must produce and execute the real Alembic migration against PostgreSQL, including upgrade and downgrade tests.

The database uses separate roles. The migration role owns DDL and schema changes; the runtime API/worker role receives only the required DML and sequence permissions. Runtime startup performs read-only capability validation for the expected schema, pgvector extension/version, vector dimension/type, and operator support, and fails fast on mismatch without attempting repair or provisioning.

## 7. Consistency Model

PostgreSQL, file storage, Redis, and a potential external vector store cannot share one transaction. The system therefore uses explicit eventual consistency.

- Database state transitions are transactional through the Unit of Work.
- P1A commits an `IndexedDocumentVersion` and its `IndexingJob` together; its single worker polls PostgreSQL durable task state with finite retries. In one short transaction, the poller claims eligible queued work with `FOR UPDATE SKIP LOCKED` (or an equivalent conditional update), sets `claimed_by`, `claimed_at`, and `heartbeat_at`, and increments `attempt`. It does not require an outbox, at-least-once event log, or replay transport.
- P1B commits an `IndexedDocumentVersion`, `IndexingJob`, and `OutboxEvent` together when reliable delivery is enabled. Its outbox dispatcher publishes queued work with independent idempotency and retries undelivered events.
- P1B commits `ChatRunRequested` with the queued run and `AnswerCommitted` or `RunFailed` with terminal business state. Its queue consumers and retained-event publishers are at-least-once and deduplicate by stable run-scoped keys.
- Workers use `job_id` and `indexed_document_version_id` as idempotency boundaries.
- New index chunks and vector records are written under the target `IndexedDocumentVersion` and remain non-serving until validation succeeds.
- The PostgreSQL retrieval read path compiles selector resolution, authorization/serving filters, distance ordering, and the limit into **one SQL statement**. The statement joins through `KnowledgeBase.active_index_revision_id` and admits only `IndexedDocumentVersion(build_status='ready', serving_status='serving')`; under `READ COMMITTED`, that single statement has one consistent snapshot. The implementation must not first fetch the selector and then issue another query under the default isolation level. If an adapter limitation makes multiple SQL statements unavoidable, that use case explicitly begins `REPEATABLE READ READ ONLY` before the first read and keeps all statements inside that short transaction; this isolation level is not a global default. An external vector adapter receives the same revision and serving filters as mandatory fields; an already-written vector is never sufficient to make a record retrievable.
- P1A permits only ordinary incremental writes to the current active revision. It does not offer an online full rebuild: while an administrator has put a knowledge base into rebuild preparation, document mutations are either rejected with `KB_REBUILD_IN_PROGRESS` or durably queued, according to the configured mode. They must not be silently applied to both or neither revision. P1A's normal upload/update/delete transaction serializes the document change, `SourceChange`, and target association/job creation; the later serving transition is a separate conditional atomic transaction that verifies the target remains the `Document.current_version_id` and no later document-level source change exists. Retries can repeat a completed phase without creating additional serving rows or promoting a superseded version.
- P1C adds complete revision rebuilding. It captures `source_snapshot_seq` from the committed source-change ledger, builds the exact document state at that sequence, and applies later ledger entries in sequence order. Immediately before cutover it serializes new source mutations for that knowledge base, catches up through the current `source_change_seq`, and verifies no sequence is missing, duplicated, or failed. This final gate is released only after the new selector is committed; subsequent changes target the newly active revision.
- Revision activation is a single PostgreSQL transaction: lock the knowledge-base selector and final source sequence; verify the new revision is complete and has one ready **candidate** association for every document non-deleted at the final applied sequence; promote exactly those candidates to serving; retire the prior revision and its serving associations; update `KnowledgeBase.active_index_revision_id`; and mark the new revision `active`. The deferred selector check and partial unique constraints make two active revisions or two serving versions of one document impossible at commit. Physical/external vectors are already present but remain unreachable through the mandatory selector and serving filters until that commit.
- Old index content remains until successful activation and asynchronous cleanup.
- Compensation jobs remove orphaned vectors or files.
- Failed jobs preserve phase, error code, diagnostic detail, and retry count. Retried indexing jobs reuse their original target and stable write keys; they do not promote a target unless all expected chunks and vectors validate, and a superseded/deleted target becomes a no-op followed by cleanup.

## 8. Indexing Flow

```text
Upload
  -> validate and checksum
  -> write source bytes to a non-serving staging path
  -> create an unavailable DocumentVersion that references the staged/final storage identity
  -> atomically rename the staged file to its final local path
  -> mark the source version available, update Document.current_version_id, and append SourceChange(source_change_seq)
  -> create IndexedDocumentVersion and IndexingJob for the target IndexRevision
  -> P1B: create OutboxEvent and dispatch through its dispatcher
  -> P1A: let the single worker claim the durable queued job
  -> parse into normalized ParsedDocument
  -> create IndexChunkDraft objects
  -> request embeddings in batches
  -> write IndexChunk metadata and VectorRecords
  -> mark IndexedDocumentVersion ready
  -> atomically switch its serving status when the target revision is active
  -> clean superseded index content asynchronously
```

For P1A the target is always the knowledge base's active revision selected in the same transaction that records the source change. An update creates a new immutable `DocumentVersion` and indexes it as a candidate; the previous ready serving version remains readable until the new candidate validates. The per-document serving switch is conditional: it succeeds only if the candidate is still `Document.current_version_id` and no later document-level `SourceChange` exists; otherwise the candidate is retired as superseded. A delete appends a delete `SourceChange`, immediately retires the document's serving/candidate associations in the active revision, and schedules vector cleanup; a later upload is a new version and follows the ordinary candidate flow. Repeated upload requests and worker retries reuse their idempotency/job and association keys, so they cannot leave duplicate chunks, vectors, or serving versions.

Local file storage uses a staged-file protocol rather than pretending it shares a transaction with PostgreSQL. Docker Compose defines one named source volume mounted into both API and worker at the same container path. The staging and final prefixes must be directories on that same mounted filesystem; startup rejects a configuration in which their device/filesystem identities differ, because atomic rename is not guaranteed across filesystems. Bytes are first written beneath the non-serving staging prefix and the provisional `DocumentVersion` remains `unavailable`. After an atomic local rename to the final path, one database transaction makes the version visible, appends its atomically allocated `SourceChange`, and creates its indexing target/job. A periodic P1A janitor removes aged staging/final files without a valid database reference and marks a database record unavailable when its expected file is missing. Delete first makes the source and index content non-serving in PostgreSQL, then retries physical file cleanup from a durable cleanup record. API/worker restart tests verify that committed files remain readable through the shared volume. Thus a crash can create an auditable cleanup item, but never a serving document whose source is not available.

The local volume is persistence for development convenience, not a backup. README must state that host failure recovery, RPO/RTO, and formal backup are not provided and that important test corpora must be reproducible from checked-in samples or import scripts. P1A provides an idempotent development cleanup command for aged staging/orphan files, retired vectors/chunks, and expired task records, plus a clearly destructive full local reset command. Formal retention, legal hold, and compliance deletion remain deferred.

P1C rebuilding does not write new source changes directly into a building revision. It materializes the frozen `source_snapshot_seq`, replays subsequent `SourceChange` entries in order, and uses the final cutover gate described in Section 7. Thus updates and deletes that occur during a rebuild are represented once in the candidate revision, while P1A's explicit reject-or-queue restriction avoids claiming that this online protocol already exists.

The parser contract is:

```python
Parser.parse(source: DocumentSource) -> ParsedDocument
```

`ParsedDocument` is parser-neutral and represents text blocks, page references, heading hierarchy, and optional table or image placeholders. `ParsedDocument` and `IndexChunkDraft` are transient P1 pipeline values, not persistent source entities. The initial `PlainTextTestParser` supports `.txt` and `.md` only. Other formats fail explicitly with `PARSER_NOT_CONFIGURED`. This adapter validates the pipeline without selecting the production parser stack.

If measured parsing cost later justifies cross-revision reuse, a separately reviewed `ParseArtifact` and `ChunkSet` cache can be keyed by parser and chunking profile hashes. That optimization is not part of P1 and does not change the `IndexedDocumentVersion` ownership model.

Before embedding, the pipeline validates the configured embedding output against the target revision's `EmbeddingSpace` (dimension, metric, data type, model identifier, deployment/configuration fingerprint, and model version). The `VectorStore` contract resolves the fixed P1A space to the migration-created physical table and rejects an incompatible write. At startup it verifies, without DDL, the installed pgvector version, expected table/schema, required vector type/dimension capacity, and matching distance operator support; incompatible deployment configuration fails before ingestion begins. A revision identifier is retrieval metadata, not a vector namespace. Changing an embedding model creates a new embedding space and index revision, then requires the P1C rebuild/cutover protocol and a reviewed migration for its physical vector storage.

## 9. Retrieval and Answering

`RetrievalService` accepts a retrieval request and returns an `EvidencePack`. It owns vector or hybrid retrieval, access filters, optional reranking, evidence normalization, and debug information. It never produces a final answer.

### 9.1 Corpus profile and retrieval baselines

P0 creates a versioned `CorpusProfile` before the embedding model, tokenizer, chunk sizes, or ANN settings are accepted. The profile records:

- primary and secondary languages and their measured proportions;
- representative document length/token distributions and structural features;
- exact-identifier density (error codes, product models, article numbers, acronyms, names, and mixed alphanumeric terms);
- Unicode normalization and case/punctuation handling rules;
- update frequency and representative revision/status filter selectivity;
- local-test sensitivity classification and confirmation that samples are public, synthetic, or desensitized.

Until representative samples are measured, the explicit planning assumption is a primarily Simplified-Chinese corpus with common English technical terms and medium-to-high identifier density. That assumption is provisional and is not sufficient to select Chinese tokenization. The measured profile and its sample manifest are checked in with the evaluation dataset; a material profile change requires retrieval regression evaluation.

P1A's product default is exact vector retrieval. Evaluation must also run a lexical baseline against the same cases, with dedicated exact-identifier and terminology tags. PostgreSQL `tsvector/tsquery` is acceptable for English or other languages where its configured parser is demonstrated to work. For Chinese or mixed-language content, the team must compare candidate tokenizers/BM25 implementations on the real sample instead of assuming PostgreSQL's default parser is adequate. The lexical baseline may remain evaluation-only in P1A; hybrid serving is selected in a future retrieval milestone (currently P2) only if measured misses justify it.

Answer behavior is composed from orthogonal policy dimensions:

```text
EffectiveAnswerPolicy
  |-- GroundingPolicy
  |-- InsufficiencyPolicy
  |-- AnswerStyle
  |-- CitationPolicy
  `-- AnswerTask
```

The dimensions have distinct responsibilities:

- `GroundingPolicy` defines the allowed knowledge boundary. Defined values are `evidence_only`, `evidence_preferred`, and `model_knowledge_allowed`.
- `InsufficiencyPolicy` defines behavior when evidence does not cover the request. Defined values are `refuse`, `partial_answer`, and `ask_for_clarification`.
- `AnswerStyle` controls expression only. Defined values are `concise`, `summary`, `standard`, and `detailed`.
- `CitationPolicy` defines whether citations are required and their `claim_level`, `paragraph_level`, or `source_list` granularity.
- `AnswerTask` defines algorithmic intent. Defined extension values include `answer`, `extract`, `compare`, `translate`, and `report`.

`compare`, `extract`, and `report` are tasks rather than styles because they may require query decomposition, structured output, different retrieval, or a dedicated workflow. `model_knowledge_allowed` is used instead of the ambiguous term `open_book`.

P1 supports this policy surface:

```text
grounding_policy     = evidence_only             # server enforced
answer_style         = concise | summary         # request override allowed
insufficiency_policy = refuse | partial_answer   # request override allowed
citation.required    = true                      # server enforced
citation.granularity = claim_level               # server enforced
answer_task          = answer                    # P1 fixed
```

Under `partial_answer`, the generator answers only evidence-supported portions and explicitly identifies missing portions. When no usable evidence exists, `partial_answer` resolves to deterministic refusal rather than using model knowledge.

`PolicyResolver` deterministically combines policies in this precedence order:

```text
server-enforced constraints
  > permitted request overrides
  > KnowledgeBase defaults
```

Server constraints fix values or allowed sets. A request may override a permitted dimension within that set; an omitted dimension falls back to the KnowledgeBase default. `PolicyResolver` validates supported combinations, prevents clients from weakening grounding or citation requirements, and produces an immutable `EffectiveAnswerPolicy` with a policy version. Orthogonal dimensions do not imply that every Cartesian product is valid.

The answer contract is:

```text
AnswerRequest + EvidencePack + EffectiveAnswerPolicy -> AnswerResult
```

The P1 generator returns a schema-constrained internal result before presentation rendering:

```json
{
  "outcome": "answered | partial | refused",
  "claims": [
    {"text": "a material, evidence-bounded claim", "citation_ids": ["cite_1"]}
  ],
  "missing_aspects": ["aspects for which the EvidencePack has no usable support"]
}
```

`PromptBuilder` composes compatible grounding, insufficiency, style, task, and citation instructions without creating one class per combination. `EvidenceStructureValidator` validates the result schema and outcome constraints (`refused` contains no claims; `partial` identifies missing aspects), permits citation identifiers only from the current `EvidencePack`, rejects duplicate or unlocatable citation records, and requires every material claim in an `answered` or `partial` result to carry at least one permitted citation. The renderer produces the user-visible prose and citations only from this validated structure; it does not append unstructured substantive prose. This is structural association and coverage validation, not a runtime proof that a cited passage semantically entails a claim. Semantic support is measured by offline and human evaluation.

Citation requirements apply to substantive answers. Deterministic refusal and clarification responses are control outcomes and do not need fabricated citations.

If runtime evidence-structure validation fails, the graph permits one bounded repair attempt using the same evidence and effective policy. A second failure produces a deterministic safe refusal stating that a structurally valid answer could not be produced and records the validation failure for evaluation; it does not release an unsupported answer.

Vector retrieval is the initial product default. Each vector request is compiled into a single `RetrievalQueryPlan` containing the active revision, access and serving-status filters, `top_k`, ANN candidate count, `ef_search`, iterative-scan mode, and the retrieval strategy. The PostgreSQL adapter renders the plan as the one-statement snapshot query required by Section 7. Filters are part of the query plan rather than post-processing: a row must satisfy all authorization, workspace, knowledge-base, revision, and serving-status predicates before it may enter the `EvidencePack`.

For HNSW, the adapter chooses `ef_search` and candidate over-fetching (`candidate_count = top_k * oversampling`) from configuration and measured filter selectivity. It enables bounded iterative scans when the first approximate scan yields fewer than `top_k` authorized candidates; exhausting the configured scan/candidate budget returns fewer results, never unauthorized results. Low-cardinality, stable filters such as serving status or a small set of knowledge bases may use validated partial indexes or partitions. High-isolation workspace or ACL boundaries require physical partitions, tables, or independent collections once enabled; they must not rely solely on a broad shared ANN index plus late filtering. The retrieval-debug record exposes the selected plan and result count to authorized operators without exposing inaccessible metadata.

Exact search is the P1A default. HNSW may be introduced only after exact-search latency measurements establish a need and the ANN plan passes both unfiltered and filtered Recall@k acceptance thresholds; when enabled, oversampling limits, `ef_search`, iterative-scan limits, and any partial-index/partition choices are deployment configuration backed by benchmark results and are not silently changed per request. Lexical, hybrid retrieval, reranking, filters, and retrieval debugging are present in contracts and request schemas. Unsupported serving strategies return explicit capability errors rather than silently falling back, except where a configured rerank failure policy permits degradation to the original vector order.

## 10. Chat Pipeline and optional GraphRunner

P1A does not require a LangGraph graph. `ChatService` creates or reuses the durable business run and returns without executing the answer flow. One independent worker claims the queued run and invokes `ChatPipelineService` directly. The pipeline has a fixed linear path and calls application-service commands for every business write.

```text
load_context -> retrieve_evidence -> assess_evidence
  -> refuse_response OR answer_with_policy
  -> validate_evidence_structure -> persist_result
```

`ChatService` runs the deterministic `PolicyResolver` before creating the durable run, so unsupported policy requests fail synchronously and the pipeline receives an already frozen effective policy. `EvidenceAssessment` distinguishes sufficient coverage, partial coverage, no usable evidence, and query ambiguity. P1A routes `refuse` to a deterministic refusal response and routes supported combinations through `answer_with_policy`. Specialized task nodes or subgraphs are added only when an `AnswerTask` changes retrieval, validation, safety boundaries, or the underlying algorithm.

The P1A command context includes:

- Request and `AuthContext`.
- Session and message identifiers.
- Chat run identifier.
- Normalized or rewritten query.
- Retrieval request and `EvidencePack`.
- Requested policy and immutable effective answer policy.
- Answer draft and citations.
- Provider usage and timing.
- Trace identifiers and normalized error state.

Business capabilities remain directly callable outside the graph for tests, evaluation, worker tasks, and retrieval debug APIs.

### 10.1 Facts and ownership

The business database is the sole product fact source for:

- Sessions and user-visible messages.
- Chat run status and final answer.
- Citations, errors, model usage, and timing.
- Durable audit records and, when enabled in P1B, outbox records.

P1A has no checkpoint store and no second execution state machine. Its recovery boundary is the durable `ChatRun` state plus idempotent final persistence. A bounded automatic retry either completes the same run or records a terminal failure; after exhaustion, an operator or explicit retry API creates the next attempt according to the run state machine.

When P1B enables LangGraph persistence, checkpoints become the execution recovery source for:

- The current graph step and pending work.
- Intermediate node outputs.
- Retry, resume, and temporary graph context.

Conversation history is always loaded from the business database by `load_context`; it is never reconstructed from checkpoint history. Product APIs and repositories never query checkpoint tables for messages or final run status. If enabled, the checkpointer may share the PostgreSQL instance, but it uses separately owned tables or schema and is accessed only through LangGraph.

If P1B checkpoint state contains retrieved evidence or answer drafts, it follows the same encryption, access-control, redaction, and data-classification requirements as business content, with a shorter retention policy where recovery needs permit.

P1A terminal SSE is a live delivery convenience, not a durable event log. The authoritative result is always `GET /chat/runs/{id}`. P1B may add Redis Streams as a replayable transport log; events remain observations rather than product facts and their retention may be shorter than ChatRun retention.

When P1B enables retained events, their payload, persistence, transport encryption, access controls, log redaction, and retention follow the same content-security classification as the business database.

### 10.2 Execution and side effects

```text
POST transaction
  -> look up idempotency key and compare the canonical request hash
  -> return the existing run immediately when the key and hash match
  -> resolve and validate EffectiveAnswerPolicy
  -> create ChatRun in queued state
  -> create or reuse user message
  -> create or reuse assistant placeholder by chat_run_id
  -> commit
  -> return 202 with run_id, status_url, and events_url

P1A single worker
  -> poll eligible queued ChatRuns from PostgreSQL with `FOR UPDATE SKIP LOCKED`
  -> atomically claim ChatRun queued -> running, record worker identity/attempt/heartbeat
  -> commit

ChatPipelineService execution
  -> retrieve, assess, answer or refuse, validate
  -> external calls run outside database transactions

persist_result command
  -> upsert assistant message by chat_run_id
  -> upsert citations by assistant_message_id and ordinal
  -> store usage, timing, and final error state
  -> set ChatRun completed
  -> commit one business transaction

terminal delivery
  -> a connected SSE request sends answer.completed or run.failed
  -> otherwise the client reads the committed result from the status URL
```

P1A runs one worker process with two explicit execution lanes: `chat` and `indexing`. Each lane has its own semaphore and queue accounting; the default local configuration reserves at least one chat slot and one indexing slot. The poller uses weighted fair selection (default weight `chat:indexing = 3:1`) plus configurable aging so a new chat can start while indexing is busy and an old indexing job cannot starve indefinitely. Lane concurrency, provider concurrency, and process-executor concurrency are separate limits rather than one unbounded task pool.

The PostgreSQL poller claims `IndexingJob` and `ChatRun` rows with a conditional transition/`FOR UPDATE SKIP LOCKED`, preventing a second accidentally started worker from normally executing the same row. Every poll/claim, heartbeat, reconciliation, and state transition uses its own short-lived `AsyncSession`; heartbeat never reuses the session performing task work, and an `AsyncSession` is never shared by concurrent tasks. No database transaction remains open across parser or provider I/O.

Each provider defines connect, read, and total-call timeouts; each indexing/chat task has an absolute deadline covering all attempts in that execution. Cancellation or deadline expiry transitions through an explicit compare-and-set operation to retryable queued state or terminal `failed`, according to the finite attempt policy. The worker refreshes `heartbeat_at` between bounded external operations. A row is stale only after a configured timeout greater than the longest permitted operation; reconciliation requeues it with a later `next_attempt_at` and exponential backoff, or records terminal failure after the finite attempt limit. The database pool budget must cover API status checks/SSE polling, both worker lanes, pollers, and independent heartbeat sessions with a documented safety margin; startup rejects a configured concurrency total that exceeds that budget. Idempotent business commands remain mandatory because process failure or a task retry can occur at any boundary. P1A does not promise transparent multi-executor takeover.

The API never performs model execution. Disconnecting an SSE client removes only that subscriber; the independent pipeline continues. P1A exposes status polling as the recovery path for a disconnected client.

`persist_result` is a single-purpose, idempotent application-service command. Re-executing it produces the same assistant message, citations, terminal run state, and usage record. Business writes use unique constraints, upsert, or compare-and-set transitions. Provider calls are at-least-once external operations: their stable run/operation key and provider request identifier are recorded when available, and genuinely repeated billable calls remain visible for reconciliation.

If the pipeline raises before final persistence, the worker idempotently records `ChatRun=failed`, marks the assistant placeholder failed, and stores the normalized error in one business transaction. A failed pipeline cannot rely on a later workflow step to persist its failure.

P1A buffers the model draft, performs evidence-structure validation plus the bounded repair policy, and commits only the validated final answer. It never publishes raw provider tokens. An optional display optimization may divide that already committed answer into `answer.validated_delta` events; this is called **validated progressive delivery**, and its first-byte metric is the first validated delta, not the model's first token.

P1B adds delivery-reliability mechanisms without changing this public run model: transactional Outbox dispatch, multiple executors with lease/epoch fencing, and idempotent retained-event publication. Checkpointed `GraphRunner` resume is enabled only if interruption recovery or material workflow branching is actually required; reliable queueing by itself does not require LangGraph. The full `execution_epoch` and checkpoint rules below apply only when that optional recovery capability is enabled.

### 10.3 Optional P1B checkpoint-recovery conflict rules

P1B permits multiple runners. The business row then stores `lease_owner`, `lease_expires_at`, and `execution_epoch`; acquisition and renewal use compare-and-set. A takeover after lease expiry increments the epoch, and finalize/fail commands match the current epoch so a stale runner cannot overwrite the current owner. `execution_epoch` is invocation runtime context, not checkpointed graph state.

The P1B executor invokes the LangGraph implementation with `thread_id=chat_run_id`; provider calls and non-deterministic operations are isolated in checkpointed tasks or single-purpose nodes. The execution contract follows LangGraph's documented [checkpoint and replay semantics](https://docs.langchain.com/oss/python/langgraph/persistence) and [idempotent task guidance](https://docs.langchain.com/oss/python/langgraph/functional-api). The business database remains the product fact source and wins whenever it differs from checkpoint state:

```text
Business DB completed + checkpoint incomplete
  -> return the committed result; do not resume; checkpoint may be collected

Business DB running + checkpoint available
  -> resume the same chat_run_id and workflow thread

Business DB running + checkpoint missing or corrupt
  -> restart with the same chat_run_id; idempotent writes prevent duplicates

Checkpoint complete + Business DB not completed
  -> retry persist_result from checkpointed output; do not report completion yet

Business DB terminal failed + checkpoint resumable
  -> do not resume until an explicit retry transitions the run to queued
```

Checkpoint retention may be shorter than business message retention. Collection is allowed only after a run reaches a committed terminal business state and the recovery retention period expires.

## 11. API Design

All core endpoints are versioned under `/api/v1`. OpenAPI is generated from Pydantic request and response models.

### 11.1 Core endpoints

```text
POST   /api/v1/knowledge-bases
GET    /api/v1/knowledge-bases
GET    /api/v1/knowledge-bases/{kb_id}
PATCH  /api/v1/knowledge-bases/{kb_id}

POST   /api/v1/knowledge-bases/{kb_id}/documents
GET    /api/v1/knowledge-bases/{kb_id}/documents
GET    /api/v1/documents/{document_id}
POST   /api/v1/documents/{document_id}/versions
DELETE /api/v1/documents/{document_id}
POST   /api/v1/documents/{document_id}/reindex  # P1C full-rebuild capability; P1A returns explicit capability error

GET    /api/v1/indexing-jobs/{job_id}
POST   /api/v1/indexing-jobs/{job_id}/retry

POST   /api/v1/retrieval/query

GET    /api/v1/chat/sessions
POST   /api/v1/chat/sessions
GET    /api/v1/chat/sessions/{session_id}/messages
POST   /api/v1/chat/runs
GET    /api/v1/chat/runs/{run_id}
GET    /api/v1/chat/runs/{run_id}/events

POST   /api/v1/eval/runs
GET    /api/v1/eval/runs/{run_id}
```

Uploads and ChatRun creation return `202 Accepted` in the local development profile. Reindex requests return `202` only when the P1C rebuild capability is enabled; P1A either omits that endpoint from its OpenAPI surface or returns `409 CAPABILITY_NOT_ENABLED` without creating a job. Document operation responses include `document_id`, `document_version_id`, `indexed_document_version_id`, `index_revision_id`, `job_id`, and job status when applicable. List endpoints use cursor pagination with `limit`, `cursor`, and `sort` parameters. Mutating operations support `Idempotency-Key`, and it is required for ChatRun creation.

### 11.2 Chat responses and streaming

`POST /api/v1/chat/runs` creates a durable run and never streams or waits for model execution. The response returns before an independent executor processes the run:

```text
POST /api/v1/chat/runs
Idempotency-Key: <required UUID>

-> 202 Accepted
Location: /api/v1/chat/runs/{run_id}
```

The P1 request includes the run inputs and only permitted policy overrides:

```json
{
  "session_id": "session_...",
  "knowledge_base_id": "kb_...",
  "message": "User question",
  "answer_policy": {
    "answer_style": "concise",
    "insufficiency_policy": "refuse"
  },
  "retrieval": {
    "mode": "vector",
    "top_k": 8
  }
}
```

The `202 Accepted` response body is:

```json
{
  "run_id": "run_...",
  "status": "queued",
  "status_url": "/api/v1/chat/runs/run_...",
  "events_url": "/api/v1/chat/runs/run_.../events",
  "effective_answer_policy": {
    "grounding_policy": "evidence_only",
    "answer_style": "concise",
    "insufficiency_policy": "refuse",
    "citation_required": true,
    "citation_granularity": "claim_level",
    "answer_task": "answer",
    "policy_version": "p1"
  }
}
```

ChatRun, the creation response, subsequent status responses, and the terminal event record the full `effective_answer_policy`, including enforced grounding, citation, task, and policy version fields. Unsupported values or combinations return `ANSWER_POLICY_NOT_SUPPORTED` before a run is created; omitting request overrides uses KnowledgeBase defaults.

Repeating POST with the same idempotency key and request hash returns the existing `run_id` and current resource representation. Reusing a key with a different request hash returns `409 IDEMPOTENCY_KEY_REUSED`. This is how a client recovers the run identifier when the original creation response is lost.

`GET /api/v1/chat/runs/{run_id}/events` is the optional live-delivery endpoint. In P1A it uses `text/event-stream` and browser `EventSource`, but it is not a token stream or an event-replay API. A client may instead poll `GET /api/v1/chat/runs/{run_id}`; that resource is authoritative in every milestone.

The P1A endpoint can be opened immediately after creation. It waits for the committed terminal result and sends SSE keepalive comments without creating product events. The generator never holds an `AsyncSession`, transaction, or database connection across a wait; each status check opens a new short-lived session and releases it immediately. Polling uses a configurable jittered interval (local default 1 second with ±20% jitter), detects client disconnects, and enforces a maximum connection duration (local default 10 minutes). Per-principal and per-run connection limits (local default two concurrent streams per principal/run pair) protect the API pool. On timeout or limit rejection, the client continues through the authoritative status URL. Reconnecting clients read the run resource again; P1A does not accept `Last-Event-ID`, retain events, allocate replay sequences, or emit `stream.reset`.

P1A SSE event types are:

```text
answer.validated_delta  # optional presentation-only event
answer.completed
run.failed
```

`answer.completed` contains the final message identifier, full validated answer, final citations, and effective answer policy. `run.failed` is derived from the committed failure state. Candidate evidence and retrieval details remain in the status or authorized debug API rather than product events.

If a P1A frontend benefits from a more animated presentation, the server may send `answer.validated_delta` pieces only after the complete validated answer has been committed. This optional **validated progressive delivery** changes neither the durability model nor the status resource and must not be described or measured as raw-token streaming. `answer.completed` remains the terminal event.

P1B introduces the retained-event protocol while retaining this endpoint and terminal event shapes. It uses Redis Streams with explicit retention, stable event identifiers and sequence allocation, and supports `Last-Event-ID`. A valid cursor replays only later events; an expired cursor receives `stream.reset` with the status URL, after which the client reloads the authoritative ChatRun. Consumer-group delivery, acknowledgement timing, pending-message reclaim, exponential-backoff scheduling, maximum attempts, and a DLQ are specified with the P1B queue implementation; Redis Pub/Sub is never the reliable queue.

After receiving a terminal `answer.completed` or `run.failed` event, the client closes its `EventSource`, and the server also ends that response. Disconnecting earlier only removes the subscriber; the independent run continues. P1B replay improves delivery but never makes Redis the product fact source. P1A uses FastAPI's native SSE support and therefore pins FastAPI `>=0.135.0`; its ping behavior and default `Cache-Control: no-cache` and `X-Accel-Buffering: no` handling are covered by integration tests. Selecting another SSE library requires an explicit dependency record and equivalent proxy/header tests.

### 11.3 Errors

Errors use a Problem Details-compatible body with:

- `type`
- `title`
- `status`
- `detail`
- `instance`
- Stable domain `code`
- `trace_id`
- `retryable`
- Optional field validation errors

Provider and SDK errors are normalized and never expose credentials or internal exception details.

Deleting a document is a soft delete at the API transaction boundary. It is excluded from retrieval immediately, while file and vector cleanup runs asynchronously.

### 11.4 Protocol separation

OpenAI-compatible APIs are used inside `ModelApiAdapter`. The core knowledge base API does not imitate OpenAI chat endpoints because it must expose retrieval, indexing, citations, and evaluation semantics. A future `/compat/openai/v1/chat/completions` facade may translate to `ChatService` without becoming the core contract.

## 12. Model Provider Design

Chat, embedding, and rerank providers are configured independently. Cloud and locally hosted models use the same HTTP adapter model.

Each provider supports:

- `base_url`
- `api_key`
- `model`
- `timeout`
- `max_retries`
- `max_concurrency`

An OpenAI-compatible wire shape is not treated as a capability guarantee. Each adapter also supplies a validated static declaration or startup discovery result containing `supports_structured_output`, supported schema mode, `max_input_tokens`, `embedding_dimension`, `max_batch_size`, `supports_usage`, `supports_streaming`, `retryable_statuses`, rate-limit semantics, and provider idempotency support. Startup fails when a required capability conflicts with the selected pipeline. P1A answer generation prefers provider-native schema output; when unavailable, it requests ordinary JSON, validates it with Pydantic, and applies the same bounded repair/refusal policy. Retryable failures use exponential backoff with random jitter and obey the task's total deadline.

Provider-specific details remain inside adapters. Secrets are loaded from environment variables or a secret manager and are excluded from database records, logs, traces, and API responses. Non-sensitive parser/chunking configuration is stored on `IndexRevision` for reproducibility; embedding compatibility is stored immutably on its referenced `EmbeddingSpace` (`dimension`, metric, vector data type, provider endpoint identity, requested/resolved model identifier, deployment revision when available, normalization convention, and configuration fingerprint). A provider that cannot report a resolved deployment revision is treated as a controlled deployment dependency: changing its embedding alias or deployment configuration requires recording a new fingerprint and creating a new space rather than silently reusing vectors.

Changing a chat model or endpoint is a configuration operation only when the declared capabilities still satisfy the pipeline contract. P1A pins the one embedding space and rejects any embedding-model change. Changing an embedding model, model version, dimension, metric, or vector data type is an index migration: add its isolated vector table/index through Alembic, create a new `EmbeddingSpace`, create an `IndexRevision` referencing it, and use the P1C full rebuild/cutover protocol before it can serve.

## 13. Authentication and Authorization Boundary

The first release has no login UI or external identity provider. It still enforces the following call chain:

```text
API -> AuthContext -> AccessPolicy -> MetadataFilter -> Retrieval
```

- Every application service requires an `AuthContext`; `None` is invalid.
- `SingleWorkspaceAccessPolicy` limits the initial system to the server-configured default workspace.
- Clients cannot select an arbitrary workspace through a trusted header.
- Repository queries include `workspace_id`.
- Retrieval always receives filters produced by `AccessPolicy`.
- ChatRun status and event endpoints authorize the run through `AuthContext`; possession of a run identifier is not authorization.
- JWT/OIDC and document ACLs later replace identity and policy implementations without changing service signatures.

`DevelopmentAuthProvider` is the only identity source in P0/P1A. It is allowed only when `DEPLOYMENT_PROFILE=development`; it derives one server-configured fixed principal and `client_id`, never a request-selected identity or workspace. The server binds to `127.0.0.1` by default. The local prototype must not be exposed as a shared, internet-facing, staging, or production deployment.

P0/P1A do not implement OIDC, JWT, groups, ACLs, or trusted-proxy authentication. Any non-development profile is disabled and fails startup until a separately reviewed identity integration constructs `AuthContext` from verified data. The `AuthContext -> AccessPolicy -> MetadataFilter` boundary is retained for that future work, but it is not presented as an enterprise security guarantee today.

Idempotency records for every mutating endpoint are scoped by `(principal_id, client_id, endpoint, idempotency_key)`, together with the canonical request hash. This prevents one caller from recovering another caller's run or mutation merely by reusing a key. The persistence and API uniqueness constraints must use this scope rather than workspace scope alone.

The P1 frontend has no login. When authentication is added, native browser `EventSource` can use same-site secure cookies, while bearer-header deployments use a fetch-based SSE client. Long-lived bearer tokens are never placed in the events URL.

An `AuditSink` contract receives key security and data lifecycle events. Initial delivery may emit structured audit logs; durable audit storage belongs to department-scale hardening.

## 14. Deployment Profiles

### 14.1 Local development profile

Docker Compose runs:

- One FastAPI instance.
- One worker process hosting indexing jobs and the direct P1A chat pipeline.
- PostgreSQL with pgvector.
- The PostgreSQL job-table poller; P1A does not require Redis, Redis Streams, or a replayable ChatEventStream.
- Local persistent file storage.
- The test frontend.

Model services remain external HTTP APIs. The inline job adapter is restricted to automated tests and local debugging; it is not a formal lite deployment mode. The worker's durable PostgreSQL job/run records, finite retries, heartbeat, and single-process claim rule are required in every P1A deployment.

API and worker mount the same named source volume at the same container path; staging and final paths are located on that volume. Compose publishes application ports to loopback only. README carries a prominent statement that this profile lacks enterprise identity, authorization, audit, backup/recovery, high availability, and multi-tenant isolation guarantees.

### 14.2 Future department profile (not currently deployable)

The department profile supports:

- Load-balanced stateless API replicas.
- P1B separate chat execution, event publishing, indexing, maintenance, and evaluation worker pools as justified by load.
- High-availability or managed PostgreSQL and Redis.
- S3-compatible object storage.
- A model API gateway.
- Central logs, metrics, traces, and alerts.

pgvector remains the default until measured retrieval latency, indexing time, write throughput, or maintenance constraints justify a dedicated vector store. No fixed document-count threshold is encoded in the architecture.

### 14.3 Configuration rules

`DEPLOYMENT_PROFILE=development` is the only enabled profile in P0/P1A. `department` is a future design placeholder and fails startup in the current implementation. Business modules never branch on the profile name; they receive configured interfaces through dependency injection.

Configuration is grouped into app, database, P1A job poller, P1B queue/event stream, file store, vector store, model provider, retrieval, workflow runner, and observability settings. The P1A profile rejects configuration that enables multi-runner recovery, a second queue system, or retained-event replay without the corresponding P1B components.

Database migrations run as an explicit command with the migration role, never from API/worker startup and never concurrently from API replicas. Runtime roles have no DDL permission.

## 15. Observability

Logs, metrics, and traces correlate through `trace_id`, `run_id`, pipeline step and attempt, `job_id`, `session_id`, `document_id`, `document_version_id`, `indexed_document_version_id`, and `index_revision_id`. P1B adds `execution_epoch`, `workflow_thread_id`, `checkpoint_id`, and graph-node labels when its recovery runner is enabled.

Default telemetry records metadata rather than document text, full questions, prompts, or answers. Controlled sampling may enable content diagnostics in an approved environment.

Initial metrics include:

- API latency, error rate, ChatRun queue time, and time to committed final answer; when the optional presentation mode is enabled, time to first validated progressive delta.
- P1B only: event publish lag, reconnect replay count, expired cursor count, and `stream.reset` count.
- Queue depth, job duration, retries, and failures.
- Parse, chunk, and embedding duration.
- Retrieval latency, empty-result rate, and Recall@k.
- Rerank degradation count.
- Answer latency, token usage, citation coverage, and provider failures.
- Answer metrics are segmented by effective grounding, style, insufficiency, citation, and task policy values.

The code exposes OpenTelemetry-compatible tracing hooks. A complete observability stack is optional in local development and belongs to a separately reviewed future department deployment.

## 16. Evaluation

Evaluation datasets use version-controlled YAML or JSONL. Each case contains a question, expected documents or passages, an optional reference answer, and tags.

Each dataset version references the measured `CorpusProfile` and includes exact-identifier/terminology, Chinese, English, mixed-language, no-answer, update, delete, revision-filter, and malicious-document cases. The same cases run against exact vector retrieval and the selected lexical baseline; ANN is compared against exact vector results before it can become a serving option.

Initial metrics are:

- Retrieval Recall@k, MRR, and empty-result rate, reported both overall and separately for representative access/revision/status filters and filter-selectivity bands.
- Citation identifier validity and structural claim coverage.
- Offline semantic-support and unsupported-claim rates under `evidence_only`, based on labeled evaluation cases and periodic human review; they are not claimed by runtime structural validation.
- Correct refusal and partial-answer behavior for insufficient evidence.
- Latency, token usage, and failure rate.

Filtered retrieval cases record the effective `RetrievalQueryPlan` parameters (including `top_k`, oversampling, `ef_search`, and iterative-scan outcome) so Recall@k regressions caused by ANN filtering are distinguishable from corpus or model changes. Filtered Recall@k is an explicit HNSW enablement gate, not merely a diagnostic metric. Reports segment exact-identifier and terminology results by vector and lexical strategy. LLM-as-Judge is optional, goes through `ModelApiAdapter`, and is never the sole acceptance criterion. Each `EvalRun` records corpus profile, dataset, index, model, retrieval strategy, and every effective answer policy dimension and version.

## 17. Testing Strategy

- Unit tests cover domain rules, state machines, answer policies, filters, direct pipeline routing, and error mapping. Optional GraphRunner implementations use the same application-service command tests.
- Policy tests cover precedence, unsupported combinations, client attempts to weaken enforced constraints, all four P1 style/insufficiency combinations, and the no-evidence fallback from partial answer to refusal.
- Contract tests cover parser, model API, vector store, file store, and PostgreSQL job-poller contracts.
- Integration tests use real PostgreSQL/pgvector and the PostgreSQL poller with deterministic HTTP model stubs. P1B queue tests use real Redis or the selected equivalent only when that capability is enabled.
- P0 database tests execute the real Alembic upgrade/downgrade, verify the two partial unique indexes and same-KB composite foreign key, and use concurrent transactions for two revision activations, two uploads allocating `SourceChange`, and version completion in reverse order.
- Retrieval consistency tests race revision activation against reads. Every result set must belong completely to the old revision or completely to the new revision; a mixed or partially filtered snapshot fails the test. Both the one-statement path and any explicitly supported `REPEATABLE READ READ ONLY` fallback are covered.
- End-to-end tests cover upload, completed indexing, retrieval, answer generation, and citation validation.
- P1A failure tests inject failures before and after `persist_result`, exercise finite task retry, heartbeat-based stale-work reconciliation, staged-file cleanup, and verify that assistant messages, citations, and usage are not duplicated. They include the ordering case “V1 upload -> V2 upload -> V2 indexing completes -> V1 indexing completes” and verify V1 cannot become serving.
- P1B concurrency tests start competing runners, verify lease takeover after expiry, and reject finalization from a stale execution epoch. When optional checkpoint recovery is enabled, recovery tests resume from checkpoints and verify that assistant messages, citations, outbox events, and usage are not duplicated.
- Creation tests lose the first POST response, repeat the same idempotency key, and recover the same run identifier; request-hash mismatch returns conflict.
- Disconnect tests verify that closing SSE does not cancel the business run and that a reconnecting P1A client can recover through the status endpoint.
- SSE resource tests verify that a waiting stream holds no database session/transaction, respects jittered polling, connection/deadline limits and disconnect detection, and cannot exhaust the configured API pool.
- Worker scheduling/load tests run a batch of indexing work while creating chats, verify chat starts within the configured queue-time target, verify aging eventually schedules indexing, and assert independent sessions/heartbeats and task deadlines.
- P1B replay tests reconnect with `Last-Event-ID`, receive only later events, and use `stream.reset` when the cursor has expired. P1B conflict tests cover every business-state/checkpoint-state combination in section 10.3 and verify that API reads always follow the business database.
- P1A delivery tests verify terminal events follow business commits and optional validated progressive pieces contain only committed answer text. P1B publisher tests additionally verify retry does not duplicate retained chunks.
- API regression tests snapshot the OpenAPI contract and Problem Details shapes.
- Test frontend coverage focuses on upload, job status, terminal-result SSE or status polling, and retrieval debugging.

Automated tests do not call real model providers. Separate manual smoke tests verify configured provider access.

## 18. Failure Handling and Security

- Parser failure marks the job failed and preserves the processing phase.
- Partial embedding failure does not activate partial index content.
- Rerank failure may fall back to original retrieval order only when the configured degradation policy allows it.
- A chat provider failure that occurs after `POST /chat/runs` has returned is recorded as a terminal `ChatRun=failed` with stable code such as `MODEL_PROVIDER_UNAVAILABLE` and `retryable=true`; the status resource exposes that business error and SSE emits `run.failed`. It never fabricates an answer. HTTP `503` is reserved for synchronous requests that fail before a run is created, such as an unavailable required dependency or invalid provider configuration.
- A P1A pipeline failure before final persistence is recorded by the worker; `persist_result` is not assumed to run after an upstream failure. The same rule applies to an optional P1B `GraphRunner`.
- P1A task retries are tolerated through run-scoped idempotency constraints and compare-and-set state transitions; P1B additionally tolerates duplicate graph execution under its lease/epoch rules.
- Maximum retry exhaustion produces a terminal failed state requiring explicit retry.
- P1 accepts only `.txt` and `.md` after extension and content validation. A file is limited to 10 MiB, UTF-8 (optionally with BOM) text, 200,000 lines, and 20,000 produced chunks; it is rejected rather than partially indexed when any limit is exceeded.
- Parsing and chunking run in a resource-isolated subprocess with a 30-second wall-clock timeout, 20 CPU-second budget, and 512 MiB memory limit. That parser subprocess has no model credentials and no network access; its parent worker may access only the explicitly configured model and storage endpoints needed by the indexing pipeline. Timeout, decoding, or resource-limit failures preserve a diagnosable failed job state.
- MIME type, filename, and checksum are admission signals, not a security boundary. Future binary formats enter a separate isolated scan and conversion pipeline before any parser adapter; archive expansion, malware scanning, macro handling, and parser resource limits are decided there rather than inherited from the text parser.
- CORS uses an explicit origin allowlist and does not permit wildcard credentialed access.
- Parsers and retrieved document content are treated as untrusted input.
- Retrieved content is passed to the model as minimal, provenance-labelled, untrusted evidence excerpts with clear data boundaries. Retrieved text is never interpreted as application control data. Prompt separation reduces but cannot eliminate indirect prompt-injection risk and is not an authorization boundary.
- Prompts state that instructions inside retrieved documents are data, not executable instructions. The initial chat pipeline gives the model no external tool execution capability, credentials, or authority to change retrieval/access filters. Service-side access filtering and citation-ID validation remain authoritative. P1A tests malicious documents that request system-prompt disclosure, instruct the model to ignore citation rules, fabricate citation IDs, or induce access to another workspace; no such case may bypass service-side policy, although model-text robustness is reported as an evaluation result rather than an absolute guarantee.
- `/health/live` checks process health; `/health/ready` checks PostgreSQL and the configured queue. Temporary model-provider failure is exposed through metrics and run errors rather than removing every API replica from service.

## 19. Test Frontend

The frontend is a backend observation tool with three views:

1. Documents: upload, version status, indexing status, failure detail, and retry.
2. Chat: knowledge base selection, concise/summary style, refuse/partial-answer behavior, effective policy display, durable run creation, terminal SSE or status polling, citations, run status, and timing. P1A reconnects by reading the status resource; P1B adds EventSource replay.
3. Retrieval debug: retrieved index chunks, scores, metadata, filters, and rerank ordering.

It has no login, ACL management, operations dashboard, prompt editor, or private backend access. Every frontend capability uses the public `/api/v1` contract.

The frontend follows the deployment identity baseline. In P0/P1A development it is available on loopback and operates as the server-configured development principal. It has no identity/workspace switcher, arbitrary authorization-header input, embedded privileged credentials, or direct model-provider access. A future non-development frontend requires the separately reviewed browser authentication mechanism; document, chat, event, and retrieval-debug views must remain subject to the same `AuthContext`, `AccessPolicy`, and metadata filters as every other client.

## 20. Delivery Phases

### P0: Foundation

- Modular monolith structure.
- FastAPI, configuration, and dependency injection.
- Domain, repository, adapter, and Unit of Work contracts.
- Checked-in compatibility manifest and dependency/image locks for the verified Python, FastAPI, SQLAlchemy, asyncpg, PostgreSQL, and pgvector combination; separate migration/runtime database roles.
- Executable Alembic schema with partial unique indexes, the same-KB composite foreign key, atomic `SourceChange` sequence allocation, and startup read-only capability checks for the fixed `EmbeddingSpace`. Redis remains optional infrastructure reserved for P1B durable delivery.
- One-statement PostgreSQL retrieval snapshot design, with an explicitly tested `REPEATABLE READ READ ONLY` fallback only if an adapter requires multiple statements.
- Versioned `CorpusProfile`, provider capability declarations, exact-vector/lexical evaluation baselines, and representative desensitized sample manifest.
- Docker Compose with loopback-only publishing and one API/worker shared source volume whose staging/final paths use the same filesystem.
- Problem Details, structured logging, health endpoints, and initial OpenAPI.

### P1A: Usable closed loop

- Knowledge base, document, document version, fixed `EmbeddingSpace`, index revision, indexed document version, index chunk, indexing job, and durable ChatRun persistence.
- Local file store with staged-file finalization and idempotent cleanup/reset commands, plus one worker using fair chat/indexing lanes, independent concurrency slots, short-lived sessions, independent heartbeat, hard deadlines, bounded retries, and idempotent upload/chat commands. Redis Pub/Sub, a second queue system, and replayable event transport are not P1A dependencies.
- Parser contract and `PlainTextTestParser`.
- Chunking, embedding, exact pgvector retrieval, and an evaluation lexical baseline; HNSW requires measured latency need and filtered Recall@k approval.
- Orthogonal answer policy model, deterministic policy resolver, concise/summary styles, refuse/partial-answer behavior, and fixed P1 grounding and citation constraints.
- Direct `ChatPipelineService` execution (`retrieve -> assess -> generate/refuse -> evidence-structure validate -> persist`), idempotent final persistence, structural citation validation, and database-backed session history. `GraphRunner` remains an interface and an optional minimal implementation, not a P1A requirement.
- Pollable ChatRun status plus resource-bounded terminal-result SSE using short database sessions, jittered polling, disconnect detection, connection limits, and a maximum duration; optional validated progressive delivery only after final-answer commit. No raw-token delivery, `Last-Event-ID`, retained replay, `evidence.ready`, or `stream.reset` in P1A.
- A version-controlled small golden dataset and lightweight runner for exact/filtered Recall@k, vector-versus-lexical identifier recall, citation validity, refusal behavior, malicious-document behavior, latency, and failure rate.
- Documents, chat, and retrieval debug frontend views.
- End-to-end upload-to-cited-answer test.

### P1B: Delivery reliability

- Transactional Outbox and independently idempotent dispatcher.
- Redis Streams or an equivalent durable queue with consumer groups, explicit acknowledgement timing, pending-message reclaim, exponential-backoff scheduling, maximum attempts, and a DLQ.
- Multiple workers with lease/epoch fencing and failure-injection coverage.
- Retained SSE events with stable IDs, `Last-Event-ID` replay, and `stream.reset` fallback; terminal status remains authoritative.
- Optional checkpointed LangGraph `GraphRunner` recovery only when an accepted workflow/recovery requirement cannot be met by the direct pipeline and durable business state.

### P1C: Index lifecycle

- Full index-revision rebuild from a frozen source sequence.
- Ordered `SourceChange` catch-up and final cutover gate.
- Atomic activation, rollback, retired-revision cleanup, and recovery workflows.
- Migration-created physical vector storage for any new `EmbeddingSpace`.

### P2: Initial engineering expansion

- Separate production parser selection and architecture decision.
- Expanded evaluation datasets, regression thresholds, reports, and optional LLM-as-Judge.
- Production hybrid retrieval and reranking.
- OpenTelemetry integration.
- Expanded API and provider contract coverage.
- Expanded binary-format security controls and prompt-injection hardening beyond the P1A text-parser baseline.

### P3: Department-scale expansion

- S3-compatible storage.
- Horizontally scaled API and worker deployments.
- Queue separation and resource limits.
- JWT/OIDC, groups, document ACLs, and durable audit storage.
- Monitoring, alerting, backup, and recovery.
- Dedicated vector store if operational metrics justify it.
- External content connectors.

The first implementation plan covers P0 and P1A only. P1B delivery reliability and P1C index lifecycle are separately scheduled and independently accepted; P2 and P3 require separate decisions and implementation plans.

## 21. Milestone Acceptance Criteria

### P1A: Usable closed loop

- Docker Compose starts the API, PostgreSQL/pgvector, one PostgreSQL-polling worker, local file storage, and test frontend. Redis is not required for P1A.
- A client can create a knowledge base through the public API.
- A client can upload a `.txt` or `.md` file, observe the indexing job complete, and see the target indexed document version become ready and serving.
- Upload limits, decoding failures, and parser resource-limit failures produce a durable, diagnosable failed job without partially serving the document.
- The verified compatibility manifest and lockfiles contain exact dependency patches and container digests; runtime startup has no DDL permission and validates that the installed pgvector extension and migration-created table support the fixed `EmbeddingSpace` type, dimension, and distance operator before ingestion.
- Real-PostgreSQL concurrency tests prove partial uniqueness, same-KB selection, atomic source sequences, reverse-order version completion, and old-or-new (never mixed) retrieval during revision activation.
- Exact vector retrieval returns index chunks with source document metadata through the single-statement snapshot query.
- The small golden evaluation set is tied to a `CorpusProfile`, compares exact vector retrieval with a lexical baseline, includes exact identifiers/terms, and reports Recall@k for representative access/revision/status filters as well as the unfiltered baseline. HNSW is disabled unless filtered Recall@k and latency gates pass.
- Concise and summary styles can each be combined with refuse and partial-answer insufficiency behavior through the API.
- P1A always enforces evidence-only grounding and required claim-level citations, and returns the complete effective policy on every ChatRun.
- `POST /api/v1/chat/runs` returns `202`, `run_id`, status URL, and events URL before execution; repeating a lost request with the same idempotency key returns the same run.
- One independent worker completes a run after the creating API request and any SSE subscription disconnect; fair chat/indexing lanes meet the configured chat queue-time target under indexing load, aging prevents indexing starvation, and claims, independent heartbeats/sessions, deadlines, reconciliation, and finite retries leave either a completed run or a visible durable failure state rather than duplicate execution or indefinite retry.
- GET SSE holds no database session while waiting, respects connection and duration limits, and terminates with `answer.completed` or `run.failed` from committed business state. A disconnect/reconnect or SSE timeout recovers by polling the status resource. Optional `answer.validated_delta` events, if enabled, contain only already committed validated answer text; raw provider draft tokens are never emitted.
- Retrying or reconciling the same ChatRun after injected failures does not create duplicate assistant messages, citations, or usage.
- `GET /api/v1/chat/runs/{run_id}` returns committed business state in all cases.
- A lightweight evaluation run records retrieval, citation, refusal, latency, and failure metrics.
- The test frontend exposes document, chat, and retrieval debug flows.
- A chat-model provider can be changed through configuration without business-code changes only when its capability declaration satisfies the pipeline contract. Changing the fixed P1A embedding model or embedding space follows the P1C migration protocol.
- Development identity is restricted to the fixed server principal on loopback. README states that the project is local validation only; all non-development profiles fail closed in P0/P1A.
- The shared source volume survives API/worker restarts, staging/final atomic-rename preconditions are validated, and idempotent cleanup/reset flows are documented and tested.
- Unit, contract, integration, and end-to-end tests for the vertical slice pass.

### P1B: Delivery reliability

- Durable Outbox and queue dispatch retry without duplicate business effects; Redis Pub/Sub is not used as the reliable queue.
- Consumer-group handling acknowledges only at the documented point, reclaims pending work after timeout, uses bounded exponential-backoff retries, and records exhausted messages in a DLQ.
- Competing runners obey lease/epoch fencing and stale-run recovery neither duplicates product facts nor lets a stale runner finalize a run. If checkpoint recovery is enabled, checkpoint resumption obeys the same invariant.
- GET SSE replays later retained events from `Last-Event-ID` and returns `stream.reset` with the status URL for an expired cursor.

### P1C: Index lifecycle

- Full index revision rebuild and activation preserve the serving and read-path invariants defined in Sections 6 through 8.
- Snapshot/catch-up applies every source sequence exactly once, cutover is atomic, and rollback/cleanup cannot expose retired or mixed-revision content.

## 22. Implementation Planning Boundary

The architecture design defines a P0/P1A implementation boundary and reserves independent P1B delivery-reliability and P1C index-lifecycle interfaces without requiring their operational machinery in the first plan. The next artifact is a detailed P0 and P1A implementation plan with small, verifiable tasks, followed by separately prioritized P1B and P1C plans informed by P1A measurements. No production parser, authentication system, external connector, formal backup/compliance system, or department deployment implementation belongs in the P0/P1A plan.
