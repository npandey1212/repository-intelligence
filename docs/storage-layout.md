# Storage Layout

Three roots, three owners. Everything persistent lives under
`artifacts/` or `db_store/`; `runtime/` never holds a database or
index.

```text
repository-intelligence/
├── artifacts/                    # pinned analysis artifacts (M1–M13)
│   ├── <snapshot_id>/
│   │   ├── facts/
│   │   ├── architecture/
│   │   ├── engineering/          # model + engineering_graph.json
│   │   ├── snapshot/
│   │   │   └── repository_snapshot.json
│   │   └── repository/
│   │       └── <repository_name>/   # pinned checkout (searched tree)
│   └── retrieval_flow/           # flow-diagnostic runs (M18)
├── db_store/                     # derived, regenerable stores
│   ├── memory/
│   │   ├── agent_memory.db       # persistent agent memory (M14)
│   │   └── repository_query.db   # query history (M9+)
│   └── rag/
│       ├── metadata.db           # chunk provenance (SQLite)
│       └── vector_store/
│           └── <repository_key>/
│               └── <commit_sha>/
│                   └── <embedding_model>/
│                       ├── manifest.json
│                       └── index.faiss
└── runtime/                      # ephemeral output ONLY
    └── traces/, logs, demo scratch
```

## Ownership

| Path | Written by | Never contains |
| --- | --- | --- |
| `artifacts/` | snapshot/facts/architecture pipeline (M1–M13) | databases, vector indexes |
| `db_store/memory/` | persistent memory manager (M14), query recording (M9+) | checked-in data |
| `db_store/rag/` | `build_index()` / `repo-intelligence retrieval-index` (M18) | source files |
| `runtime/` | trace sinks, demo scratch | `agent_memory.db`, `repository_query.db`, FAISS indexes, RAG metadata |

`runtime/` is deliberately disposable: deleting it never loses
pinned artifacts or stores.

## Environment seam

`REPO_INTELLIGENCE_DB_STORE` (constant `DB_STORE_ENV_VAR` in
`repo_intelligence.storage`) relocates the entire `db_store` tree:

```bash
export REPO_INTELLIGENCE_DB_STORE=/tmp/scratch/db_store
```

All helpers (`default_db_store_root()`, `default_memory_db_path()`,
`default_rag_store_root()`, `rag_vector_store_path()`,
`default_query_db_path()`, …) resolve through it. Tests set it to a
per-test tmp directory so no suite run ever touches the project's
real stores.

## Source of truth

```text
Pinned Repository Snapshot
        ↓
Deterministic Repository Facts
        ↓
Engineering Model
        ↓
Engineering Graph
```

`db_store/rag` is a retrieval projection of the pinned checkout:
regenerable, never authoritative. If RAG results and the
deterministic graph disagree, expose the discrepancy — do not
silently reconcile.

## Migration

Legacy layouts are handled by code, not by hand:

* flat runs (`<output>/<run>/repository/`) resolve alongside the
  named layout (`.../repository/<repository_name>/`);
* pre-`db_store` memory databases under `runtime/agent_memory/` are
  migrated by `python -m repo_intelligence.storage` (add
  `--dry-run` to preview; see
  `docs/m14-persistent-agent-memory.md`).
