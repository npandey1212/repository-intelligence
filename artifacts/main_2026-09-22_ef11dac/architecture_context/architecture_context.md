# Architecture Context --- Agentic Data Onboarding Platform (ADOP)

-   **Repository:** `aws-samples/sample-Agentic-Ai-Data-Operations`
-   **Pinned snapshot used for this context:** `main_2026-09-22_ef11dac`
-   **Pinned commit:** `ef11dac8618b6344d6a5e4add6ef51cd2811d7e0`
-   **Purpose of this document:** Give the Planner a concise,
    evidence-oriented operational model of the repository so it can
    create relevant investigation plans.
-   **Evidence policy:** Treat repository source, configuration,
    schemas, workflow definitions, and runbooks as evidence. A
    component's mention in documentation does not by itself prove that
    it is fully implemented or active at runtime. Keep documented
    intent, configured capability, implementation evidence, and inferred
    relationships distinct.

## 1. Executive summary

The Agentic Data Onboarding Platform (ADOP) is a Claude Code-oriented
agentic workflow repository for generating data-engineering artifacts
from a natural-language data-onboarding request. The documented goal is
to reduce repeated manual work involved in creating data ingestion and
transformation pipelines, data-quality rules, orchestration definitions,
metadata/semantic artifacts, infrastructure configuration, tests, and
supporting documentation.

The platform's documented data-processing pattern is **Bronze → Silver →
Gold**:

-   **Bronze:** raw source data retained without transformation.
-   **Silver:** cleaned, validated, and deduplicated data. The
    repository's explanatory documentation describes a quality threshold
    of at least `0.80`.
-   **Gold:** curated, business-ready data. The explanatory
    documentation describes a quality threshold of at least `0.95`.

The AI agents primarily operate during development to generate or
validate artifacts. The intended deployed pipeline is generated
code/configuration---such as PySpark scripts and Airflow DAGs---run
through ordinary engineering and deployment mechanisms. Do not assume
the agents remain in the production runtime data path unless a specific
source proves that.

The main documented workflows are: - `/onboard-workflow [REGULATION]`
--- the data onboarding workflow. -
`/devops-workflow [WORKLOAD] [FRAMEWORK]` --- the DevOps /
infrastructure preparation workflow.

The repository also documents an important permission boundary:
specialized sub-agents generate artifacts, while direct MCP/AWS
operations are restricted to the main conversation/agent according to
the routing and guardrail documentation. Verify this boundary against
the pinned `TOOL_ROUTING.md`, `MCP_GUARDRAILS.md`, and registry before
making claims about a particular operation.

This context is a planning aid, not a substitute for evidence
collection. For every nontrivial relationship, the Investigator should
open the source file and cite the relevant lines or section.

## 2. Repository purpose and scope

### Primary purpose

Given a description of a data source, workload, refresh cadence,
data-quality needs, and applicable controls, the platform coordinates
agents that produce artifacts for an onboarding workload. The documented
outputs can include:

-   workload configuration and specifications;
-   metadata/profile findings and schema-related artifacts;
-   transformation scripts for Bronze, Silver, and Gold stages;
-   data-quality rules and quality specifications;
-   orchestration DAGs;
-   infrastructure-as-code artifacts;
-   tests and validation material;
-   ontology/semantic-layer staging artifacts where applicable;
-   documentation and deployment-related configuration.

The canonical workload output location described in the repository
documentation is `workloads/<workload-name>/`. Confirm the exact
artifacts for a specific workload from its files rather than assuming
every workload contains every artifact type.

### Important scope distinctions

1.  **Workflow definition versus agent implementation:** a command or
    runbook may describe intended orchestration; inspect its content to
    determine what it actually calls and in what order.
2.  **Configured versus available:** a server in `.mcp.json` is
    configured, but that alone does not prove that credentials, network
    access, or the corresponding operation succeed.
3.  **Available versus planned:** some DevOps capabilities are
    documented as planned. Do not describe them as implemented without
    corroborating source or tests.
4.  **Generated artifacts versus runtime services:** a logical agent
    that generates a DAG or Terraform file is not automatically a
    deployed runtime service.
5.  **Documentation versus execution evidence:** README/runbook
    statements describe intended behavior. Code, configuration, tests,
    and execution traces provide different kinds of evidence; report
    discrepancies explicitly.

## 3. Primary entry points

Investigate these first when a question concerns how the platform starts
or orchestrates work.

  --------------------------------------------------------------------------------
  Entry point / file                           Role to establish from source
  -------------------------------------------- -----------------------------------
  `README.md`                                  Repository purpose, supported
                                               workflows, setup, examples, and
                                               top-level component map.

  `.claude/commands/onboard-workflow.md`       Entry point and steps for the
                                               data-onboarding workflow.

  `.claude/commands/devops-workflow.md`        Entry point and steps for the
                                               DevOps workflow.

  `runbooks/data-onboarding-agent/README.md`   Onboarding agent responsibilities,
                                               phases, checks, and handoffs.

  `runbooks/devops-agent/README.md`            DevOps scope and
                                               implementation-status notes.

  `TOOL_ROUTING.md`                            Tool/server selection, routing
                                               rules, fallbacks, and caller
                                               restrictions.

  `MCP_GUARDRAILS.md`                          MCP/AWS operational guardrails and
                                               prohibited operations.

  `.mcp.json`                                  MCP server configuration entries.
                                               This is configuration evidence, not
                                               proof that every server is
                                               reachable.

  `tool-registry/servers.yaml`                 Registry-level server types,
                                               capabilities, tool names, purposes,
                                               and fallback metadata.

  `docs/workflow-diagrams.md`                  Documented workflow relationships
                                               and diagrams; compare with
                                               command/runbook details.

  `pyproject.toml`                             Python project metadata and
                                               declared dependencies.

  `contracts/`                                 Structured specifications/schemas
                                               used by the data workflows; inspect
                                               individual schemas for exact fields
                                               and constraints.
  --------------------------------------------------------------------------------

The paths above are navigation anchors. They are not a claim that every
file is executable code or that every documented relationship has been
runtime-verified.

## 4. Main workflows

### 4.1 Data onboarding

The documented user entry point is `/onboard-workflow [REGULATION]`. The
workflow is intended to coordinate onboarding from a natural-language
request to workload artifacts.

The onboarding runbook describes phases including:

1.  **Health check and auto-detection** --- inspect the
    environment/workload state and detect relevant existing material.
2.  **Discovery** --- gather missing workload, schema, business,
    quality, schedule, or regulatory information.
3.  **Deduplication and validation** --- establish relevant input and
    validation expectations.
4.  **Profiling** --- characterize data and inform downstream
    generation.
5.  **Artifact generation** --- coordinate specialized agents to
    generate their assigned outputs.
6.  **Deployment / operational phase** --- use permitted MCP operations
    through the allowed caller path, subject to guardrails and
    validation.

Treat this as a documented phase outline, not proof that every phase
always executes in this exact sequence. Inspect
`.claude/commands/onboard-workflow.md` and the current onboarding
runbook to determine branching, parallel tasks, prerequisites, retry
behavior, and completion criteria for the question at hand.

### 4.2 DevOps workflow

The documented entry point is `/devops-workflow [WORKLOAD] [FRAMEWORK]`.
It concerns infrastructure/deployment preparation for a workload and
includes infrastructure-as-code generation.

The DevOps runbook distinguishes available IaC generation from other
capabilities described as planned, including broader CI/CD, monitoring,
cost optimization, and self-healing work. For any capability-status
question: - read `runbooks/devops-agent/README.md`; - inspect the
corresponding agent definition and supporting code/configuration; -
check tests or examples if available; - report `implemented`, `partial`,
`planned`, or `unknown` with evidence rather than treating all
documented roadmap items as working features.

### 4.3 Data lifecycle

The platform documents the familiar medallion pattern:

`Source data → Bronze (raw) → Silver (cleaned/validated/deduplicated) → Gold (curated/business-ready)`

Quality thresholds described in the repository's explanatory material
are: - Silver: `>= 0.80` - Gold: `>= 0.95`

The repository contains contract schemas such as: -
`contracts/v1/bronze_spec.schema.json` -
`contracts/v1/silver_spec.schema.json` -
`contracts/v1/gold_spec.schema.json` -
`contracts/v1/quality_spec.schema.json`

For a concrete claim about fields, gate evaluation, or promotion
behavior, inspect the schemas and the implementation that consumes them.
The threshold descriptions alone do not establish how a score is
calculated or enforced.

## 5. Agent and component model

The repository includes specialized agent definitions under
`.claude/agents/` and corresponding workflow/runbook material. Relevant
documented roles include:

  -------------------------------------------------------------------------------------------------------
  Agent / role            Intended responsibility            Evidence to inspect
  ----------------------- ---------------------------------- --------------------------------------------
  Data Onboarding Agent   Orchestrates onboarding,           `.claude/agents/`,
                          coordinates discovery and          `.claude/commands/onboard-workflow.md`,
                          specialized work, and owns the     `runbooks/data-onboarding-agent/README.md`
                          broader workflow.                  

  Metadata Agent          Profiles data and produces         Corresponding `.claude/agents/` definition,
                          metadata/schema/business-context   runbooks, workload artifacts
                          artifacts.                         

  Data Transformation     Produces transformation artifacts  Agent definition, contracts, generated
  Agent                   across the data lifecycle.         workload scripts

  Data Quality Agent      Defines or generates data-quality  Agent definition, quality contract, tests,
                          checks/specifications.             generated quality artifacts

  DAG / Orchestration     Produces orchestration DAG         `dag-agent` definition, workflow command,
  Agent                   artifacts.                         generated DAGs

  IaC / DevOps Agent      Produces infrastructure-as-code or `iac-agent` definition, DevOps command and
                          deployment-preparation artifacts.  runbook

  Ontology Agent          Handles ontology/semantic modeling Agent definition, relevant contracts and
                          tasks where included in the        outputs
                          workflow.                          

  Ontology Staging Agent  Supports ontology staging for a    Agent definition, runbook and status
                          future semantic-layer integration; statements
                          do not assume the future service   
                          is deployed or complete.           
  -------------------------------------------------------------------------------------------------------

This table is a navigation map, not a complete authoritative inventory
of all agent files. The architecture extractor should enumerate the
actual `.claude/agents/` directory at the pinned commit and preserve
exact names. Do not infer that every agent is invoked in every workflow.

### Relationships the Planner should distinguish

-   `invokes`: a workflow/agent calls another agent or command.
-   `coordinates`: an orchestrator controls or sequences work.
-   `runs_in_parallel_with`: two tasks may execute concurrently, when
    the source explicitly says so.
-   `produces`: an agent generates an artifact.
-   `consumes`: an agent reads a prior artifact or result.
-   `validates`: a test, quality gate, or validator checks an artifact.
-   `hands_off_to`: one phase hands results to another phase.
-   `permitted_to_call`: a caller is allowed to invoke a tool/server
    under a policy.
-   `prohibited_from_calling`: a caller is explicitly restricted.
-   `documents`: a document describes a component or relationship.

Do not reduce all of these to a generic `TRIGGERS` edge. Preserve the
relation type and the evidence that justifies it.

## 6. Tools, MCP servers, and routing

### 6.1 Sources of truth

Use the following sources together; they answer different questions:

-   `.mcp.json`: which MCP server entries are configured and how they
    are launched/connected.
-   `tool-registry/servers.yaml`: registry metadata for server type,
    purpose, tools, and fallbacks.
-   `TOOL_ROUTING.md`: which tools/server paths are intended for which
    phase and caller, including fallback rules.
-   `MCP_GUARDRAILS.md`: operational constraints and prohibited actions.
-   Agent and command definitions: which capabilities are requested or
    referenced by a workflow.
-   Tests and execution traces, where available: whether behavior is
    validated or was actually observed.

A configured server, registry entry, tool mentioned in a prompt, and
tool actually invoked are four different facts. Do not collapse them.

### 6.2 Configured MCP server inventory

The `.mcp.json` snapshot has 13 configured server entries, including:

-   `glue-athena`
-   `lakeformation`
-   `sagemaker-catalog`
-   `core`
-   `iam`
-   `lambda`
-   `s3-tables`
-   `cloudtrail`
-   `redshift`
-   `cloudwatch`
-   `cost-explorer`
-   `dynamodb`
-   `pii-detection`

Treat this as a configuration inventory only. The actual tool names,
arguments, server commands, enabled state, and fallbacks must be checked
in `.mcp.json` and `tool-registry/servers.yaml`. Do not assume all
configured servers are required for every workflow or that every listed
operation is available in the current environment.

The registry's documented purpose categories include Glue/Athena access,
Lake Formation permissions and LF-Tags, catalog/metadata operations, IAM
inspection/simulation, Lambda, S3 Tables, CloudTrail, Redshift,
CloudWatch, cost exploration, DynamoDB, and PII detection/security. Use
the registry to confirm exact names and scope before planning
tool-specific work.

### 6.3 Caller permissions and guardrails

`TOOL_ROUTING.md` and `MCP_GUARDRAILS.md` are critical sources for tool
interaction questions. The documented design distinguishes: - the main
conversation/agent, which may invoke approved MCP tools and use
specified CLI/Bash fallbacks; - specialized sub-agents spawned through
the `Agent` tool, which generate scripts/configuration/artifacts and are
not permitted to directly perform MCP/AWS operations under the
`sub-agent-no-mcp` invariant.

Before asserting that an agent can call a server, verify both: 1. the
tool/server exists and is configured/registered; and 2. the caller is
permitted to invoke it in that workflow phase.

Also inspect fallback behavior: a missing optional MCP capability may
have a documented CLI or alternative path, but a fallback is not the
same as the preferred path and may have different safety conditions.

### 6.4 MCP interaction trace template

For a question such as "How do agents, tools, and MCP servers
interact?", build an evidence-backed chain for each relevant operation:

`Workflow command → orchestrator/phase → authorized caller → selected tool → MCP server → external AWS/API capability → returned observation/result → generated artifact or next workflow step`

Not every operation will traverse every link. Mark absent or unproven
links as unknown. Do not invent a direct agent-to-server edge solely
because the agent's task relates to AWS.

## 7. Contracts, artifacts, and validation

The `contracts/` directory includes versioned schemas for core data
artifacts. Known schema paths include:

-   `contracts/v1/bronze_spec.schema.json`
-   `contracts/v1/silver_spec.schema.json`
-   `contracts/v1/gold_spec.schema.json`
-   `contracts/v1/quality_spec.schema.json`

Use schema files to identify required fields and validation constraints.
Then trace where each schema is read or produced. A schema's existence
alone does not prove a workflow enforces it.

The documented workload output convention is
`workloads/<workload-name>/`. Depending on workflow scope, outputs can
include configuration/specification files, PySpark scripts, SQL, quality
checks, Airflow DAGs, IaC, tests, and documentation. For an actual
workload, inventory its generated files and map each to the agent or
phase that produced it only when evidence supports that relationship.

Investigations about quality or promotion should trace the complete
path:

`quality contract/rule → generated check → execution/validation mechanism → score/result → promotion decision`

Do not assume this full chain exists merely because thresholds are
described in documentation.

## 8. Implementation status and uncertainty

Use explicit status labels at capability level:

-   **Implemented:** supported by relevant source/configuration and,
    where applicable, tests or observed execution.
-   **Configured:** present in configuration/registry, but runtime
    reachability or successful operation has not been demonstrated.
-   **Documented:** stated in a README, runbook, or command prompt;
    implementation may still need verification.
-   **Partial:** some pieces exist, but the full workflow or capability
    is incomplete.
-   **Planned:** explicitly described as future work.
-   **Unknown:** available evidence does not establish status.
-   **Conflicting:** two sources disagree; cite both and do not silently
    choose one.

Particular cautions: - The DevOps runbook describes IaC generation as
available and broader CI/CD, monitoring, cost optimization, and
self-healing capabilities as planned. - The ontology-staging material
describes preparation for a future semantic-layer product; do not treat
the future product as an existing deployed runtime dependency without
evidence. - The existence of MCP configuration does not prove that a
server starts or that an operation succeeds. - An agent prompt
describing a behavior does not prove the behavior is enforced by code or
tests. - An architecture diagram or README can be stale; compare it with
current configuration, source, and tests.

## 9. Planner guidance: select plans based on the user's question

The Planner should create the smallest set of focused plans that
collectively answer the request. Do not create every plan below for
every query.

### A. "How do agents, tools, and MCP servers interact?"

Recommended focused plans: 1. **Agent/workflow map:** enumerate agent
definitions and workflow commands; extract invocation and phase
relationships. 2. **MCP/tool inventory:** extract server configuration
and registry tool names/purposes. 3. **Routing and permission rules:**
trace allowed callers, prohibited calls, phase-specific rules, and
fallbacks. 4. **End-to-end interaction trace:** choose the workflow
relevant to the question and prove one or more concrete operation
chains. 5. **Synthesis and gap report:** merge results, distinguish
observed/documented/inferred relationships, and list unresolved edges.

Do not ask one investigator to reread the whole README or every agent
file repeatedly. Each plan should have a bounded path/symbol scope and a
clear completion condition.

### B. "How does onboarding work end to end?"

Recommended plans: 1. Read the onboarding command and runbook; extract
entry conditions and phases. 2. Map phase-to-agent assignments,
dependencies, and parallelism from the workflow source. 3. Map generated
artifacts and contracts to producers/consumers. 4. Trace quality checks,
validation, and deployment/operational handoffs. 5. Synthesize a
phase-ordered workflow with evidence and unresolved behavior.

### C. "Which tools can each agent use?"

Recommended plans: 1. Enumerate agents and their explicit tool
references. 2. Enumerate configured MCP servers and registry tools. 3.
Extract caller permissions and guardrails. 4. Cross-check references
against the registry/configuration. 5. Report allowed, prohibited,
conditional/fallback, and unknown relationships separately.

### D. "Is a DevOps capability implemented?"

Recommended plans: 1. Read the DevOps runbook for explicit status
claims. 2. Inspect the agent definition and corresponding
implementation/configuration. 3. Search tests/examples for the claimed
behavior. 4. Compare evidence and return a status label with confidence
and citations.

### E. "How do quality gates affect Bronze/Silver/Gold?"

Recommended plans: 1. Read the stage and quality contracts. 2. Find
code/prompts that generate quality checks. 3. Find the validator or
workflow step that consumes scores/results. 4. Establish whether
thresholds are enforced and where promotion is decided. 5. Report
documented thresholds separately from implementation-confirmed
enforcement.

### Plan output contract

Every plan should specify: - **Objective:** one answerable
sub-question. - **Scope:** exact paths, symbols, configuration entries,
or workflow phases to inspect. - **Evidence to collect:** facts and
relationships required for a useful result. - **Budget:** bounded tool
calls and bounded returned text; avoid repeatedly retrieving the same
file. - **Completion condition:** what evidence is sufficient to stop. -
**Output:** concise structured findings with source path and
line/section references. - **Unknowns:** missing evidence,
contradictions, and assumptions that must not be presented as facts.

If a source is missing or a relationship cannot be proven within the
plan's budget, report the gap and hand it to synthesis. Do not keep
repeating an unproductive search.

## 10. Evidence requirements for Investigators and Analysts

For each significant claim, record: - claim text; - subject and
relation/object where applicable; - evidence path and line range or
named section; - source type (code, configuration, schema, command,
runbook, README, test, trace); - status label
(implemented/configured/documented/planned/unknown/conflicting); -
whether the claim is directly observed, derived, or inferred; -
unresolved caveats.

Prefer direct source/configuration evidence over generic summaries. Use
README/runbooks to understand intended behavior, but use
code/config/tests to validate implementation where possible. When
sources conflict, preserve the conflict rather than silently reconciling
it.

## 11. Key files to inspect by investigation topic

  ---------------------------------------------------------------------------------------------------
  Topic                   First sources                                Cross-check sources
  ----------------------- -------------------------------------------- ------------------------------
  Purpose / capabilities  `README.md`                                  workflow commands, runbooks,
                                                                       examples

  Onboarding workflow     `.claude/commands/onboard-workflow.md`,      `docs/workflow-diagrams.md`,
                          `runbooks/data-onboarding-agent/README.md`   agent definitions, generated
                                                                       workload

  DevOps workflow         `.claude/commands/devops-workflow.md`,       IaC agent definition,
                          `runbooks/devops-agent/README.md`            examples, tests

  Agents                  `.claude/agents/`                            command files, runbooks, tool
                                                                       references

  MCP configuration       `.mcp.json`                                  `tool-registry/servers.yaml`

  Tool routing            `TOOL_ROUTING.md`                            `MCP_GUARDRAILS.md`, commands,
                                                                       agent prompts

  Guardrails              `MCP_GUARDRAILS.md`                          registry invariants, tests,
                                                                       execution evidence

  Data stages / quality   `contracts/v1/`                              transformation and quality
                                                                       agents, generated artifacts,
                                                                       validation code

  Dependencies            `pyproject.toml`                             imports and tests

  Ontology / semantic     ontology agent definitions and related       contracts, generated outputs,
  staging                 runbooks                                     explicit status notes

  Actual workload output  `workloads/<workload-name>/`                 producer agent/command and
                                                                       schemas
  ---------------------------------------------------------------------------------------------------

Paths should be verified against the pinned checkout before relying on
them. If a named file is absent, record that fact and search for its
actual counterpart rather than silently substituting a different source.

## 12. Known limitations of this context

This document summarizes the repository based on available
repository-level descriptions and known configuration/structure. It is
not a full source-code execution trace. In particular: - The exact
implementation of every agent and every server tool has not been
reproduced here. - The listed MCP server names are configuration
entries; their runtime availability is not established by the list. -
The documented onboarding phase sequence must be checked against the
command/runbook for conditional branches and exact sequencing. - The
stated Bronze/Silver/Gold thresholds are documented values; enforcement
mechanics must be traced. - Capability status can vary by file/version
and must be verified at the pinned commit. - No relationship should be
considered runtime-observed unless supported by execution evidence.

The Planner should use this context to choose relevant sources and
structure investigations, not to answer implementation-specific
questions without opening those sources.

## 13. Guiding principle

**Plan around relationships and claims that must be proven, not around a
list of files to summarize.**

The correct investigation loop is:

`Question → focused plans → bounded evidence retrieval → claim/relationship validation → synthesis with explicit unknowns`

A successful plan should reduce unnecessary context, avoid repeated tool
calls, and produce enough source-grounded evidence to answer the user's
actual question. The architecture context is useful only when it directs
the Planner toward the right evidence without pretending that
undocumented or unverified behavior is known.
