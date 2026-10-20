# Architecture Visualization Run

## User Question

How do the agents, tools, and MCP servers interact?

repository_scope: sample-Agentic-Ai-Data-Operations
commit_scope: ef11dac8618b6344d6a5e4add6ef51cd2811d7e0

## Architecture Investigator

Status: complete
Summary: Read-only inspection of 5 file(s) established 85 component(s) and 66 relationship(s).
Key findings:
- Component data-onboarding-agent (agent): Data Onboarding Agent [observed]
- Component ontology-staging-agent (agent): Ontology Staging Agent (AWS Semantic Layer Handoff) [observed]
- Component environment-setup-agent (agent): Environment Setup Agent [observed]
- Component glue-athena (mcp_server): MCP server registered in servers.yaml [observed]
- Component create-database (tool): tool served by the glue-athena MCP server [observed]
- Component get-table (tool): tool served by the glue-athena MCP server [observed]
- Component get-tables (tool): tool served by the glue-athena MCP server [observed]
- Component create-crawler (tool): tool served by the glue-athena MCP server [observed]
- Component start-crawler (tool): tool served by the glue-athena MCP server [observed]
- Component start-job-run (tool): tool served by the glue-athena MCP server [observed]

## Flow / Dependency Investigator

Status: complete
Summary: Read-only inspection of 5 file(s) established the path add-lf-tags-to-resource -> EXPOSED_BY -> lakeformation over 1 hop(s).
path_status: FOUND
Flow: add-lf-tags-to-resource -> lakeformation

## Architecture Visualization Agent

Diagram type: AGENT_TOOL_MCP
Title: How do the agents, tools, and MCP servers interact? (85 component(s))
Nodes: 85
Edges: 66
Node list:
- data-onboarding-agent (agent, observed, confidence high): data-onboarding-agent
- ontology-staging-agent (agent, observed, confidence high): ontology-staging-agent
- environment-setup-agent (agent, observed, confidence high): environment-setup-agent
- glue-athena (mcp_server, observed, confidence high): glue-athena
- create-database (tool, observed, confidence high): create-database
- get-table (tool, observed, confidence high): get-table
- get-tables (tool, observed, confidence high): get-tables
- create-crawler (tool, observed, confidence high): create-crawler
- start-crawler (tool, observed, confidence high): start-crawler
- start-job-run (tool, observed, confidence high): start-job-run
- get-job-run (tool, observed, confidence high): get-job-run
- athena-query (tool, observed, confidence high): athena-query
- lakeformation (mcp_server, observed, confidence high): lakeformation
- create-lf-tag (tool, observed, confidence high): create-lf-tag
- list-lf-tags (tool, observed, confidence high): list-lf-tags
- add-lf-tags-to-resource (tool, observed, confidence high): add-lf-tags-to-resource
- remove-lf-tags-from-resource (tool, observed, confidence high): remove-lf-tags-from-resource
- get-resource-lf-tags (tool, observed, confidence high): get-resource-lf-tags
- grant-permissions (tool, observed, confidence high): grant-permissions
- revoke-permissions (tool, observed, confidence high): revoke-permissions
- batch-grant-permissions (tool, observed, confidence high): batch-grant-permissions
- get-lf-tag (tool, observed, confidence high): get-lf-tag
- iam (mcp_server, observed, confidence high): iam
- list-roles (tool, observed, confidence high): list-roles
- simulate-principal-policy (tool, observed, confidence high): simulate-principal-policy
- list-role-policies (tool, observed, confidence high): list-role-policies
- get-role-policy (tool, observed, confidence high): get-role-policy
- put-role-policy (tool, observed, confidence high): put-role-policy
- create-role (tool, observed, confidence high): create-role
- list-users (tool, observed, confidence high): list-users
- get-user (tool, observed, confidence high): get-user
- attach-user-policy (tool, observed, confidence high): attach-user-policy
- attach-group-policy (tool, observed, confidence high): attach-group-policy
- cloudtrail (mcp_server, observed, confidence high): cloudtrail
- lookup-events (tool, observed, confidence high): lookup-events
- lake-query (tool, observed, confidence high): lake-query
- list-event-data-stores (tool, observed, confidence high): list-event-data-stores
- get-query-results (tool, observed, confidence high): get-query-results
- get-query-status (tool, observed, confidence high): get-query-status
- redshift (mcp_server, observed, confidence high): redshift
- list-clusters (tool, observed, confidence high): list-clusters
- list-databases (tool, observed, confidence high): list-databases
- list-schemas (tool, observed, confidence high): list-schemas
- list-tables (tool, observed, confidence high): list-tables
- list-columns (tool, observed, confidence high): list-columns
- execute-query (tool, observed, confidence high): execute-query
- core (mcp_server, observed, confidence high): core
- s3-operations (tool, observed, confidence high): s3-operations
- kms-operations (tool, observed, confidence high): kms-operations
- secrets-manager (tool, observed, confidence high): secrets-manager
- s3-tables (mcp_server, observed, confidence high): s3-tables
- s3-tables-operations (tool, observed, confidence high): s3-tables-operations
- pii-detection (mcp_server, observed, confidence high): pii-detection
- detect-pii-in-table (tool, observed, confidence high): detect-pii-in-table
- scan-database-for-pii (tool, observed, confidence high): scan-database-for-pii
- create-lf-tags (tool, observed, confidence high): create-lf-tags
- get-pii-columns (tool, observed, confidence high): get-pii-columns
- apply-column-security (tool, observed, confidence high): apply-column-security
- get-pii-report (tool, observed, confidence high): get-pii-report
- sagemaker-catalog (mcp_server, observed, confidence high): sagemaker-catalog
- put-custom-metadata (tool, observed, confidence high): put-custom-metadata
- get-custom-metadata (tool, observed, confidence high): get-custom-metadata
- list-tables-with-metadata (tool, observed, confidence high): list-tables-with-metadata
- search-metadata (tool, observed, confidence high): search-metadata
- delete-custom-metadata (tool, observed, confidence high): delete-custom-metadata
- lambda (mcp_server, observed, confidence high): lambda
- aws-lambdafn-lf-access-grant-new (tool, observed, confidence high): aws-lambdafn-lf-access-grant-new
- aws-lambda-lf-revoke-access-new (tool, observed, confidence high): aws-lambda-lf-revoke-access-new
- spark-on-aws-lambda (tool, observed, confidence high): spark-on-aws-lambda
- tagging-finder (tool, observed, confidence high): tagging-finder
- lf-access-grant (tool, observed, confidence high): lf-access-grant
- cloudwatch (mcp_server, observed, confidence high): cloudwatch
- logs (tool, observed, confidence high): logs
- metrics (tool, observed, confidence high): metrics
- alarms (tool, observed, confidence high): alarms
- dashboards (tool, observed, confidence high): dashboards
- cost-explorer (mcp_server, observed, confidence high): cost-explorer
- cost-and-usage (tool, observed, confidence high): cost-and-usage
- dynamodb (mcp_server, observed, confidence high): dynamodb
- table-crud (tool, observed, confidence high): table-crud
- query (tool, observed, confidence high): query
- scan (tool, observed, confidence high): scan
- sns-sqs (mcp_server, observed, confidence high): sns-sqs
- eventbridge (mcp_server, observed, confidence high): eventbridge
- stepfunctions (mcp_server, observed, confidence high): stepfunctions
Edge list:
- create-database -EXPOSED_BY-> glue-athena (derived)
- get-table -EXPOSED_BY-> glue-athena (derived)
- get-tables -EXPOSED_BY-> glue-athena (derived)
- create-crawler -EXPOSED_BY-> glue-athena (derived)
- start-crawler -EXPOSED_BY-> glue-athena (derived)
- start-job-run -EXPOSED_BY-> glue-athena (derived)
- get-job-run -EXPOSED_BY-> glue-athena (derived)
- athena-query -EXPOSED_BY-> glue-athena (derived)
- create-lf-tag -EXPOSED_BY-> lakeformation (derived)
- list-lf-tags -EXPOSED_BY-> lakeformation (derived)
- add-lf-tags-to-resource -EXPOSED_BY-> lakeformation (derived)
- remove-lf-tags-from-resource -EXPOSED_BY-> lakeformation (derived)
- get-resource-lf-tags -EXPOSED_BY-> lakeformation (derived)
- grant-permissions -EXPOSED_BY-> lakeformation (derived)
- revoke-permissions -EXPOSED_BY-> lakeformation (derived)
- batch-grant-permissions -EXPOSED_BY-> lakeformation (derived)
- get-lf-tag -EXPOSED_BY-> lakeformation (derived)
- list-roles -EXPOSED_BY-> iam (derived)
- simulate-principal-policy -EXPOSED_BY-> iam (derived)
- list-role-policies -EXPOSED_BY-> iam (derived)
- get-role-policy -EXPOSED_BY-> iam (derived)
- put-role-policy -EXPOSED_BY-> iam (derived)
- create-role -EXPOSED_BY-> iam (derived)
- list-users -EXPOSED_BY-> iam (derived)
- get-user -EXPOSED_BY-> iam (derived)
- attach-user-policy -EXPOSED_BY-> iam (derived)
- attach-group-policy -EXPOSED_BY-> iam (derived)
- lookup-events -EXPOSED_BY-> cloudtrail (derived)
- lake-query -EXPOSED_BY-> cloudtrail (derived)
- list-event-data-stores -EXPOSED_BY-> cloudtrail (derived)
- get-query-results -EXPOSED_BY-> cloudtrail (derived)
- get-query-status -EXPOSED_BY-> cloudtrail (derived)
- list-clusters -EXPOSED_BY-> redshift (derived)
- list-databases -EXPOSED_BY-> redshift (derived)
- list-schemas -EXPOSED_BY-> redshift (derived)
- list-tables -EXPOSED_BY-> redshift (derived)
- list-columns -EXPOSED_BY-> redshift (derived)
- execute-query -EXPOSED_BY-> redshift (derived)
- s3-operations -EXPOSED_BY-> core (derived)
- kms-operations -EXPOSED_BY-> core (derived)
- secrets-manager -EXPOSED_BY-> core (derived)
- s3-tables-operations -EXPOSED_BY-> s3-tables (derived)
- detect-pii-in-table -EXPOSED_BY-> pii-detection (derived)
- scan-database-for-pii -EXPOSED_BY-> pii-detection (derived)
- create-lf-tags -EXPOSED_BY-> pii-detection (derived)
- get-pii-columns -EXPOSED_BY-> pii-detection (derived)
- apply-column-security -EXPOSED_BY-> pii-detection (derived)
- get-pii-report -EXPOSED_BY-> pii-detection (derived)
- put-custom-metadata -EXPOSED_BY-> sagemaker-catalog (derived)
- get-custom-metadata -EXPOSED_BY-> sagemaker-catalog (derived)
- list-tables-with-metadata -EXPOSED_BY-> sagemaker-catalog (derived)
- search-metadata -EXPOSED_BY-> sagemaker-catalog (derived)
- delete-custom-metadata -EXPOSED_BY-> sagemaker-catalog (derived)
- aws-lambdafn-lf-access-grant-new -EXPOSED_BY-> lambda (derived)
- aws-lambda-lf-revoke-access-new -EXPOSED_BY-> lambda (derived)
- spark-on-aws-lambda -EXPOSED_BY-> lambda (derived)
- tagging-finder -EXPOSED_BY-> lambda (derived)
- lf-access-grant -EXPOSED_BY-> lambda (derived)
- logs -EXPOSED_BY-> cloudwatch (derived)
- metrics -EXPOSED_BY-> cloudwatch (derived)
- alarms -EXPOSED_BY-> cloudwatch (derived)
- dashboards -EXPOSED_BY-> cloudwatch (derived)
- cost-and-usage -EXPOSED_BY-> cost-explorer (derived)
- table-crud -EXPOSED_BY-> dynamodb (derived)
- query -EXPOSED_BY-> dynamodb (derived)
- scan -EXPOSED_BY-> dynamodb (derived)

## Validation

Status: passed
Validation checks:
- node_validity: PASS
- relationship_validity: PASS — canonical kinds present: EXPOSED_BY
- evidence_validity: PASS
- classification_validity: PASS — classifications present: derived, observed
- status_validity: PASS — status 'complete'
- scope_validity: PASS

## Rendering

Renderer: M5 deterministic renderer (repo_intelligence.graph.exporters)
Artifacts:
- diagram_spec.json: /home/FRACTAL/nilesh.pandey/repository-intelligence/artifacts/architecture_visualization/diagram_spec.json
- architecture.mmd: /home/FRACTAL/nilesh.pandey/repository-intelligence/artifacts/architecture_visualization/architecture.mmd
- architecture.svg: NOT AVAILABLE — existing renderer does not support SVG
- run_summary.md: /home/FRACTAL/nilesh.pandey/repository-intelligence/artifacts/architecture_visualization/run_summary.md
SVG:
NOT AVAILABLE — existing renderer does not support SVG

## Evidence

5 reference(s) carried by this diagram:
- README.md
- tool-registry/servers.yaml
- tool-registry/invariants.yaml
- MCP_GUARDRAILS.md
- .mcp.json

## Uncertainties

- none recorded

## M17 Execution

Agents:
- INVESTIGATOR (architecture)
- INVESTIGATOR (flow)
- ANALYST (visualization)
- validator (deterministic M16 verdict)
- renderer (deterministic Mermaid/DOT/SVG stage)
Status: COMPLETED

## Trace

Trace/artifact location: runtime/agent_runs
