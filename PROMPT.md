Once you understand the manual workflow, your goal is to upgrade agent.py so it can handle this autonomously. Common improvements include:

Dynamic Prompting: Instructing the model to always fetch credentials and login tokens first.
State Management: Storing returned tokens/variables between steps so the agent doesn't have to keep re-logging in.
Fallback/Retry Logic: Parsing error messages returned by world.execute() and feeding them back to the LLM to fix its own Python code.

To store agent logs and have a memory of previous run use I want you to use HYDRA DB

# HydraDB Agent Integration Guide (Python Only)

A minimal, production-ready guide to integrate HydraDB using Python.

This guide removes TypeScript, JS, and raw HTTP complexity and focuses only on Python SDK usage.

---

# 1. Installation

```bash
pip install "hydradb-sdk>=2,<3"
2. Setup
import os
from hydra_db import HydraDB

client = HydraDB(token=os.environ["HYDRA_DB_API_KEY"])

Required env:

export HYDRA_DB_API_KEY="your_api_key"
3. Core Concepts
Tenant
A tenant = isolated workspace
Everything lives inside a tenant
Sub-tenant
Logical partition inside tenant (usually user-level or workspace-level)
MUST be consistent across ingest + query

Common patterns:

SaaS → tenant per customer, sub-tenant per user/team
B2C → tenant per app, sub-tenant per user
4. Tenant Lifecycle (IMPORTANT)
Step 1: Create tenant
client.tenants.create(tenant_id="my_tenant")
Step 2: Wait until ready
import time

while True:
    status = client.tenants.status(tenant_id="my_tenant")
    if status.data.infra.ready_for_ingestion:
        break
    time.sleep(5)
5. Ingest Data

HydraDB supports two main types:

knowledge → documents, files, app data
memory → user preferences / behavioral signals
5.1 Ingest Memory
import json

response = client.context.ingest(
    type="memory",
    tenant_id="my_tenant",
    sub_tenant_id="user_123",
    memories=json.dumps([
        {
            "id": "pref_1",
            "text": "User prefers detailed technical explanations and dark mode.",
            "infer": True
        }
    ])
)

ingest_id = response.data.results[0].id
5.2 Ingest Knowledge (files/documents)
with open("file.pdf", "rb") as f:
    response = client.context.ingest(
        type="knowledge",
        tenant_id="my_tenant",
        documents=[
            ("file.pdf", f, "application/pdf")
        ],
        document_metadata=json.dumps([
            {
                "id": "doc_1",
                "title": "My Document",
                "additional_metadata": {
                    "author": "Akhilesh"
                }
            }
        ])
    )
6. Poll ingestion status

ALL ingestion is async.

while True:
    status = client.context.status(
        tenant_id="my_tenant",
        sub_tenant_id="user_123",
        ids=[ingest_id]
    ).data.statuses[0]

    if status.indexing_status in ("graph_creation", "completed"):
        break

    if status.indexing_status in ("errored", "failed"):
        raise Exception("Indexing failed")

    time.sleep(2)
7. Query (Search)

Single unified retrieval API.

7.1 Knowledge search
result = client.query(
    tenant_id="my_tenant",
    query="What is this document about?",
    type="knowledge",
    query_by="hybrid",
    mode="thinking"
)
7.2 Memory search (personalization)
result = client.query(
    tenant_id="my_tenant",
    sub_tenant_id="user_123",
    query="What does the user prefer?",
    type="memory",
    query_by="hybrid",
    mode="thinking"
)
7.3 Combined (recommended for agents)
result = client.query(
    tenant_id="my_tenant",
    sub_tenant_id="user_123",
    query="Answer with full context",
    type="all",
    query_by="hybrid",
    mode="thinking"
)
8. Query Modes (important)
Mode	Meaning
fast	low latency
thinking	best quality, graph + rerank + expansion

👉 Use thinking for agents.

9. Query Types
Type	Purpose
knowledge	documents, files, external content
memory	user preferences & behavioral signals
all	combined retrieval (recommended)
10. Key Rules (DO NOT BREAK)
Always:
Use same sub_tenant_id for ingest + query
Wait for tenant readiness before ingesting
Wait for indexing before querying
Never:
Use metadata_filters instead of sub_tenant_id
Mix sub_tenants across write/read
Query immediately after ingest without polling
Assume processing is synchronous
11. Error Handling

Retry ONLY:

429 (rate limit)
500 (server error)
503 (service unavailable)
from hydra_db.errors import (
    TooManyRequestsError,
    InternalServerError,
    ServiceUnavailableError
)

try:
    result = client.query(...)
except (TooManyRequestsError, InternalServerError, ServiceUnavailableError):
    # retry with backoff
    pass
12. Recommended Agent Pattern
# 1. create tenant
# 2. wait readiness
# 3. ingest memory/knowledge
# 4. poll indexing
# 5. query using type="all"
# 6. feed result into LLM
13. Best Practices
Use memory for personalization (VERY powerful for agents)
Use knowledge for all external documents
Use all for final agent responses
Keep sub_tenant_id = user_id in most apps
Prefer thinking mode for quality over speed
14. Mental Model

Think of HydraDB as:

“A persistent memory + knowledge graph layer for LLM agents”

Tenants = apps
Sub-tenants = users
Knowledge = world data
Memory = user brain
Query = retrieval brain


The agent should have an eval framework and should learn from the logs and errors it makes that we store in hydra db

