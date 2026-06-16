import os
import json
from dotenv import load_dotenv
from hydra_db import HydraDB

# Colors for output
C_GREEN = "\033[92m"
C_RED = "\033[91m"
C_BLUE = "\033[94m"
C_RESET = "\033[0m"

def smoke_test():
    load_dotenv()
    
    api_key = os.environ.get("HYDRA_DB_API_KEY") or os.environ.get("HYDRA_DB_KEY")
    base_url = os.environ.get("HYDRA_DB_URL")
    tenant_id = "appworld_silvanites"
    doc_sub_tenant = "7kzhuidiiw"
    spell_sub_tenant = "7kzhuidiiw" # Testing same sub-tenant for now

    print(f"{C_BLUE}--- HydraDB Smoke Test ---{C_RESET}")
    print(f"Tenant ID: {tenant_id}")
    print(f"API Key present: {'Yes' if api_key else 'No'}")
    
    if not api_key:
        print(f"{C_RED}Error: HYDRA_DB_API_KEY missing in .env{C_RESET}")
        return

    client = HydraDB(token=api_key, base_url=base_url)

    # Test 1: Query API Docs (Primary Knowledge)
    print(f"\n{C_BLUE}Test 1: Querying API Docs (sub_tenant={doc_sub_tenant})...{C_RESET}")
    try:
        res = client.query(
            tenant_id=tenant_id,
            sub_tenant_id=doc_sub_tenant,
            query="Spotify login API",
            type="all",
            query_by="hybrid"
        )
        if hasattr(res, 'data') and res.data and hasattr(res.data, 'chunks') and res.data.chunks:
            print(f"{C_GREEN}✓ SUCCESS: Found {len(res.data.chunks)} chunks.{C_RESET}")
            print(f"Top Result Snippet: {res.data.chunks[0].chunk_content[:100]}...")
        else:
            print(f"{C_RED}✗ FAILURE: No chunks returned for API Docs query.{C_RESET}")
    except Exception as e:
        print(f"{C_RED}✗ ERROR: {e}{C_RESET}")

    # Test 2: Query Spell Check Context
    print(f"\n{C_BLUE}Test 2: Querying Spell Check context (sub_tenant={spell_sub_tenant})...{C_RESET}")
    try:
        res = client.query(
            tenant_id=tenant_id,
            sub_tenant_id=spell_sub_tenant,
            query="spotify tracks",
            type="knowledge",
            query_by="hybrid"
        )
        if hasattr(res, 'data') and res.data and hasattr(res.data, 'chunks') and res.data.chunks:
            print(f"{C_GREEN}✓ SUCCESS: Found {len(res.data.chunks)} chunks.{C_RESET}")
            print(f"Top Result Snippet: {res.data.chunks[0].chunk_content[:100]}...")
        else:
            print(f"{C_RED}! WARNING: No chunks returned for Spell Check query. This sub-tenant might be empty.{C_RESET}")
    except Exception as e:
        print(f"{C_RED}✗ ERROR: {e}{C_RESET}")

if __name__ == "__main__":
    smoke_test()
