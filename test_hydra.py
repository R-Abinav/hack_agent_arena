import os
from dotenv import load_dotenv
from hydra_db import HydraDB

load_dotenv()

hydra = HydraDB(
    token=os.environ.get("HYDRA_DB_KEY"),
    base_url=os.environ.get("HYDRA_DB_URL")
)

HYDRA_TENANT = os.environ.get("HYDRA_TENANT_ID", "default-tenant")
SUB_TENANT_ID = os.environ.get("HYDRA_SUB_TENANT_ID", "agent_v1")

print("Querying HydraDB...")
try:
    result = hydra.query(
        tenant_id=HYDRA_TENANT,
        sub_tenant_id=SUB_TENANT_ID,
        query="dummy query",
        type="memory",
        query_by="hybrid",
        mode="thinking"
    )
    print(f"Result length: {len(str(result))}")
    print(str(result)[:500])
except Exception as e:
    print("Error:", e)
