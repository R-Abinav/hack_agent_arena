import os
import json
import time
from hydra_db import HydraDB
from dotenv import load_dotenv

load_dotenv()

def index_docs():
    api_key = os.environ.get("HYDRA_DB_API_KEY") or os.environ.get("HYDRA_DB_KEY")
    base_url = os.environ.get("HYDRA_DB_URL")
    experiment = os.environ.get("APPWORLD_EXPERIMENT", "silvanites")
    tenant_id = f"appworld_{experiment}"

    if not api_key:
        print("Error: HYDRA_DB_API_KEY or HYDRA_DB_KEY not set.")
        return

    client = HydraDB(token=api_key, base_url=base_url)

    # 1. Create tenant
    print(f"Creating tenant: {tenant_id}")
    try:
        client.tenants.create(tenant_id=tenant_id)
    except Exception as e:
        print(f"Tenant might already exist: {e}")

    # 2. Wait until ready
    print("Waiting for tenant readiness...")
    while True:
        try:
            status = client.tenants.status(tenant_id=tenant_id)
            if status.data.infra.ready_for_ingestion:
                break
        except Exception as e:
            print(f"Error checking status: {e}")
        time.sleep(5)
    print("Tenant is ready.")

    # 3. Ingest documents
    docs_dir = "data/api_docs"
    ingest_ids = []

    for root, dirs, files in os.walk(docs_dir):
        for file in files:
            if file.endswith(".json"):
                file_path = os.path.join(root, file)
                print(f"Ingesting {file_path}...")
                with open(file_path, "rb") as f:
                    try:
                        response = client.context.ingest(
                            type="knowledge",
                            tenant_id=tenant_id,
                            documents=[
                                (file, f, "application/json")
                            ],
                            document_metadata=json.dumps([
                                {
                                    "id": f"doc_{file_path.replace('/', '_')}",
                                    "title": f"API Doc: {file}",
                                    "additional_metadata": {
                                        "path": file_path,
                                        "category": os.path.basename(root)
                                    }
                                }
                            ])
                        )
                        ingest_ids.append(response.data.results[0].id)
                    except Exception as e:
                        print(f"Error ingesting {file_path}: {e}")

    # 4. Poll indexing status
    print(f"Polling status for {len(ingest_ids)} documents...")
    if not ingest_ids:
        print("No documents ingested.")
        return

    # Check first and last for simplicity, or just wait a bit
    # To be thorough, check all
    pending_ids = set(ingest_ids)
    while pending_ids:
        # Check in batches of 10
        current_ids = list(pending_ids)[:10]
        try:
            status_resp = client.context.status(
                tenant_id=tenant_id,
                ids=current_ids
            )
            for status in status_resp.data.statuses:
                if status.indexing_status in ("graph_creation", "completed"):
                    pending_ids.remove(status.id)
                elif status.indexing_status in ("errored", "failed"):
                    print(f"Indexing failed for {status.id}")
                    pending_ids.remove(status.id)
        except Exception as e:
            print(f"Error checking ingestion status: {e}")
        
        if pending_ids:
            print(f"Still waiting for {len(pending_ids)} documents...")
            time.sleep(5)

    print("All documents indexed successfully.")

if __name__ == "__main__":
    index_docs()
