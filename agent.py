"""
://agent_arena — AppWorld starter agent (ReAct code agent).

This is a WORKING template you can hack on. The loop and every AppWorld API
call below were verified against appworld==0.1.3. Your job is to make the agent
smarter: better prompting, planning, error recovery, retrieval, etc.

How AppWorld works (the rules your agent plays by):
  - Each task gives you a natural-language instruction from your "supervisor".
  - You act by writing PYTHON code. The env runs it and returns whatever you
    print(). A preloaded object `apis` is your only interface to the 9 apps.
  - Discover APIs at runtime:
        apis.api_docs.show_app_descriptions()
        apis.api_docs.show_api_descriptions(app_name='spotify')
        apis.api_docs.show_api_doc(app_name='spotify', api_name='login')
  - Get credentials to log into apps:
        apis.supervisor.show_account_passwords()
    (most app APIs need an access_token returned by that app's `login`).
  - Finish with:
        apis.supervisor.complete_task(answer=<answer or None>)
    Pass `answer` only when the task asks a question; otherwise leave it None.

Run:
  export ANTHROPIC_API_KEY=sk-...             # or put it in .env
  export APPWORLD_EXPERIMENT=team_<yourname>   # your unique team id
  export APPWORLD_DATASET=dev                  # dev while building; switch to the
                                               # official split at submission time
  python agent.py
"""

import os
import re
import time
import json
import traceback

try:  # optional: load ANTHROPIC_API_KEY etc. from a local .env
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

from appworld import AppWorld, load_task_ids
from provider import get_llm

try:
    from hydra_db import HydraDB
    from hydra_db.errors import TooManyRequestsError, InternalServerError, ServiceUnavailableError
except ImportError:
    HydraDB = None

# ---- config ---------------------------------------------------------------
MODEL = os.environ.get("MODEL", "groq/llama-3.3-70b-versatile")
DATASET = os.environ.get("APPWORLD_DATASET", "dev")          # dev | test_normal | test_challenge
EXPERIMENT = os.environ.get("APPWORLD_EXPERIMENT", "silvanites")
MAX_INTERACTIONS = int(os.environ.get("MAX_INTERACTIONS", "50"))
MAX_TASKS = int(os.environ.get("MAX_TASKS", "1"))            # 1 = run just one task by default
USE_HYDRA = os.environ.get("USE_HYDRA", "false").lower() == "true"

llm = get_llm()

SYSTEM_PROMPT = """You are an autonomous coding agent operating inside AppWorld (silvanites).
You complete the supervisor's task by writing Python code that the environment executes.

STRATEGY:
1. **Discover & Authenticate**: 
   - Use `apis.supervisor.show_account_passwords()` to get credentials.
   - Use `apis.api_docs.show_app_descriptions()` to find the right app.
   - Use `apis.api_docs.show_api_descriptions(app_name='...')` to find the login and relevant action APIs.
   - **MANDATORY**: Call the app's `login` API first to get an `access_token`. Store it in a variable.
2. **Execute Decisively**:
   - Don't just look at docs; once you see the API you need, CALL IT.
   - Use `apis.api_docs.show_api_doc(app_name='...', api_name='...')` if you are unsure of the exact parameters.
3. **Verify & Finish**:
   - Check the output of your actions.
   - Once the task is done, call `apis.supervisor.complete_task(answer=...)`.

RULES:
- Reply with EXACTLY ONE Python code block per turn, nothing else.
- Variables PERSIST across turns. RETAIN your `access_token`!
- **Learning from Memory**: Use the provided 'Relevant Memory' to avoid past mistakes and reuse successful patterns.
- **No Hallucination**: Only use APIs listed in the documentation.
- When and ONLY when the task is fully done, call `apis.supervisor.complete_task`.
"""


def call_llm(messages: list[dict], max_retries=3) -> str:
    for attempt in range(max_retries):
        try:
            return llm.call(messages, SYSTEM_PROMPT)
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)


def extract_code(text: str) -> str:
    # handle both ```python and ``` blocks
    m = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.S)
    if m:
        return m.group(1).strip()
    # fallback: if no backticks, look for any indented block or just return the whole thing
    # but the prompt says EXACTLY one block, so we'll be lenient
    return text.strip()


def solve(world: AppWorld, hydra_client=None, tenant_id=None) -> None:
    # We use a GLOBAL sub_tenant_id for the agent's "brain".
    # This ensures lessons learned in Task A (e.g., how to use Spotify API)
    # are immediately available for Task B, even if the supervisor is different.
    sub_tenant_id = "global_agent_memory"

    # 1. Deterministic Hard-Coded Workflow
    init_code = '''
try:
    print("=== ACCOUNTS & PASSWORDS ===")
    print(apis.supervisor.show_account_passwords())
    print("\\n=== APP DESCRIPTIONS ===")
    print(apis.api_docs.show_app_descriptions())
except Exception as e:
    print(f"Error fetching initial state: {e}")
'''
    init_output = world.execute(init_code)
    
    # 2. Query HydraDB for relevant past knowledge (Global retrieval)
    hydra_context = ""
    if hydra_client and tenant_id:
        try:
            print(f"  > Querying Global Memory: tenant={tenant_id}")
            res = hydra_client.query(
                tenant_id=tenant_id,
                sub_tenant_id=sub_tenant_id,
                query=world.task.instruction,
                type="all",
                query_by="hybrid",
                mode="thinking"
            )
            if hasattr(res, 'data') and res.data:
                hydra_context = f"\nRelevant Memory (Lessons from past runs):\n{res.data}\n"
        except Exception as e:
            print(f"  ! HydraDB query error: {e}")

    messages = [{
        "role": "user",
        "content": (
            f"Supervisor: {world.task.supervisor}\n\n"
            f"Task: {world.task.instruction}\n\n"
            f"Pre-fetched Initialization Output:\n{init_output}\n{hydra_context}\n"
            "Begin. Remember: one python code block per turn. Store tokens in variables."
        ),
    }]
    
    trajectory = []
    
    for step in range(MAX_INTERACTIONS):
        reply = call_llm(messages)
        print(f"  --- Step {step+1} LLM Reply ---\n{reply}\n-----------------------")
        code = extract_code(reply)
        print(f"  --- Step {step+1} Executing Code ---\n{code}\n-----------------------")
        output = world.execute(code)
        
        trajectory.append({"step": step, "code": code, "output": output})
        print(f"  step {step+1}: ran {len(code)} chars -> {str(output)[:120]!r}")
        
        messages.append({"role": "assistant", "content": reply})
        
        # 3. Error parsing and fallback logic
        if "Exception:" in str(output) or "Traceback" in str(output) or "Error:" in str(output) or "SyntaxError" in str(output):
            messages.append({
                "role": "user", 
                "content": f"Execution failed with the following error output:\n{output}\n\nIMPORTANT: You hallucinated an API name or used it incorrectly. Check the API docs for the correct name and parameters before trying again."
            })
        else:
            messages.append({"role": "user", "content": f"Execution output:\n{output}"})
            
        if world.task_completed():
            print("  ✓ task_completed")
            break
    else:
        print("  ✗ hit MAX_INTERACTIONS without completion")
        
    # 4. Ingest Trajectory to HydraDB (Learn from successes AND failures)
    if hydra_client and tenant_id:
        try:
            # Generate a post-mortem summary
            status = "SUCCESS" if world.task_completed() else "FAILURE"
            
            # Truncate trajectory outputs for the summary prompt
            compact_trajectory = []
            for t in trajectory:
                compact_trajectory.append({
                    "step": t["step"],
                    "code": t["code"],
                    "output": str(t["output"])[:500] + "..." if len(str(t["output"])) > 500 else str(t["output"])
                })

            summary_prompt = f"""Analyze this trajectory for Task: "{world.task.instruction}" (Status: {status}).
Distill it into a UNIVERSAL TECHNICAL PATTERN for the agent's global brain.

CRITICAL INSTRUCTION: 
- DO NOT include specific names, phone numbers, emails, or values found in the data.
- DO NOT include the "answer" to the task.
- FOCUS ONLY on API syntax, parameter names, error recovery steps, and logical workflows.

Format:
[CORRECT API PATTERN]
- <Which API was called and with what argument types/keys?>
- <Logic: e.g., 'Must login to Spotify before calling show_library'>

[ERROR AVOIDANCE]
- <The specific error/misconception (e.g., 'Thought field was "id", actually "song_id"')>
- <The exact code fix that solved it>

[UNIVERSAL RULE]
- <A general rule for this API/App that applies to ALL users>
"""
            lesson_learned = llm.call([{"role": "user", "content": summary_prompt}], "You are an expert AppWorld debugger.")
            
            learning_payload = [{
                "id": f"task_{world.task.id}",
                "text": f"Task: {world.task.instruction}\nStatus: {status}\nLesson Learned: {lesson_learned}",
                "infer": True
            }]
            
            # Also ingest specific error fixes if they happened
            for i in range(1, len(trajectory)):
                out_prev = str(trajectory[i-1]["output"])
                out_curr = str(trajectory[i]["output"])
                if any(x in out_prev for x in ["Exception", "Error", "Traceback", "SyntaxError"]):
                    if not any(x in out_curr for x in ["Exception", "Error", "Traceback", "SyntaxError"]):
                        learning_payload.append({
                            "id": f"fix_{world.task.id}_{i}",
                            "text": f"Error fix for task '{world.task.instruction}':\nFailed Code: {trajectory[i-1]['code']}\nError: {out_prev}\nFixed Code: {trajectory[i]['code']}",
                            "infer": True
                        })

            print(f"  > Ingesting {len(learning_payload)} lessons to HydraDB: tenant={tenant_id}, sub_tenant={sub_tenant_id}")
            resp = hydra_client.context.ingest(
                type="memory",
                tenant_id=tenant_id,
                sub_tenant_id=sub_tenant_id,
                memories=json.dumps(learning_payload)
            )
            print(f"  ✓ Ingestion response: {resp}")
        except Exception as e:
            print(f"  ! HydraDB ingestion error: {e}")


def main() -> None:
    # Initialize HydraDB
    hydra_client = None
    tenant_id = f"appworld_{EXPERIMENT}"
    api_key = os.environ.get("HYDRA_DB_API_KEY") or os.environ.get("HYDRA_DB_KEY")
    base_url = os.environ.get("HYDRA_DB_URL")
    if USE_HYDRA and HydraDB and api_key:
        hydra_client = HydraDB(token=api_key, base_url=base_url)
        try:
            hydra_client.tenants.create(tenant_id=tenant_id)
            print("Waiting for HydraDB tenant to be ready...")
            for _ in range(12):
                status = hydra_client.tenants.status(tenant_id=tenant_id)
                if status.data.infra.ready_for_ingestion:
                    print("HydraDB tenant is ready.")
                    break
                time.sleep(5)
        except Exception as e:
            print(f"HydraDB init warning: {e}")

    task_ids = load_task_ids(DATASET)
    if MAX_TASKS:
        task_ids = task_ids[:MAX_TASKS]
    print(f"Running '{EXPERIMENT}' on {len(task_ids)} '{DATASET}' tasks with {MODEL}")
    for i, task_id in enumerate(task_ids, 1):
        print(f"[{i}/{len(task_ids)}] {task_id}")
        with AppWorld(task_id=task_id, experiment_name=EXPERIMENT) as world:
            try:
                solve(world, hydra_client, tenant_id)
            except Exception as e:  # never let one task kill the whole run
                print(f"  ! error: {e}")
    print(f"\nDone. Outputs in ./experiments/outputs/{EXPERIMENT}/")
    print("Hand that folder to the organizers (or zip and submit per instructions).")


if __name__ == "__main__":
    main()
