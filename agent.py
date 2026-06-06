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
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "ollama")
MODEL = os.environ.get("MODEL", "")
DATASET = os.environ.get("APPWORLD_DATASET", "dev")          # dev | test_normal | test_challenge
EXPERIMENT = os.environ.get("APPWORLD_EXPERIMENT", "silvanites")
MAX_INTERACTIONS = int(os.environ.get("MAX_INTERACTIONS", "50"))
MAX_TASKS = int(os.environ.get("MAX_TASKS", "2"))           #fafo we do!
USE_HYDRA = os.environ.get("USE_HYDRA", "false").lower() == "true"

llm = get_llm(provider=LLM_PROVIDER, model=MODEL)

SYSTEM_PROMPT = """You are a Python coding agent in AppWorld (silvanites).
Your response MUST be EXACTLY one Python code block. 
FORBIDDEN: NO explanations, NO comments, NO conversational text, NO markdown outside the block.
FORBIDDEN: NO hypothetical libraries (e.g. MelodyMatcher).

STRICT GROUND RULES:
1. **ONLY CODE**: Your response is executed directly. Any text will cause a SyntaxError.
2. **NO GUESSING**: You MUST call `apis.api_docs.show_api_doc` for an API before using it.
3. **AUTH FIRST**: You MUST call `apis.supervisor.show_account_passwords()` to get credentials.
4. **LOGIN**: You MUST call `apis.<app>.login(...)` and store the `access_token` in a variable.
5. **PERSISTENCE**: Variables (tokens, data) PERSIST between turns. DO NOT re-login if you already have a token.

PHASED STRATEGY:
PHASE 1 (CONTEXT): Call `apis.supervisor.show_account_passwords()` and `apis.api_docs.show_app_descriptions()`.
PHASE 2 (GROUNDING): Call `apis.api_docs.show_api_descriptions(app_name='...')` and `apis.api_docs.show_api_doc(...)`.
PHASE 3 (EXECUTION): Call `apis.<app>.login(...)` and then the action APIs.
PHASE 4 (COMPLETION): Call `apis.supervisor.complete_task()`.
"""


def call_llm(messages: list[dict], system_prompt: str = SYSTEM_PROMPT, max_retries=3) -> str:
    for attempt in range(max_retries):
        try:
            return llm.call(messages, system_prompt)
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


# ---- colors ---------------------------------------------------------------
C_GREEN = "\033[92m"
C_RED = "\033[91m"
C_YELLOW = "\033[93m"
C_BLUE = "\033[94m"
C_CYAN = "\033[96m"
C_RESET = "\033[0m"

def spell_check(instruction: str, hydra_client=None, tenant_id=None) -> str:
    print(f"{C_BLUE}  > Performing doc-aware spell check on instruction...{C_RESET}")
    
    if not instruction:
        print("  > Instruction is empty, skipping spell check.")
        return instruction

    doc_context = ""
    sub_tenant_id = "7kzhuidiiw"
    
    if hydra_client and tenant_id:
        try:
            print(f"  > Fetching doc context for spell check from HydraDB...")
            res = hydra_client.query(
                tenant_id=tenant_id,
                sub_tenant_id=sub_tenant_id,
                query=instruction,
                type="knowledge",
                query_by="hybrid",
                mode="thinking"
            )
            if hasattr(res, 'data') and res.data and hasattr(res.data, 'chunks'):
                top_chunks = res.data.chunks[:5]
                context_parts = [f"Doc Snippet: {chunk.chunk_content}" for chunk in top_chunks]
                doc_context = "\n".join(context_parts)
        except Exception as e:
            print(f"{C_RED}  ! HydraDB query error during spell check: {e}{C_RESET}")

    spell_check_prompt = [
        {
            "role": "user", 
            "content": (
                f"STRICT TASK: Fix spelling and grammar errors in the instruction below. "
                f"DO NOT add new information. DO NOT rewrite the task. DO NOT change the technical meaning. "
                f"Use the API context ONLY to correct typos in app/API names.\n\n"
                f"API Context:\n{doc_context}\n\n"
                f"Instruction: {instruction}\n\n"
                f"Return ONLY the corrected string."
            )
        }
    ]
    
    corrected = call_llm(spell_check_prompt, "You are a conservative technical editor. Fix typos only.")
    corrected = corrected.strip().strip('"').strip("'")
    if not corrected or len(corrected) > len(instruction) * 1.5:
        print(f"  > Corrected Instruction looks suspicious or empty, using original.")
        return instruction
    print(f"  > Corrected Instruction: {corrected}")
    return corrected

def get_error_fix_from_docs(error_output: str, failing_code: str, hydra_client, tenant_id) -> str:
    print(f"{C_BLUE}  > Searching for a fix in API docs for the error...{C_RESET}")
    if not hydra_client or not tenant_id:
        return ""
    
    sub_tenant_id = "7kzhuidiiw"
    query = f"Fix this error: {error_output}\nFailing code: {failing_code}"
    
    if not query or query.strip() == "Fix this error: \nFailing code:":
        print("  > Query for error fix is empty, skipping.")
        return ""

    doc_context = ""
    try:
        res = hydra_client.query(
            tenant_id=tenant_id,
            sub_tenant_id=sub_tenant_id,
            query=query,
            type="knowledge",
            query_by="hybrid",
            mode="thinking"
        )
        if hasattr(res, 'data') and res.data and hasattr(res.data, 'chunks'):
            top_chunks = res.data.chunks[:5]
            context_parts = [f"Doc Snippet: {chunk.chunk_content}" for chunk in top_chunks]
            doc_context = "\n".join(context_parts)
    except Exception as e:
        print(f"{C_RED}  ! HydraDB query error during error fix search: {e}{C_RESET}")
        
    return doc_context

def query_hydra(hydra_client, tenant_id, sub_tenant_id, query_text, app_hint=None):
    if not hydra_client or not tenant_id:
        return ""
    
    queries = [query_text]
    if app_hint:
        queries.append(f"API documentation for {app_hint}")
    
    all_context = []
    seen_chunks = set()
    
    for q in queries:
        try:
            print(f"{C_BLUE}  > HITTING HYDRA DB: query='{q[:50]}...'{C_RESET}")
            res = hydra_client.query(
                tenant_id=tenant_id,
                sub_tenant_id=sub_tenant_id,
                query=q,
                type="all",
                query_by="hybrid",
                mode="thinking"
            )
            if hasattr(res, 'data') and res.data and hasattr(res.data, 'chunks'):
                new_chunks = 0
                for chunk in res.data.chunks[:5]:
                    if chunk.chunk_id not in seen_chunks:
                        all_context.append(f"Context: {chunk.chunk_content}")
                        seen_chunks.add(chunk.chunk_id)
                        new_chunks += 1
                if new_chunks > 0:
                    print(f"{C_GREEN}  > Retrieved {new_chunks} NEW context chunks from HydraDB for '{q[:20]}'.{C_RESET}")
        except Exception as e:
            print(f"{C_RED}  ! HydraDB query error for '{q[:20]}': {e}{C_RESET}")
            
    if all_context:
        return "\nRelevant Context (Memory & API Docs):\n" + "\n".join(all_context) + "\n"
    return ""

def solve(world: AppWorld, hydra_client=None, tenant_id=None, sub_tenant_id=None) -> None:
    # Use provided sub_tenant_id or default to the silvanites doc index
    sub_tenant_id = sub_tenant_id or os.environ.get("HYDRA_SUB_TENANT_ID", "7kzhuidiiw")
    tenant_id = tenant_id or os.environ.get("HYDRA_TENANT_ID", "appworld_silvanites")

    # 0. Spell check the instruction
    instruction = spell_check(world.task.instruction, hydra_client, tenant_id)

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
    
    # Extract app names from init_output for a better query
    app_hint = None
    if "=== APP DESCRIPTIONS ===" in str(init_output):
        # Rough extraction of mentioned apps
        apps = ["spotify", "gmail", "amazon", "todoist", "venmo", "splitwise", "phone", "simple_note"]
        mentioned = [a for a in apps if a in str(init_output).lower()]
        if mentioned:
            app_hint = ", ".join(mentioned)

    # 2. Initial Hydra Query with app hints
    hydra_context = query_hydra(hydra_client, tenant_id, sub_tenant_id, instruction, app_hint=app_hint)

    messages = [{
        "role": "user",
        "content": (
            f"Task: {instruction}\n\n"
            f"{hydra_context}\n"
            "Step 1: Discover. Call `apis.supervisor.show_account_passwords()` and `apis.api_docs.show_app_descriptions()` now."
        ),
    }]
    
    trajectory = []
    
    for step in range(MAX_INTERACTIONS):
        reply = call_llm(messages)
        print(f"{C_CYAN}  --- Step {step+1} LLM Reply ---{C_RESET}\n{reply}\n-----------------------")
        code = extract_code(reply)
        print(f"{C_YELLOW}  --- Step {step+1} Executing Code ---{C_RESET}\n{code}\n-----------------------")
        output = world.execute(code)
        
        trajectory.append({"step": step, "code": code, "output": output})
        print(f"  step {step+1}: ran {len(code)} chars -> {str(output)[:120]!r}")
        
        messages.append({"role": "assistant", "content": reply})
        
        # 3. Error parsing and fallback logic
        if "Exception:" in str(output) or "Traceback" in str(output) or "Error:" in str(output) or "SyntaxError" in str(output):
            print(f"{C_RED}  ! Error detected at step {step+1}. Hitting HydraDB for solution...{C_RESET}")
            # Use the new specialized error fix function
            error_context = get_error_fix_from_docs(str(output), code, hydra_client, tenant_id)
            if not error_context:
                error_context = query_hydra(hydra_client, tenant_id, sub_tenant_id, f"Error: {output}\nTask: {instruction}", app_hint=app_hint)
            
            error_msg = (
                f"Execution failed with the following error output:\n{output}\n\n"
                f"{error_context}\n"
                "IMPORTANT: You hallucinated an API name or used it incorrectly. "
                "The only way to interact with the environment is through the pre-loaded `apis` object. "
                "DO NOT try to import other libraries or hallucinate functions. "
                "Check the API docs provided in the context for the correct names and parameters."
            )
            messages.append({"role": "user", "content": error_msg})
        else:
            messages.append({"role": "user", "content": f"Execution output:\n{output}"})
            
        if world.task_completed():
            print(f"{C_GREEN}  ✓ task_completed{C_RESET}")
            break
    else:
        print(f"{C_RED}  ✗ hit MAX_INTERACTIONS without completion{C_RESET}")
        
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

            summary_prompt = f"""Analyze this trajectory for Task: "{instruction}" (Status: {status}).
Distill it into a UNIVERSAL TECHNICAL PATTERN for the agent's global brain.

CRITICAL INSTRUCTION: 
- DO NOT include specific names, phone numbers, emails, or values found in the data.
- DO NOT include the "answer" to the task.
- FOCUS ONLY on API syntax, parameter names, error recovery steps, and logical workflows.

If any errors occurred and were fixed, follow this mandatory logging format for each error:
Error: <error_msg>
Fix: <How to fix error>
Doc: <The specific API documentation or parameter rule that was violated/corrected>

Format the final response as:
[CORRECT API PATTERN]
- <Which API was called and with what argument types/keys?>
- <Logic: e.g., 'Must login to Spotify before calling show_library'>

[ERROR FIX LOGS]
(Repeat for each error fixed)
Error: ...
Fix: ...
Doc: ...

[UNIVERSAL RULE]
- <A general rule for this API/App that applies to ALL users>
"""
            lesson_learned = llm.call([{"role": "user", "content": summary_prompt}], "You are an expert AppWorld debugger.")
            
            learning_payload = [{
                "id": f"task_{world.task.id}",
                "text": f"Task: {instruction}\nStatus: {status}\nLesson Learned: {lesson_learned}",
                "infer": True
            }]
            
            # Also ingest specific error fixes if they happened, following the requested format
            for i in range(1, len(trajectory)):
                out_prev = str(trajectory[i-1]["output"])
                out_curr = str(trajectory[i]["output"])
                if any(x in out_prev for x in ["Exception", "Error", "Traceback", "SyntaxError"]):
                    if not any(x in out_curr for x in ["Exception", "Error", "Traceback", "SyntaxError"]):
                        # Request a concise Error/Fix/Doc summary for this specific step
                        step_summary_prompt = f"""Distill this specific error fix into the following format:
Error: <error_msg>
Fix: <How to fix error>
Doc: <For api doc>

Trajectory step {i}:
Failed Code: {trajectory[i-1]['code']}
Error: {out_prev}
Fixed Code: {trajectory[i]['code']}
Fixed Output: {out_curr}
"""
                        step_lesson = llm.call([{"role": "user", "content": step_summary_prompt}], "You are an expert AppWorld debugger.")
                        
                        learning_payload.append({
                            "id": f"fix_{world.task.id}_{i}",
                            "text": step_lesson,
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
    # Prioritize HYDRA_TENANT_ID, fallback to appworld_silvanites if docs were indexed there
    tenant_id = os.environ.get("HYDRA_TENANT_ID", "appworld_silvanites")
    sub_tenant_id = os.environ.get("HYDRA_SUB_TENANT_ID", "7kzhuidiiw")
    
    api_key = os.environ.get("HYDRA_DB_API_KEY") or os.environ.get("HYDRA_DB_KEY")
    base_url = os.environ.get("HYDRA_DB_URL")
    if USE_HYDRA and HydraDB and api_key:
        hydra_client = HydraDB(token=api_key, base_url=base_url)
        try:
            # Ensure tenant exists but don't fail if it does
            hydra_client.tenants.create(tenant_id=tenant_id)
        except Exception:
            pass

    task_id_override = os.environ.get("TASK_ID")
    if task_id_override:
        task_ids = [task_id_override]
    else:
        task_ids = load_task_ids(DATASET)
        if MAX_TASKS:
            task_ids = task_ids[:MAX_TASKS]
    
    print(f"Running '{EXPERIMENT}' on {len(task_ids)} tasks with {LLM_PROVIDER}/{MODEL}")
    
    for i, task_id in enumerate(task_ids, 1):
        print(f"[{i}/{len(task_ids)}] {task_id}")
        with AppWorld(task_id=task_id, experiment_name=EXPERIMENT) as world:
            try:
                solve(world, hydra_client, tenant_id, sub_tenant_id)
            except Exception as e:
                print(f"  ! error in task {task_id}: {e}")
                traceback.print_exc()
    
    print(f"\nDone. Outputs in ./experiments/outputs/{EXPERIMENT}/")
    print("Hand that folder to the organizers (or zip and submit per instructions).")


if __name__ == "__main__":
    main()
