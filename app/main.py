"""
FastAPI app — Multi-change request LLM pipeline for ABAP code modification.
🧠 Runs Relevance + Modifier Agents per structured change request (with code block name).
🗂️ Generates ONE text file only if relevant changes exist.
"""

import os
import io
import uuid
import logging
import json
from datetime import datetime
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from openai import OpenAI

# -------------------- CONFIG --------------------
load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("abap_multi_change")

app = FastAPI(title="ABAP Multi-Change Evaluator", version="4.0")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL = os.getenv("LLM_MODEL", "gpt-5")

jobs = {}  # in-memory job tracker

# -------------------- MODELS --------------------
class ChangeItem(BaseModel):
    CODE_BLOCK_NAME: str | None = None
    CHANGE: str


class CodeChangePayload(BaseModel):
    PGM_NAME: str | None = None
    INC_NAME: str | None = None
    TYPE: str | None = None
    NAME: str = Field(..., description="Program/include name used for file naming")
    START_LINE: str | None = None
    END_LINE: str | None = None
    CODE: str
    CHANGE_REQUEST: list[ChangeItem]


# -------------------- LLM CALL --------------------
def call_llm(system_prompt: str, user_prompt: str) -> str:
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.exception("LLM call failed: %s", e)
        raise HTTPException(status_code=500, detail=f"LLM API error: {e}")


# -------------------- RELEVANCE AGENT --------------------
def run_relevance_agent(code: str, change_item: ChangeItem):
    """Checks if the change applies to the given ABAP method."""
    system_prompt = (
        "You are an expert SAP ABAP code reviewer. "
        "You will determine whether a given change request applies to this ABAP method.\n\n"
        "STEP 1: Summarize the purpose of the ABAP method.\n"
        "STEP 2: Summarize what the change request (and its referenced code block name) intends to do.\n"
        "STEP 3: Mark relevance YES only if the change should logically be applied to this code.\n"
        "Output strictly in JSON format:\n"
        "{ 'purpose': '<summary>', 'change_intent': '<summary>', 'relevance': 'YES' or 'NO', 'reason': '<short reason>' }"
    )

    user_prompt = (
        f"CODE BLOCK NAME: {change_item.CODE_BLOCK_NAME or 'N/A'}\n"
        f"CHANGE REQUEST:\n{change_item.CHANGE}\n\n"
        f"ABAP CODE:\n{code}"
    )

    reply = call_llm(system_prompt, user_prompt)
    relevance, reason = "NO", "Parsing error"

    try:
        parsed = json.loads(reply.replace("'", '"'))
        relevance = parsed.get("relevance", "NO").upper()
        reason = parsed.get("reason", "").strip()
        logger.info(f"Relevance: {relevance} | Reason: {reason}")
    except Exception:
        if "YES" in reply.upper():
            relevance, reason = "YES", reply
        elif "NO" in reply.upper():
            relevance, reason = "NO", reply

    return relevance, reason


# -------------------- MODIFIER AGENT --------------------
def run_modifier_agent(code: str, change_item: ChangeItem):
    """Modifies the ABAP code as per the relevant change request."""
    system_prompt = (
        "You are an ABAP refactoring assistant. "
        "Given a program or method and a change request, modify the code to fully implement the change. "
        "Do not add explanations — output only the complete modified ABAP code. "
        "Add ABAP comments where changes are made: 'Added by PwC<YYYYMMDD>T<HHMMSS>'. "
        "Only apply the requested change. Always return the full ABAP code."
    )

    user_prompt = (
        f"CODE BLOCK NAME: {change_item.CODE_BLOCK_NAME or 'N/A'}\n"
        f"CHANGE REQUEST:\n{change_item.CHANGE}\n\n"
        f"EXISTING CODE:\n{code}"
    )

    return call_llm(system_prompt, user_prompt)


# -------------------- BACKGROUND JOB --------------------
def process_job(job_id: str, payload: CodeChangePayload):
    try:
        jobs[job_id] = {"status": "running", "results": []}
        current_code = payload.CODE
        applied_changes = []

        for idx, change_item in enumerate(payload.CHANGE_REQUEST, start=1):
            logger.info(f"[{job_id}] Evaluating change {idx}/{len(payload.CHANGE_REQUEST)}")

            relevance, reason = run_relevance_agent(current_code, change_item)
            result = {
                "index": idx,
                "relevance": relevance,
                "reason": reason,
                "code_block": change_item.CODE_BLOCK_NAME or ""
            }

            if relevance == "YES":
                modified_code = run_modifier_agent(current_code, change_item)
                result["modified_code"] = modified_code
                current_code = modified_code  # ✅ Feed new version to next iteration
                applied_changes.append(f"{idx} ({change_item.CODE_BLOCK_NAME or 'Unknown'})")
                logger.info(f"[{job_id}] Change {idx} applied successfully.")
            else:
                result["modified_code"] = ""
                logger.info(f"[{job_id}] Change {idx} not relevant, skipped.")

            jobs[job_id]["results"].append(result)

        # 🧩 No applicable changes
        if not applied_changes:
            jobs[job_id]["status"] = "completed"
            jobs[job_id]["result"] = {"status": "no_change_required"}
            logger.info(f"[{job_id}] No relevant changes. No file created.")
            return

        # ✅ Build final combined output
        header = (
            f"*=== FINAL MODIFIED CODE ===*\n"
            f"* Applied Changes: {', '.join(applied_changes)} *\n"
            f"* Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} *\n\n"
        )
        final_output = header + current_code

        filename = f"{payload.NAME}_modified_{datetime.now().strftime('%Y%m%dT%H%M%S')}.txt"
        buffer = io.BytesIO(final_output.encode("utf-8"))
        buffer.seek(0)

        jobs[job_id]["status"] = "completed"
        jobs[job_id]["result"] = {
            "status": "change_applied",
            "file_buffer": buffer,
            "filename": filename
        }
        logger.info(f"[{job_id}] Final file created: {filename}")

    except Exception as e:
        jobs[job_id] = {"status": "failed", "error": str(e)}
        logger.exception(f"[{job_id}] Job failed: %s", e)


# -------------------- ENDPOINTS --------------------
@app.post("/apply_change")
async def apply_change(payload: CodeChangePayload, background_tasks: BackgroundTasks):
    """Start background job for change evaluation."""
    job_id = str(uuid.uuid4())[:8]
    jobs[job_id] = {"status": "queued"}
    background_tasks.add_task(process_job, job_id, payload)
    logger.info(f"Job {job_id} queued for {payload.NAME}")
    return {"job_id": job_id, "status": "queued"}


@app.get("/apply_change/{job_id}")
async def job_status(job_id: str):
    """Poll job status or return file if completed."""
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job["status"] != "completed":
        return {"job_id": job_id, "status": job["status"]}

    result = job["result"]
    if result["status"] == "no_change_required":
        return JSONResponse({
            "job_id": job_id,
            "status": "completed",
            "message": "No relevant changes found — no file generated."
        })

    # Otherwise return the single modified text file
    return StreamingResponse(
        result["file_buffer"],
        media_type="text/plain",
        headers={"Content-Disposition": f"attachment; filename={result['filename']}"}
    )
