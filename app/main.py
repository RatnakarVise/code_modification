"""
FastAPI app — Multi-change request LLM pipeline for ABAP code modification.
🧠 Runs Relevance + Modifier Agents per change request.
🗂️ Generates ONE text file only if relevant changes exist.
"""

import os
import io
import uuid
import logging
from datetime import datetime
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from openai import OpenAI
import json

# -------------------- CONFIG --------------------
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("abap_multi_change")

app = FastAPI(title="ABAP Multi-Change Evaluator", version="3.5")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL = os.getenv("LLM_MODEL", "gpt-5")

jobs = {}  # in-memory job tracker


# -------------------- REQUEST MODEL --------------------
class CodeChangePayload(BaseModel):
    PGM_NAME: str | None = None
    INC_NAME: str | None = None
    TYPE: str | None = None
    NAME: str = Field(..., description="Program/include name used for file naming")
    START_LINE: str | None = None
    END_LINE: str | None = None
    CODE: str
    CHANGE_REQUEST: list[str]


# -------------------- LLM HELPER --------------------
def call_llm(system_prompt: str, user_prompt: str) -> str:
    """Wrapper for OpenAI LLM call."""
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
def run_relevance_agent(code: str, change_request: str):
    system_prompt = (
        "You are an expert ABAP refactoring assistant. "
        "Determine whether the change logically applies to the given code. "
        "Respond strictly in JSON format: "
        "{ 'relevance': 'YES' or 'NO', 'reason': '<text>' }"
    )
    user_prompt = f"CHANGE REQUEST:\n{change_request}\n\nABAP CODE:\n{code}"

    reply = call_llm(system_prompt, user_prompt)
    relevance, reason = "NO", "Parsing error"

    try:
        parsed = json.loads(reply.replace("'", '"'))
        relevance = parsed.get("relevance", "NO").upper()
        reason = parsed.get("reason", "").strip()
    except Exception:
        if "YES" in reply.upper():
            relevance, reason = "YES", reply
        elif "NO" in reply.upper():
            relevance, reason = "NO", reply

    logger.info(f"RelevanceAgent → {relevance}: {reason}")
    return relevance, reason


# -------------------- MODIFIER AGENT --------------------
def run_modifier_agent(code: str, change_request: str):
    system_prompt = (
        "You are an ABAP refactoring assistant. "
        "Given a program or method and a change request, modify the code to fully implement the change. "
        "Do not add explanations — output only the complete modified ABAP code. "
        "Add ABAP comments where changes are made: 'Added by PwC<YYYYMMDD>T<HHMMSS>'. "
        "Only apply the requested change. Always return the full ABAP code."
    )
    user_prompt = f"CHANGE REQUEST:\n{change_request}\n\nEXISTING CODE:\n{code}"

    return call_llm(system_prompt, user_prompt)


# -------------------- BACKGROUND JOB --------------------
def process_job(job_id: str, payload: CodeChangePayload):
    try:
        jobs[job_id] = {"status": "running", "results": []}
        current_code = payload.CODE
        applied_changes = []

        for idx, change_req in enumerate(payload.CHANGE_REQUEST, start=1):
            logger.info(f"[{job_id}] Evaluating change {idx}/{len(payload.CHANGE_REQUEST)}")

            relevance, reason = run_relevance_agent(current_code, change_req)
            result = {"index": idx, "relevance": relevance, "reason": reason}

            if relevance == "YES":
                modified_code = run_modifier_agent(current_code, change_req)
                result["modified_code"] = modified_code
                current_code = modified_code  # ✅ update for next iteration
                applied_changes.append(str(idx))
                logger.info(f"[{job_id}] Change {idx} applied successfully.")
            else:
                result["modified_code"] = ""
                logger.info(f"[{job_id}] Change {idx} not relevant, skipped.")

            jobs[job_id]["results"].append(result)

        # 🧩 If no relevant changes
        if not applied_changes:
            jobs[job_id]["status"] = "completed"
            jobs[job_id]["result"] = {"status": "no_change_required"}
            logger.info(f"[{job_id}] No relevant changes. No file created.")
            return

        # ✅ Build final output — only final code with summary
        header = (
            f"*=== FINAL MODIFIED CODE ===*\n"
            f"* Applied Changes: {', '.join(applied_changes)} *\n"
            f"* Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} *\n\n"
        )
        final_output = header + current_code

        filename = f"{payload.NAME}_modified_{datetime.now().strftime('%Y%m%dT%H%M%S')}.txt"

        file_buffer = io.BytesIO(final_output.encode("utf-8"))
        file_buffer.seek(0)

        jobs[job_id]["status"] = "completed"
        jobs[job_id]["result"] = {
            "status": "change_applied",
            "file_buffer": file_buffer,
            "filename": filename
        }

        logger.info(f"[{job_id}] Final file {filename} created with cumulative changes.")

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


@app.get("/job_status/{job_id}")
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
        headers={
            "Content-Disposition": f"attachment; filename={result['filename']}"
        },
    )
