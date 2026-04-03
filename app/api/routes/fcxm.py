# api/routes/fcxm.py
import random
import uuid
from typing import Any, Dict, List, Optional, Literal

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/api/fcxm", tags=["fcxm"])



def run_fcxm_job(job_id: str, req: FCXMRunRequest):
    try:
        FCXM_JOB_PROGRESS[job_id] = {"status": "running", "message": "Running…"}

        print("\n=== FCXM RUN REQUEST ===")
        print("JOB:", job_id)
        print("\nSAMPLES:")
        for s in req.samples:
            print(f" - role={s.role} name={s.name} id={s.id}")
            for fp in s.file_paths:
                print("    file:", fp)

        print("\nPANEL ROWS:")
        for r in req.panel_rows:
            print(f" - channel={r.channel} role={r.role} antibody={r.antibody} population={r.population}")
        print("=== END FCXM RUN REQUEST ===\n")

        FCXM_JOB_RESULTS[job_id] = {"ok": True}
        FCXM_JOB_PROGRESS[job_id] = {"status": "done", "message": "Done."}

    except Exception as e:
        FCXM_JOB_PROGRESS[job_id] = {"status": "error", "message": str(e)}

@router.post("/run", response_model=FCXMRunStartResponse)
async def fcxm_run(req: FCXMRunRequest, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())
    FCXM_JOB_PROGRESS[job_id] = {"status": "queued", "message": "Queued."}
    background_tasks.add_task(run_fcxm_job, job_id, req)
    return {"job_id": job_id}

@router.get("/run/{job_id}", response_model=FCXMRunProgressResponse)
async def fcxm_run_progress(job_id: str):
    if job_id not in FCXM_JOB_PROGRESS:
        raise HTTPException(status_code=404, detail="Job not found")

    prog = FCXM_JOB_PROGRESS[job_id]
    res = FCXM_JOB_RESULTS.get(job_id)

    return {
        "status": prog.get("status", "queued"),
        "message": prog.get("message"),
        "result": res,
    }

def _seed_from(s: str) -> int:
    return sum(ord(c) for c in s) % (2**31 - 1)

def _make_cloud(rng: random.Random, n: int, cx: float, cy: float, sx: float, sy: float):
    return [(rng.gauss(cx, sx), rng.gauss(cy, sy)) for _ in range(n)]

@router.post("/results", response_model=FCXMResultsResponse)
async def fcxm_results(req: FCXMResultsRequest):
    rng = random.Random(_seed_from(req.fcs_filename + "|" + req.gate))
    jitter = (rng.randint(0, 22)) / 23.0

    gate_options = ["IgG+", "Live", "Lymph", "Singlets"]

    raw = [
        ("Gate 1: FSC/SSC", "FSC-A", "SSC-A", _make_cloud(rng, 900, 40+10*jitter, 30+8*jitter, 12, 10),
         lambda x,y: x > 45 and y < 38),
        ("Gate 2: Singlets", "FSC-A", "FSC-H", _make_cloud(rng, 900, 55+8*jitter, 55+6*jitter, 10, 10),
         lambda x,y: abs(x - y) < 8),
        ("Gate 3: Lymph", "CD45", "SSC-A", _make_cloud(rng, 900, 25+6*jitter, 45+10*jitter, 8, 14),
         lambda x,y: x > 25 and y > 40),
        ("Gate 4: Marker", "Marker A", "Marker B", _make_cloud(rng, 900, 60+12*jitter, 25+6*jitter, 14, 8),
         lambda x,y: x > 62 and y < 28),
        ("Gate 5: Marker", "Marker A", "Marker B", _make_cloud(rng, 900, 60+12*jitter, 25+6*jitter, 14, 8),
         lambda x,y: x > 62 and y < 28),
        ("Gate 6: Marker", "Marker A", "Marker B", _make_cloud(rng, 900, 60+12*jitter, 25+6*jitter, 14, 8),
         lambda x,y: x > 62 and y < 28),
    ]

    gating_plots: List[GatingPlot] = []
    for title, xl, yl, pts, fn in raw:
        tagged = [SimPoint(x=x, y=y, inGate=bool(fn(x, y))) for (x, y) in pts]
        gating_plots.append(GatingPlot(title=title, x_label=xl, y_label=yl, points=tagged))

    cut = 2.0 + 0.25 * jitter
    base = _make_cloud(rng, 1400, 2.2 + 0.5*jitter, 35 + 6*jitter, 0.55, 10)

    final_scatter: List[SimPoint] = []
    line_values: List[float] = []
    for x, y in base:
        ing = x >= cut
        final_scatter.append(SimPoint(x=x, y=y, inGate=ing))
        line_values.append(x)

    return FCXMResultsResponse(
        gate_options=gate_options,
        gating_plots=gating_plots,
        final_scatter=final_scatter,
        line_values=line_values,
        cutoff=cut,
    )

