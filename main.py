"""
BioFactor SCADA – FastAPI backend
"""

import asyncio
import random
import math
from datetime import datetime, timedelta
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy import select, desc, func, and_
from sqlalchemy.ext.asyncio import AsyncSession

from database import (
    init_db, AsyncSessionLocal,
    Lote, LecturaSensor, Alerta, WorkflowLog, ProduccionDiaria,
    LoteStatus, AlertSeverity, WorkflowStatus, StageEnum
)
from agent import ScadaAgent, WORKFLOWS, THRESHOLDS

# ─── LIFESPAN ─────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    # Start background sensor simulator
    task = asyncio.create_task(sensor_simulator())
    task_agent = asyncio.create_task(agent_loop())
    yield
    task.cancel()
    task_agent.cancel()

app = FastAPI(title="BioFactor SCADA API", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

agent = ScadaAgent()

# ─── BACKGROUND TASKS ─────────────────────────────────────────────────────────

_temp_state = {}
_hum_state  = {}
_nh3_state  = {}

async def sensor_simulator():
    """Generates realistic sensor readings every 10 seconds."""
    while True:
        try:
            async with AsyncSessionLocal() as session:
                result = await session.execute(
                    select(Lote).where(Lote.status.in_([LoteStatus.ACTIVO, LoteStatus.ALERTA]))
                )
                lotes = result.scalars().all()
                for lote in lotes:
                    lid = lote.id
                    # Maintain running state per lote
                    base_temp = 29.5 if lote.modulo == "MOD-01" else 31.0
                    if lid not in _temp_state:
                        _temp_state[lid] = base_temp
                        _hum_state[lid]  = 66.0
                        _nh3_state[lid]  = 38.0 + lote.dia_actual * 0.3

                    _temp_state[lid] = max(24, min(36, _temp_state[lid] + random.gauss(0, 0.15)))
                    _hum_state[lid]  = max(45, min(88, _hum_state[lid]  + random.gauss(0, 0.3)))
                    _nh3_state[lid]  = max(10, min(65, _nh3_state[lid]  + random.gauss(0.05, 0.4)))

                    r = LecturaSensor(
                        lote_id=lote.id,
                        temperatura=round(_temp_state[lid], 2),
                        humedad=round(_hum_state[lid], 2),
                        nh3=round(_nh3_state[lid], 2),
                        co2=round(900 + lote.dia_actual * 15 + random.gauss(0, 20), 1),
                        ph_sustrato=round(6.8 + random.gauss(0, 0.1), 2),
                        masa_larva_g=round(lote.kg_larva_real * 1000 / max(1, lote.dia_actual) * lote.dia_actual, 1),
                        etapa=lote.etapa_actual,
                        modulo=lote.modulo
                    )
                    session.add(r)
                await session.commit()
        except Exception as e:
            print(f"[simulator] error: {e}")
        await asyncio.sleep(10)

async def agent_loop():
    """Runs the AI agent analysis every 30 seconds."""
    await asyncio.sleep(5)
    while True:
        try:
            findings = await agent.analyze_latest_readings()
            for finding in findings:
                for wf_key in finding.get("suggested_workflows", []):
                    # Auto-execute non-critical workflows
                    reading = finding.get("reading", {})
                    await agent.execute_workflow(
                        wf_key,
                        finding["lote"],
                        {**reading, "modulo": finding["modulo"], "triggered_by": "agente_ia_auto"}
                    )
        except Exception as e:
            print(f"[agent_loop] error: {e}")
        await asyncio.sleep(30)

# ─── PYDANTIC SCHEMAS ─────────────────────────────────────────────────────────

class LoteCreate(BaseModel):
    codigo: str
    kg_entrada: float = 1000.0
    modulo: str = "MOD-01"
    notas: str = ""

class LecturaSensorCreate(BaseModel):
    lote_id: int
    temperatura: float
    humedad: float
    nh3: float
    co2: float
    ph_sustrato: float
    masa_larva_g: float
    etapa: str
    modulo: str

class QueryRequest(BaseModel):
    query: str

class WorkflowRequest(BaseModel):
    workflow_key: str
    lote_codigo: str
    input_data: dict = {}

class AlertaUpdate(BaseModel):
    resuelta: bool
    accion_tomada: str = ""

# ─── ROUTES ───────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def root():
    with open("index.html") as f:
        return f.read()

# --- Dashboard summary ---
@app.get("/api/dashboard")
async def get_dashboard():
    async with AsyncSessionLocal() as session:
        # All lotes
        lotes_r = await session.execute(select(Lote).order_by(desc(Lote.created_at)))
        lotes = lotes_r.scalars().all()

        # Active alerts (unresolved)
        alerts_r = await session.execute(
            select(func.count()).select_from(Alerta).where(Alerta.resuelta == False)
        )
        active_alerts = alerts_r.scalar()

        # Latest reading per lote
        lote_data = []
        for lote in lotes:
            lr = await session.execute(
                select(LecturaSensor)
                .where(LecturaSensor.lote_id == lote.id)
                .order_by(desc(LecturaSensor.timestamp))
                .limit(1)
            )
            reading = lr.scalar_one_or_none()
            lote_data.append({
                "id": lote.id,
                "codigo": lote.codigo,
                "dia_actual": lote.dia_actual,
                "dias_ciclo": lote.dias_ciclo,
                "etapa": lote.etapa_actual,
                "status": lote.status,
                "modulo": lote.modulo,
                "kg_entrada": lote.kg_entrada,
                "kg_larva_proy": lote.kg_larva_proy,
                "kg_larva_real": lote.kg_larva_real,
                "kg_frass_proy": lote.kg_frass_proy,
                "kg_frass_real": lote.kg_frass_real,
                "rendimiento_pct": round((lote.kg_larva_real / lote.kg_larva_proy) * 100, 1),
                "notas": lote.notas,
                "ultimo_lectura": {
                    "temperatura": reading.temperatura if reading else None,
                    "humedad": reading.humedad if reading else None,
                    "nh3": reading.nh3 if reading else None,
                    "co2": reading.co2 if reading else None,
                    "ph_sustrato": reading.ph_sustrato if reading else None,
                    "timestamp": reading.timestamp.isoformat() if reading else None
                } if reading else None
            })

        # Recent workflows
        wf_r = await session.execute(
            select(WorkflowLog).order_by(desc(WorkflowLog.timestamp)).limit(5)
        )
        workflows = wf_r.scalars().all()

        # Thresholds for UI
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "active_alerts": active_alerts,
            "lotes": lote_data,
            "thresholds": THRESHOLDS,
            "recent_workflows": [
                {"id": w.id, "name": w.workflow_name, "lote": w.lote_codigo,
                 "status": w.status, "ts": w.timestamp.isoformat()} for w in workflows
            ]
        }

# --- Lotes CRUD ---
@app.get("/api/lotes")
async def get_lotes():
    async with AsyncSessionLocal() as session:
        r = await session.execute(select(Lote).order_by(desc(Lote.created_at)))
        lotes = r.scalars().all()
        return [{"id": l.id, "codigo": l.codigo, "dia_actual": l.dia_actual,
                 "etapa": l.etapa_actual, "status": l.status, "modulo": l.modulo,
                 "kg_larva_real": l.kg_larva_real, "kg_frass_real": l.kg_frass_real,
                 "rendimiento_pct": round((l.kg_larva_real / l.kg_larva_proy) * 100, 1)} for l in lotes]

@app.post("/api/lotes", status_code=201)
async def create_lote(data: LoteCreate):
    async with AsyncSessionLocal() as session:
        lote = Lote(
            codigo=data.codigo,
            fecha_inicio=datetime.utcnow(),
            kg_entrada=data.kg_entrada,
            modulo=data.modulo,
            notas=data.notas
        )
        session.add(lote)
        await session.commit()
        return {"id": lote.id, "codigo": lote.codigo, "status": "created"}

@app.get("/api/lotes/{lote_id}")
async def get_lote(lote_id: int):
    async with AsyncSessionLocal() as session:
        lote = await session.get(Lote, lote_id)
        if not lote:
            raise HTTPException(404, "Lote no encontrado")
        return {"id": lote.id, "codigo": lote.codigo, "dia_actual": lote.dia_actual,
                "etapa": lote.etapa_actual, "status": lote.status, "modulo": lote.modulo,
                "kg_entrada": lote.kg_entrada, "kg_larva_proy": lote.kg_larva_proy,
                "kg_larva_real": lote.kg_larva_real, "kg_frass_proy": lote.kg_frass_proy,
                "kg_frass_real": lote.kg_frass_real, "notas": lote.notas,
                "fecha_inicio": lote.fecha_inicio.isoformat()}

# --- Sensor Readings ---
@app.get("/api/lecturas")
async def get_lecturas(
    lote_id: Optional[int] = None,
    modulo: Optional[str] = None,
    limit: int = Query(default=50, le=500),
    hours: int = Query(default=24, le=720)
):
    async with AsyncSessionLocal() as session:
        q = select(LecturaSensor).where(
            LecturaSensor.timestamp >= datetime.utcnow() - timedelta(hours=hours)
        )
        if lote_id:
            q = q.where(LecturaSensor.lote_id == lote_id)
        if modulo:
            q = q.where(LecturaSensor.modulo == modulo)
        q = q.order_by(desc(LecturaSensor.timestamp)).limit(limit)
        r = await session.execute(q)
        readings = r.scalars().all()
        return [{"id": r.id, "lote_id": r.lote_id, "modulo": r.modulo,
                 "timestamp": r.timestamp.isoformat(), "temperatura": r.temperatura,
                 "humedad": r.humedad, "nh3": r.nh3, "co2": r.co2,
                 "ph_sustrato": r.ph_sustrato, "masa_larva_g": r.masa_larva_g} for r in readings]

@app.post("/api/lecturas", status_code=201)
async def create_lectura(data: LecturaSensorCreate):
    async with AsyncSessionLocal() as session:
        r = LecturaSensor(**data.model_dump())
        session.add(r)
        await session.commit()
        return {"id": r.id, "status": "created"}

@app.get("/api/lecturas/stats")
async def get_stats(
    modulo: Optional[str] = None,
    hours: int = Query(default=24, le=720)
):
    async with AsyncSessionLocal() as session:
        q = select(
            LecturaSensor.modulo,
            func.round(func.avg(LecturaSensor.temperatura), 2).label("temp_avg"),
            func.round(func.min(LecturaSensor.temperatura), 2).label("temp_min"),
            func.round(func.max(LecturaSensor.temperatura), 2).label("temp_max"),
            func.round(func.avg(LecturaSensor.humedad), 2).label("hum_avg"),
            func.round(func.max(LecturaSensor.nh3), 2).label("nh3_max"),
            func.round(func.avg(LecturaSensor.nh3), 2).label("nh3_avg"),
            func.count(LecturaSensor.id).label("total_lecturas")
        ).where(LecturaSensor.timestamp >= datetime.utcnow() - timedelta(hours=hours))
        if modulo:
            q = q.where(LecturaSensor.modulo == modulo)
        q = q.group_by(LecturaSensor.modulo)
        r = await session.execute(q)
        rows = r.all()
        return [dict(r._mapping) for r in rows]

# --- Alerts ---
@app.get("/api/alertas")
async def get_alertas(resuelta: Optional[bool] = None, limit: int = 50):
    async with AsyncSessionLocal() as session:
        q = select(Alerta, Lote.codigo.label("lote_codigo")) \
            .join(Lote, Alerta.lote_id == Lote.id, isouter=True) \
            .order_by(desc(Alerta.timestamp)).limit(limit)
        if resuelta is not None:
            q = q.where(Alerta.resuelta == resuelta)
        r = await session.execute(q)
        rows = r.all()
        return [{
            "id": row.Alerta.id, "lote": row.lote_codigo,
            "severidad": row.Alerta.severidad, "variable": row.Alerta.variable,
            "mensaje": row.Alerta.mensaje, "valor_actual": row.Alerta.valor_actual,
            "valor_limite": row.Alerta.valor_limite, "resuelta": row.Alerta.resuelta,
            "accion_tomada": row.Alerta.accion_tomada,
            "timestamp": row.Alerta.timestamp.isoformat()
        } for row in rows]

@app.patch("/api/alertas/{alert_id}")
async def update_alerta(alert_id: int, data: AlertaUpdate):
    async with AsyncSessionLocal() as session:
        alerta = await session.get(Alerta, alert_id)
        if not alerta:
            raise HTTPException(404, "Alerta no encontrada")
        alerta.resuelta = data.resuelta
        alerta.accion_tomada = data.accion_tomada
        await session.commit()
        return {"status": "updated"}

# --- Workflows ---
@app.get("/api/workflows/available")
async def get_workflows_available():
    return [{"key": k, "name": v["name"], "trigger": v["trigger"], "steps": v["steps"]}
            for k, v in WORKFLOWS.items()]

@app.post("/api/workflows/execute")
async def execute_workflow(data: WorkflowRequest):
    result = await agent.execute_workflow(data.workflow_key, data.lote_codigo, data.input_data)
    return result

@app.get("/api/workflows/history")
async def get_workflow_history(limit: int = 50):
    async with AsyncSessionLocal() as session:
        r = await session.execute(
            select(WorkflowLog).order_by(desc(WorkflowLog.timestamp)).limit(limit)
        )
        rows = r.scalars().all()
        return [{"id": w.id, "workflow": w.workflow_name, "lote": w.lote_codigo,
                 "status": w.status, "triggered_by": w.triggered_by,
                 "input": w.input_data, "result": w.result,
                 "duration_ms": w.duration_ms, "timestamp": w.timestamp.isoformat()} for w in rows]

# --- AI Agent ---
@app.post("/api/agent/analyze")
async def agent_analyze():
    findings = await agent.analyze_latest_readings()
    return {"timestamp": datetime.utcnow().isoformat(), "findings": findings}

@app.post("/api/agent/query")
async def agent_query(data: QueryRequest):
    result = await agent.run_query(data.query)
    return result

# --- Production data ---
@app.get("/api/produccion")
async def get_produccion(lote_codigo: Optional[str] = None):
    async with AsyncSessionLocal() as session:
        q = select(ProduccionDiaria).order_by(ProduccionDiaria.fecha)
        if lote_codigo:
            q = q.where(ProduccionDiaria.lote_codigo == lote_codigo)
        r = await session.execute(q)
        rows = r.scalars().all()
        return [{"fecha": p.fecha.isoformat(), "lote": p.lote_codigo, "dia": p.dia_ciclo,
                 "kg_larva": p.kg_larva_acum, "kg_frass": p.kg_frass_acum,
                 "temp_prom": p.temp_promedio, "hum_prom": p.hum_promedio,
                 "nh3_max": p.nh3_max, "alertas": p.alertas_count,
                 "rendimiento_pct": p.rendimiento_pct} for p in rows]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
