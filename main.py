"""
BioFactor — Bioconversion OS (AI-native)
Backend FastAPI. El kernel cognitivo del agente corre como tarea de fondo;
los sensores se simulan; el dashboard consume estado + un feed en vivo (SSE).
"""

import asyncio
import os
import random
import math
import json
import io
from datetime import datetime, timedelta
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel
from sqlalchemy import select, desc, func, and_

from database import (
    init_db, AsyncSessionLocal, DB_BACKEND,
    Lote, LecturaSensor, Alerta, WorkflowLog, ProduccionDiaria,
    AgentDecision, Approval, AgentMemory, SystemConfig, Planta,
    LoteStatus, AlertSeverity, WorkflowStatus, StageEnum, ApprovalStatus,
)
from agent import ScadaAgent, WORKFLOWS, THRESHOLDS
import economics
import prediction
import events
import llm

# ─── LIFESPAN ─────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    t1 = asyncio.create_task(sensor_simulator())
    t2 = asyncio.create_task(kernel_loop())
    yield
    for t in (t1, t2):
        t.cancel()

app = FastAPI(title="BioFactor — Bioconversion OS", version="2.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

agent = ScadaAgent()

# ─── ROBUSTEZ: manejador global de errores (sin 500 desnudos) ──────────────────

@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    print(f"[error] {request.method} {request.url.path}: {exc!r}")
    return JSONResponse(status_code=500, content={
        "error": "internal_error",
        "detail": str(exc),
        "path": request.url.path,
    })

# ─── BACKGROUND TASKS ─────────────────────────────────────────────────────────

_temp_state, _hum_state, _nh3_state = {}, {}, {}

async def sensor_simulator():
    """Genera lecturas realistas cada 10s y las publica al feed en vivo."""
    while True:
        try:
            async with AsyncSessionLocal() as session:
                lotes = (await session.execute(
                    select(Lote).where(Lote.status.in_([LoteStatus.ACTIVO, LoteStatus.ALERTA]))
                )).scalars().all()
                snapshot = []
                for lote in lotes:
                    lid = lote.id
                    base_temp = 29.5 if lote.modulo == "MOD-01" else 31.0
                    if lid not in _temp_state:
                        _temp_state[lid] = base_temp
                        _hum_state[lid] = 66.0
                        _nh3_state[lid] = 38.0 + lote.dia_actual * 0.3
                    _temp_state[lid] = max(24, min(36, _temp_state[lid] + random.gauss(0, 0.15)))
                    _hum_state[lid] = max(45, min(88, _hum_state[lid] + random.gauss(0, 0.3)))
                    _nh3_state[lid] = max(10, min(65, _nh3_state[lid] + random.gauss(0.05, 0.4)))
                    r = LecturaSensor(
                        lote_id=lote.id, temperatura=round(_temp_state[lid], 2),
                        humedad=round(_hum_state[lid], 2), nh3=round(_nh3_state[lid], 2),
                        co2=round(900 + lote.dia_actual * 15 + random.gauss(0, 20), 1),
                        ph_sustrato=round(6.8 + random.gauss(0, 0.1), 2),
                        masa_larva_g=round(lote.kg_larva_real * 1000, 1),
                        etapa=lote.etapa_actual, modulo=lote.modulo)
                    session.add(r)
                    snapshot.append({"lote": lote.codigo, "modulo": lote.modulo,
                                     "temperatura": r.temperatura, "humedad": r.humedad, "nh3": r.nh3})
                await session.commit()
            if snapshot:
                await events.publish("reading", {"lecturas": snapshot})
        except Exception as e:
            print(f"[simulator] error: {e}")
        await asyncio.sleep(10)

async def kernel_loop():
    """Corre el ciclo cognitivo del agente periódicamente."""
    await asyncio.sleep(8)
    while True:
        try:
            await agent.cognitive_cycle()
        except Exception as e:
            print(f"[kernel_loop] error: {e}")
        await asyncio.sleep(25)

# ─── SCHEMAS ──────────────────────────────────────────────────────────────────

class LoteCreate(BaseModel):
    codigo: str
    kg_entrada: float = 1000.0
    modulo: str = "MOD-01"
    notas: str = ""

class QueryRequest(BaseModel):
    query: str

class WorkflowRequest(BaseModel):
    workflow_key: str
    lote_codigo: str
    input_data: dict = {}

class AlertaUpdate(BaseModel):
    resuelta: bool
    accion_tomada: str = ""

class SimulateRequest(BaseModel):
    lote_codigo: str
    cambios: dict

class AutonomyRequest(BaseModel):
    level: str  # info | warning | critical

class ApprovalResolve(BaseModel):
    nota: str = ""
    by: str = "operador"

# ─── ROOT + HEALTH ─────────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@app.get("/", response_class=HTMLResponse)
async def root():
    with open(os.path.join(BASE_DIR, "index.html"), encoding="utf-8") as f:
        return f.read()

@app.get("/api/health")
async def health():
    try:
        async with AsyncSessionLocal() as session:
            await session.execute(select(func.count()).select_from(Lote))
        return {"status": "ok", "db": DB_BACKEND, "llm": "openai" if llm.llm_available() else "deterministico",
                "time": datetime.utcnow().isoformat()}
    except Exception as e:
        return JSONResponse(status_code=503, content={"status": "degraded", "detail": str(e)})

# ─── DASHBOARD ─────────────────────────────────────────────────────────────────

@app.get("/api/dashboard")
async def get_dashboard():
    async with AsyncSessionLocal() as session:
        lotes = (await session.execute(select(Lote).order_by(desc(Lote.created_at)))).scalars().all()
        active_alerts = (await session.execute(
            select(func.count()).select_from(Alerta).where(Alerta.resuelta == False))).scalar()
        pending_approvals = (await session.execute(
            select(func.count()).select_from(Approval).where(Approval.status == ApprovalStatus.PENDING))).scalar()

        lote_data = []
        for lote in lotes:
            reading = (await session.execute(
                select(LecturaSensor).where(LecturaSensor.lote_id == lote.id)
                .order_by(desc(LecturaSensor.timestamp)).limit(1))).scalar_one_or_none()
            lote_data.append({
                "id": lote.id, "codigo": lote.codigo, "dia_actual": lote.dia_actual,
                "dias_ciclo": lote.dias_ciclo, "etapa": lote.etapa_actual, "status": lote.status,
                "modulo": lote.modulo, "kg_entrada": lote.kg_entrada,
                "kg_larva_proy": lote.kg_larva_proy, "kg_larva_real": lote.kg_larva_real,
                "kg_frass_proy": lote.kg_frass_proy, "kg_frass_real": lote.kg_frass_real,
                "rendimiento_pct": round((lote.kg_larva_real / lote.kg_larva_proy) * 100, 1) if lote.kg_larva_proy else 0,
                "notas": lote.notas,
                "ultimo_lectura": {
                    "temperatura": reading.temperatura, "humedad": reading.humedad, "nh3": reading.nh3,
                    "co2": reading.co2, "ph_sustrato": reading.ph_sustrato,
                    "timestamp": reading.timestamp.isoformat()} if reading else None,
            })

        workflows = (await session.execute(
            select(WorkflowLog).order_by(desc(WorkflowLog.timestamp)).limit(5))).scalars().all()
        fin = economics.resumen_financiero(lotes)
        autonomy = await agent.get_autonomy_level()

        return {
            "timestamp": datetime.utcnow().isoformat(),
            "active_alerts": active_alerts,
            "pending_approvals": pending_approvals,
            "autonomy_level": autonomy,
            "llm_mode": "openai" if llm.llm_available() else "deterministico",
            "lotes": lote_data,
            "thresholds": THRESHOLDS,
            "financiero": {"total_valor_proyectado_cop": fin["total_valor_proyectado_cop"],
                           "total_valor_proyectado_usd": fin["total_valor_proyectado_usd"],
                           "total_margen_proyectado_cop": fin["total_margen_proyectado_cop"]},
            "recent_workflows": [{"id": w.id, "name": w.workflow_name, "lote": w.lote_codigo,
                                  "status": w.status, "ts": w.timestamp.isoformat()} for w in workflows],
        }

# ─── LOTES ─────────────────────────────────────────────────────────────────────

@app.get("/api/lotes")
async def get_lotes():
    async with AsyncSessionLocal() as session:
        lotes = (await session.execute(select(Lote).order_by(desc(Lote.created_at)))).scalars().all()
        return [{"id": l.id, "codigo": l.codigo, "dia_actual": l.dia_actual, "etapa": l.etapa_actual,
                 "status": l.status, "modulo": l.modulo, "kg_larva_real": l.kg_larva_real,
                 "kg_frass_real": l.kg_frass_real,
                 "rendimiento_pct": round((l.kg_larva_real / l.kg_larva_proy) * 100, 1) if l.kg_larva_proy else 0}
                for l in lotes]

@app.post("/api/lotes", status_code=201)
async def create_lote(data: LoteCreate):
    async with AsyncSessionLocal() as session:
        planta = (await session.execute(select(Planta).limit(1))).scalar_one_or_none()
        lote = Lote(codigo=data.codigo, fecha_inicio=datetime.utcnow(), kg_entrada=data.kg_entrada,
                    modulo=data.modulo, notas=data.notas, planta_id=planta.id if planta else None)
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

@app.get("/api/lotes/{lote_id}/prediccion")
async def predecir_rendimiento(lote_id: int):
    async with AsyncSessionLocal() as session:
        lote = await session.get(Lote, lote_id)
        if not lote:
            raise HTTPException(404, "Lote no encontrado")
        return await prediction.predict_yield(session, lote)

@app.get("/api/lotes/{lote_id}/trazabilidad")
async def get_trazabilidad(lote_id: int):
    async with AsyncSessionLocal() as session:
        lote = await session.get(Lote, lote_id)
        if not lote:
            raise HTTPException(404, "Lote no encontrado")
        planta = await session.get(Planta, lote.planta_id) if lote.planta_id else None
        prod = (await session.execute(select(ProduccionDiaria)
                .where(ProduccionDiaria.lote_codigo == lote.codigo)
                .order_by(ProduccionDiaria.dia_ciclo))).scalars().all()
        alertas = (await session.execute(select(Alerta).where(Alerta.lote_id == lote.id)
                   .order_by(desc(Alerta.timestamp)))).scalars().all()
        wfs = (await session.execute(select(WorkflowLog).where(WorkflowLog.lote_codigo == lote.codigo)
               .order_by(desc(WorkflowLog.timestamp)))).scalars().all()
        stats = (await session.execute(select(
            func.round(func.avg(LecturaSensor.temperatura), 2),
            func.round(func.avg(LecturaSensor.humedad), 2),
            func.round(func.max(LecturaSensor.nh3), 2),
        ).where(LecturaSensor.lote_id == lote.id))).first()
        return {
            "lote": lote.codigo, "planta": planta.nombre if planta else "—",
            "ciudad": planta.ciudad if planta else "—",
            "fecha_inicio": lote.fecha_inicio.isoformat(), "modulo": lote.modulo,
            "etapa_actual": lote.etapa_actual, "status": lote.status,
            "kg_entrada_residuo": lote.kg_entrada, "kg_larva": lote.kg_larva_real,
            "kg_frass": lote.kg_frass_real,
            "condiciones_promedio": {"temp_c": stats[0] if stats else None,
                                     "humedad_pct": stats[1] if stats else None,
                                     "nh3_max_ppm": stats[2] if stats else None},
            "economia": economics.valor_lote(lote),
            "produccion_diaria": [{"dia": p.dia_ciclo, "kg_larva": p.kg_larva_acum,
                                   "kg_frass": p.kg_frass_acum, "rendimiento_pct": p.rendimiento_pct}
                                  for p in prod],
            "alertas_historicas": [{"severidad": a.severidad, "variable": a.variable,
                                    "mensaje": a.mensaje, "timestamp": a.timestamp.isoformat()} for a in alertas],
            "intervenciones": [{"workflow": w.workflow_name, "status": w.status,
                                "timestamp": w.timestamp.isoformat()} for w in wfs],
            "certificacion": "Trazabilidad digital completa — apto para certificación ICA",
        }

@app.get("/api/lotes/{lote_id}/qr")
async def get_qr(lote_id: int, request: Request):
    import qrcode
    url = f"{request.base_url}api/lotes/{lote_id}/trazabilidad"
    img = qrcode.make(url)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return Response(content=buf.getvalue(), media_type="image/png")

# ─── LECTURAS ──────────────────────────────────────────────────────────────────

@app.get("/api/lecturas")
async def get_lecturas(lote_id: Optional[int] = None, modulo: Optional[str] = None,
                       limit: int = Query(default=50, le=500), hours: int = Query(default=24, le=720)):
    async with AsyncSessionLocal() as session:
        q = select(LecturaSensor).where(
            LecturaSensor.timestamp >= datetime.utcnow() - timedelta(hours=hours))
        if lote_id:
            q = q.where(LecturaSensor.lote_id == lote_id)
        if modulo:
            q = q.where(LecturaSensor.modulo == modulo)
        q = q.order_by(desc(LecturaSensor.timestamp)).limit(limit)
        readings = (await session.execute(q)).scalars().all()
        return [{"id": r.id, "lote_id": r.lote_id, "modulo": r.modulo,
                 "timestamp": r.timestamp.isoformat(), "temperatura": r.temperatura,
                 "humedad": r.humedad, "nh3": r.nh3, "co2": r.co2,
                 "ph_sustrato": r.ph_sustrato, "masa_larva_g": r.masa_larva_g} for r in readings]

@app.get("/api/lecturas/stats")
async def get_stats(modulo: Optional[str] = None, hours: int = Query(default=24, le=720)):
    async with AsyncSessionLocal() as session:
        q = select(
            LecturaSensor.modulo,
            func.round(func.avg(LecturaSensor.temperatura), 2).label("temp_avg"),
            func.round(func.min(LecturaSensor.temperatura), 2).label("temp_min"),
            func.round(func.max(LecturaSensor.temperatura), 2).label("temp_max"),
            func.round(func.avg(LecturaSensor.humedad), 2).label("hum_avg"),
            func.round(func.max(LecturaSensor.nh3), 2).label("nh3_max"),
            func.round(func.avg(LecturaSensor.nh3), 2).label("nh3_avg"),
            func.count(LecturaSensor.id).label("total_lecturas"),
        ).where(LecturaSensor.timestamp >= datetime.utcnow() - timedelta(hours=hours))
        if modulo:
            q = q.where(LecturaSensor.modulo == modulo)
        q = q.group_by(LecturaSensor.modulo)
        rows = (await session.execute(q)).all()
        return [dict(r._mapping) for r in rows]

# ─── ALERTAS ───────────────────────────────────────────────────────────────────

@app.get("/api/alertas")
async def get_alertas(resuelta: Optional[bool] = None, limit: int = 50):
    async with AsyncSessionLocal() as session:
        q = select(Alerta, Lote.codigo.label("lote_codigo")) \
            .join(Lote, Alerta.lote_id == Lote.id, isouter=True) \
            .order_by(desc(Alerta.timestamp)).limit(limit)
        if resuelta is not None:
            q = q.where(Alerta.resuelta == resuelta)
        rows = (await session.execute(q)).all()
        return [{"id": row.Alerta.id, "lote": row.lote_codigo, "severidad": row.Alerta.severidad,
                 "variable": row.Alerta.variable, "mensaje": row.Alerta.mensaje,
                 "valor_actual": row.Alerta.valor_actual, "valor_limite": row.Alerta.valor_limite,
                 "resuelta": row.Alerta.resuelta, "accion_tomada": row.Alerta.accion_tomada,
                 "timestamp": row.Alerta.timestamp.isoformat()} for row in rows]

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

# ─── WORKFLOWS ─────────────────────────────────────────────────────────────────

@app.get("/api/workflows/available")
async def get_workflows_available():
    return [{"key": k, "name": v["name"], "trigger": v["trigger"],
             "role": v.get("role", "proceso"), "steps": v["steps"]} for k, v in WORKFLOWS.items()]

@app.post("/api/workflows/execute")
async def execute_workflow(data: WorkflowRequest):
    return await agent.execute_workflow(data.workflow_key, data.lote_codigo,
                                        {**data.input_data, "triggered_by": "operador_manual"})

@app.get("/api/workflows/history")
async def get_workflow_history(limit: int = 50):
    async with AsyncSessionLocal() as session:
        rows = (await session.execute(
            select(WorkflowLog).order_by(desc(WorkflowLog.timestamp)).limit(limit))).scalars().all()
        return [{"id": w.id, "workflow": w.workflow_name, "lote": w.lote_codigo, "status": w.status,
                 "triggered_by": w.triggered_by, "input": w.input_data, "result": w.result,
                 "duration_ms": w.duration_ms, "timestamp": w.timestamp.isoformat()} for w in rows]

# ─── FINANCIERO ────────────────────────────────────────────────────────────────

@app.get("/api/financiero")
async def get_financiero():
    async with AsyncSessionLocal() as session:
        lotes = (await session.execute(select(Lote))).scalars().all()
        fin = economics.resumen_financiero(lotes)
        fin["timestamp"] = datetime.utcnow().isoformat()
        return fin

# ─── AGENTE IA ─────────────────────────────────────────────────────────────────

@app.post("/api/agent/analyze")
async def agent_analyze():
    findings = await agent.analyze_latest_readings()
    return {"timestamp": datetime.utcnow().isoformat(), "findings": findings}

@app.post("/api/agent/query")
async def agent_query(data: QueryRequest):
    return await agent.run_query(data.query)

@app.post("/api/agent/simulate")
async def agent_simulate(req: SimulateRequest):
    async with AsyncSessionLocal() as session:
        lote = (await session.execute(select(Lote).where(Lote.codigo == req.lote_codigo))).scalar_one_or_none()
        if not lote:
            raise HTTPException(404, "Lote no encontrado")
        reading = (await session.execute(select(LecturaSensor).where(LecturaSensor.lote_id == lote.id)
                   .order_by(desc(LecturaSensor.timestamp)).limit(1))).scalar_one_or_none()
        rdict = {} if not reading else {"temperatura": reading.temperatura,
                                        "humedad": reading.humedad, "nh3": reading.nh3}
        return prediction.simulate_intervention(lote, rdict, req.cambios)

@app.get("/api/agent/decisions")
async def agent_decisions(limit: int = Query(default=30, le=100)):
    async with AsyncSessionLocal() as session:
        rows = (await session.execute(
            select(AgentDecision).order_by(desc(AgentDecision.timestamp)).limit(limit))).scalars().all()
        return [{"id": d.id, "timestamp": d.timestamp.isoformat(), "lote": d.lote_codigo,
                 "modulo": d.modulo, "role": d.role, "severidad": d.severidad, "titulo": d.titulo,
                 "percepcion": d.percepcion, "razonamiento": d.razonamiento,
                 "accion": d.accion_propuesta, "outcome": d.outcome, "confianza": d.confianza,
                 "modelo": d.modelo, "tokens": d.tokens} for d in rows]

@app.get("/api/agent/memory")
async def agent_memory():
    async with AsyncSessionLocal() as session:
        rows = (await session.execute(
            select(AgentMemory).order_by(desc(AgentMemory.timestamp)).limit(50))).scalars().all()
        return [{"id": m.id, "scope": m.scope, "clave": m.clave, "valor": m.valor,
                 "nota": m.nota, "timestamp": m.timestamp.isoformat()} for m in rows]

# ─── APROBACIONES (HITL) ───────────────────────────────────────────────────────

@app.get("/api/approvals")
async def get_approvals(status: Optional[str] = "pending", limit: int = 50):
    async with AsyncSessionLocal() as session:
        q = select(Approval).order_by(desc(Approval.timestamp)).limit(limit)
        if status:
            q = q.where(Approval.status == status)
        rows = (await session.execute(q)).scalars().all()
        return [{"id": a.id, "lote": a.lote_codigo, "titulo": a.titulo, "rationale": a.rationale,
                 "workflow_key": a.workflow_key, "severidad": a.severidad, "status": a.status,
                 "timestamp": a.timestamp.isoformat(),
                 "resolved_by": a.resolved_by, "nota": a.nota_operador} for a in rows]

@app.post("/api/approvals/{approval_id}/approve")
async def approve(approval_id: int, body: ApprovalResolve = ApprovalResolve()):
    return await agent.resolve_approval(approval_id, True, body.by, body.nota)

@app.post("/api/approvals/{approval_id}/reject")
async def reject(approval_id: int, body: ApprovalResolve = ApprovalResolve()):
    return await agent.resolve_approval(approval_id, False, body.by, body.nota)

# ─── DIAL DE AUTONOMÍA ─────────────────────────────────────────────────────────

@app.get("/api/config/autonomy")
async def get_autonomy():
    level = await agent.get_autonomy_level()
    return {"level": level, "options": ["info", "warning", "critical"],
            "descripcion": {"info": "Solo observa; toda intervención requiere aprobación",
                            "warning": "Auto-ejecuta rutinas; críticas requieren aprobación",
                            "critical": "Autónomo total (con auditoría)"}}

@app.post("/api/config/autonomy")
async def set_autonomy(req: AutonomyRequest):
    level = await agent.set_autonomy_level(req.level)
    return {"level": level}

# ─── PRODUCCIÓN ────────────────────────────────────────────────────────────────

@app.get("/api/produccion")
async def get_produccion(lote_codigo: Optional[str] = None):
    async with AsyncSessionLocal() as session:
        q = select(ProduccionDiaria).order_by(ProduccionDiaria.fecha)
        if lote_codigo:
            q = q.where(ProduccionDiaria.lote_codigo == lote_codigo)
        rows = (await session.execute(q)).scalars().all()
        return [{"fecha": p.fecha.isoformat(), "lote": p.lote_codigo, "dia": p.dia_ciclo,
                 "kg_larva": p.kg_larva_acum, "kg_frass": p.kg_frass_acum, "temp_prom": p.temp_promedio,
                 "hum_prom": p.hum_promedio, "nh3_max": p.nh3_max, "alertas": p.alertas_count,
                 "rendimiento_pct": p.rendimiento_pct} for p in rows]

# ─── SSE LIVE STREAM ───────────────────────────────────────────────────────────

@app.get("/api/stream")
async def stream():
    async def gen():
        q = events.subscribe()
        try:
            for msg in events.recent():
                yield f"data: {msg}\n\n"
            while True:
                try:
                    msg = await asyncio.wait_for(q.get(), timeout=15)
                    yield f"data: {msg}\n\n"
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
        finally:
            events.unsubscribe(q)
    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no",
                                      "Connection": "keep-alive"})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
