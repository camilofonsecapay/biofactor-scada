"""
BioFactor — Capa LLM (agente con herramientas / tool-use)
El salto AI-native: el agente NO responde de memoria. Razona y orquesta
herramientas (function calling) que consultan la DB real, predicen
rendimiento, calculan economía y simulan intervenciones.

Dual-mode: si hay OPENAI_API_KEY usa OpenAI; si no (o si falla), cae a un
motor determinístico → el sistema nunca se rompe en una demo.
"""

import os
import json
import time
from datetime import datetime, timedelta

from sqlalchemy import select, desc, func
from database import (
    AsyncSessionLocal, Lote, LecturaSensor, Alerta, WorkflowLog,
    AgentDecision, LoteStatus,
)
import economics
import prediction

MODEL_FAST = os.getenv("OPENAI_MODEL_FAST", "gpt-4o-mini")
MODEL_DEEP = os.getenv("OPENAI_MODEL_DEEP", "gpt-4o")


def llm_available() -> bool:
    return bool(os.getenv("OPENAI_API_KEY"))


def _client():
    from openai import AsyncOpenAI
    return AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ─── TOOLS (function calling schema) ────────────────────────────────────────────

TOOLS = [
    {"type": "function", "function": {
        "name": "get_dashboard_state",
        "description": "Estado general de la planta: lotes activos, alertas sin resolver, totales económicos. Úsalo para preguntas amplias.",
        "parameters": {"type": "object", "properties": {}},
    }},
    {"type": "function", "function": {
        "name": "get_lote",
        "description": "Detalle de un lote por su código (ej. BSF-2026-06A): día, etapa, kg larva/frass, última lectura de sensores.",
        "parameters": {"type": "object", "properties": {
            "codigo": {"type": "string", "description": "Código del lote"}}, "required": ["codigo"]},
    }},
    {"type": "function", "function": {
        "name": "query_readings",
        "description": "Estadísticas de lecturas de sensores (promedio/min/max de temperatura, humedad, NH₃) por módulo en las últimas N horas.",
        "parameters": {"type": "object", "properties": {
            "modulo": {"type": "string", "description": "Filtrar por módulo, ej. MOD-02 (opcional)"},
            "hours": {"type": "integer", "description": "Ventana de horas (default 24)"}}},
    }},
    {"type": "function", "function": {
        "name": "predict_yield",
        "description": "Predice el rendimiento final (kg de larva al día 14) de un lote, con nivel de confianza y valor económico.",
        "parameters": {"type": "object", "properties": {
            "codigo": {"type": "string"}}, "required": ["codigo"]},
    }},
    {"type": "function", "function": {
        "name": "compute_economics",
        "description": "Valor económico (COP/USD) acumulado y proyectado. Sin código = todos los lotes; con código = uno solo. Incluye desglose larva/frass/gate fee y margen.",
        "parameters": {"type": "object", "properties": {
            "codigo": {"type": "string", "description": "Código del lote (opcional)"}}},
    }},
    {"type": "function", "function": {
        "name": "simulate_intervention",
        "description": "Simula el impacto (rendimiento + económico) de un cambio de condiciones en un lote. cambios es un objeto como {\"temperatura\": -2} o {\"nh3\": -10}.",
        "parameters": {"type": "object", "properties": {
            "codigo": {"type": "string"},
            "cambios": {"type": "object", "description": "Delta por variable, ej. {\"temperatura\": -2}"}},
            "required": ["codigo", "cambios"]},
    }},
    {"type": "function", "function": {
        "name": "get_alerts",
        "description": "Lista de alertas. resuelta=false para activas (default), true para resueltas.",
        "parameters": {"type": "object", "properties": {
            "resuelta": {"type": "boolean"}}},
    }},
    {"type": "function", "function": {
        "name": "get_decisions",
        "description": "Decisiones recientes del agente (traza de razonamiento autónomo): qué percibió, qué razonó, qué acción tomó.",
        "parameters": {"type": "object", "properties": {
            "limit": {"type": "integer"}}},
    }},
    {"type": "function", "function": {
        "name": "execute_workflow",
        "description": "Ejecuta un workflow del proceso sobre un lote. Solo úsalo si el operador lo pide explícitamente. workflow_key ∈ {alta_temperatura, baja_humedad, nh3_elevado, ciclo_completo, bajo_rendimiento, analisis_diario}.",
        "parameters": {"type": "object", "properties": {
            "workflow_key": {"type": "string"},
            "codigo": {"type": "string"},
            "input_data": {"type": "object"}},
            "required": ["workflow_key", "codigo"]},
    }},
]


# ─── TOOL EXECUTORS ─────────────────────────────────────────────────────────────

async def _tool_get_dashboard_state() -> dict:
    async with AsyncSessionLocal() as s:
        lotes = (await s.execute(select(Lote))).scalars().all()
        activas = (await s.execute(
            select(func.count()).select_from(Alerta).where(Alerta.resuelta == False)
        )).scalar()
        fin = economics.resumen_financiero(lotes)
        return {
            "lotes_total": len(lotes),
            "alertas_activas": activas,
            "valor_proyectado_cop": fin["total_valor_proyectado_cop"],
            "valor_proyectado_usd": fin["total_valor_proyectado_usd"],
            "lotes": [{"codigo": l.codigo, "dia": l.dia_actual, "dias_ciclo": l.dias_ciclo,
                       "etapa": l.etapa_actual, "status": l.status, "modulo": l.modulo,
                       "kg_larva_real": l.kg_larva_real} for l in lotes],
        }


async def _tool_get_lote(codigo: str) -> dict:
    async with AsyncSessionLocal() as s:
        lote = (await s.execute(select(Lote).where(Lote.codigo == codigo))).scalar_one_or_none()
        if not lote:
            return {"error": f"Lote {codigo} no encontrado"}
        lr = (await s.execute(select(LecturaSensor).where(LecturaSensor.lote_id == lote.id)
                              .order_by(desc(LecturaSensor.timestamp)).limit(1))).scalar_one_or_none()
        return {
            "codigo": lote.codigo, "dia_actual": lote.dia_actual, "dias_ciclo": lote.dias_ciclo,
            "etapa": lote.etapa_actual, "status": lote.status, "modulo": lote.modulo,
            "kg_entrada": lote.kg_entrada, "kg_larva_real": lote.kg_larva_real,
            "kg_larva_proy": lote.kg_larva_proy, "kg_frass_real": lote.kg_frass_real,
            "notas": lote.notas,
            "ultima_lectura": None if not lr else {
                "temperatura": lr.temperatura, "humedad": lr.humedad, "nh3": lr.nh3,
                "co2": lr.co2, "ph_sustrato": lr.ph_sustrato, "timestamp": lr.timestamp.isoformat()},
        }


async def _tool_query_readings(modulo: str = None, hours: int = 24) -> dict:
    async with AsyncSessionLocal() as s:
        q = select(
            LecturaSensor.modulo,
            func.round(func.avg(LecturaSensor.temperatura), 2).label("temp_avg"),
            func.round(func.max(LecturaSensor.temperatura), 2).label("temp_max"),
            func.round(func.avg(LecturaSensor.humedad), 2).label("hum_avg"),
            func.round(func.max(LecturaSensor.nh3), 2).label("nh3_max"),
            func.round(func.avg(LecturaSensor.nh3), 2).label("nh3_avg"),
            func.count(LecturaSensor.id).label("n"),
        ).where(LecturaSensor.timestamp >= datetime.utcnow() - timedelta(hours=hours))
        if modulo:
            q = q.where(LecturaSensor.modulo == modulo)
        q = q.group_by(LecturaSensor.modulo)
        rows = (await s.execute(q)).all()
        return {"ventana_horas": hours, "por_modulo": [dict(r._mapping) for r in rows]}


async def _tool_predict_yield(codigo: str) -> dict:
    async with AsyncSessionLocal() as s:
        lote = (await s.execute(select(Lote).where(Lote.codigo == codigo))).scalar_one_or_none()
        if not lote:
            return {"error": f"Lote {codigo} no encontrado"}
        return await prediction.predict_yield(s, lote)


async def _tool_compute_economics(codigo: str = None) -> dict:
    async with AsyncSessionLocal() as s:
        if codigo:
            lote = (await s.execute(select(Lote).where(Lote.codigo == codigo))).scalar_one_or_none()
            if not lote:
                return {"error": f"Lote {codigo} no encontrado"}
            return economics.valor_lote(lote)
        lotes = (await s.execute(select(Lote))).scalars().all()
        return economics.resumen_financiero(lotes)


async def _tool_simulate_intervention(codigo: str, cambios: dict) -> dict:
    async with AsyncSessionLocal() as s:
        lote = (await s.execute(select(Lote).where(Lote.codigo == codigo))).scalar_one_or_none()
        if not lote:
            return {"error": f"Lote {codigo} no encontrado"}
        lr = (await s.execute(select(LecturaSensor).where(LecturaSensor.lote_id == lote.id)
                              .order_by(desc(LecturaSensor.timestamp)).limit(1))).scalar_one_or_none()
        reading = {} if not lr else {"temperatura": lr.temperatura, "humedad": lr.humedad, "nh3": lr.nh3}
        return prediction.simulate_intervention(lote, reading, cambios or {})


async def _tool_get_alerts(resuelta: bool = False) -> dict:
    async with AsyncSessionLocal() as s:
        q = select(Alerta, Lote.codigo).join(Lote, Alerta.lote_id == Lote.id, isouter=True) \
            .where(Alerta.resuelta == resuelta).order_by(desc(Alerta.timestamp)).limit(20)
        rows = (await s.execute(q)).all()
        return {"alertas": [{"lote": r.codigo, "severidad": r.Alerta.severidad,
                             "variable": r.Alerta.variable, "mensaje": r.Alerta.mensaje,
                             "timestamp": r.Alerta.timestamp.isoformat()} for r in rows]}


async def _tool_get_decisions(limit: int = 10) -> dict:
    async with AsyncSessionLocal() as s:
        rows = (await s.execute(select(AgentDecision)
                .order_by(desc(AgentDecision.timestamp)).limit(min(limit or 10, 30)))).scalars().all()
        return {"decisiones": [{"lote": d.lote_codigo, "rol": d.role, "titulo": d.titulo,
                                "severidad": d.severidad, "razonamiento": d.razonamiento,
                                "accion": d.accion_propuesta, "outcome": d.outcome,
                                "timestamp": d.timestamp.isoformat()} for d in rows]}


async def _tool_execute_workflow(workflow_key: str, codigo: str, input_data: dict = None) -> dict:
    from agent import ScadaAgent  # lazy para evitar import circular
    data = {**(input_data or {}), "triggered_by": "operador_chat"}
    return await ScadaAgent().execute_workflow(workflow_key, codigo, data)


_DISPATCH = {
    "get_dashboard_state": _tool_get_dashboard_state,
    "get_lote": _tool_get_lote,
    "query_readings": _tool_query_readings,
    "predict_yield": _tool_predict_yield,
    "compute_economics": _tool_compute_economics,
    "simulate_intervention": _tool_simulate_intervention,
    "get_alerts": _tool_get_alerts,
    "get_decisions": _tool_get_decisions,
    "execute_workflow": _tool_execute_workflow,
}


async def _dispatch_tool(name: str, args: dict) -> dict:
    fn = _DISPATCH.get(name)
    if not fn:
        return {"error": f"herramienta desconocida: {name}"}
    try:
        return await fn(**(args or {}))
    except TypeError as e:
        return {"error": f"argumentos inválidos para {name}: {e}"}
    except Exception as e:
        return {"error": f"fallo ejecutando {name}: {e}"}


SYSTEM_PROMPT = """Eres BioFactor Copilot, el cerebro operativo de una planta industrial de \
bioconversión de residuos orgánicos con larvas de Mosca Soldado Negra (BSF / Hermetia illucens) \
de Urbaser en Montería, Colombia.

El proceso: 1.000 kg de residuo → ~130 kg de larva (proteína) + ~450 kg de frass (biofertilizante) \
en un ciclo de 14 días. Es biológicamente delicado: ±2°C de temperatura puede reducir el rendimiento \
de larva 30-40%; NH₃ alto indica putrefacción del sustrato.

Actúas según el rol que aplique a la pregunta: PROCESO (biología/control), FINANCIERO (economía), \
CALIDAD/TRAZABILIDAD (certificación ICA) u OPERACIONES (alertas/despacho).

Reglas:
- NUNCA inventes datos. Usa las herramientas para obtener cifras reales de la planta antes de afirmar nada.
- Responde en español, conciso y técnico, como un ingeniero de planta senior.
- Termina con una recomendación operativa accionable cuando aplique.
- Si el operador pide ejecutar una acción/workflow, hazlo solo si lo pide explícitamente.
- Hoy es {hoy}."""


async def run_agent_query(query_text: str, deep: bool = True) -> dict:
    """Query del operador resuelta con tool-use real. Cae a determinístico si no hay LLM."""
    if not llm_available():
        return await deterministic_query(query_text)

    model = MODEL_DEEP if deep else MODEL_FAST
    t0 = time.time()
    tokens = 0
    tools_used = []
    try:
        client = _client()
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT.format(hoy=datetime.utcnow().date().isoformat())},
            {"role": "user", "content": query_text},
        ]
        for _ in range(6):  # tope de iteraciones de tool-use
            resp = await client.chat.completions.create(
                model=model, messages=messages, tools=TOOLS, tool_choice="auto", temperature=0.2)
            if resp.usage:
                tokens += resp.usage.total_tokens
            msg = resp.choices[0].message
            if msg.tool_calls:
                messages.append({
                    "role": "assistant", "content": msg.content or "",
                    "tool_calls": [{"id": tc.id, "type": "function",
                                    "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                                   for tc in msg.tool_calls],
                })
                for tc in msg.tool_calls:
                    try:
                        args = json.loads(tc.function.arguments or "{}")
                    except json.JSONDecodeError:
                        args = {}
                    result = await _dispatch_tool(tc.function.name, args)
                    tools_used.append({"tool": tc.function.name, "args": args})
                    messages.append({"role": "tool", "tool_call_id": tc.id,
                                     "content": json.dumps(result, ensure_ascii=False, default=str)})
                continue
            return {
                "query": query_text, "respuesta": msg.content, "mode": "llm",
                "modelo": model, "tools_used": tools_used, "tokens": tokens,
                "latencia_ms": int((time.time() - t0) * 1000),
            }
        return {"query": query_text, "respuesta": "No pude resolver la consulta en los pasos disponibles.",
                "mode": "llm", "modelo": model, "tools_used": tools_used}
    except Exception as e:
        print(f"[llm] fallback determinístico por error: {e}")
        fb = await deterministic_query(query_text)
        fb["llm_error"] = str(e)
        return fb


async def narrate(role: str, contexto: str) -> dict:
    """Narración breve del kernel (modelo rápido). Fallback a plantilla."""
    if not llm_available():
        return {"texto": contexto, "modelo": "deterministico", "tokens": 0}
    try:
        client = _client()
        resp = await client.chat.completions.create(
            model=MODEL_FAST, temperature=0.3, max_tokens=160,
            messages=[
                {"role": "system", "content": f"Eres el agente BioFactor actuando como rol {role}. "
                 "Narra en 1-2 frases, en español técnico, qué observas y qué decides. Sin preámbulos."},
                {"role": "user", "content": contexto},
            ])
        return {"texto": resp.choices[0].message.content,
                "modelo": MODEL_FAST,
                "tokens": resp.usage.total_tokens if resp.usage else 0}
    except Exception as e:
        print(f"[llm] narrate fallback: {e}")
        return {"texto": contexto, "modelo": "deterministico", "tokens": 0}


# ─── MOTOR DETERMINÍSTICO (fallback, sin LLM) ──────────────────────────────────

async def deterministic_query(query_text: str) -> dict:
    """Motor por reglas/keywords. Garantiza respuesta aunque no haya API key."""
    q = (query_text or "").lower().strip()
    async with AsyncSessionLocal() as session:
        if "temperatura" in q and ("promedio" in q or "avg" in q or "modulo" in q or "módulo" in q):
            rows = (await session.execute(select(
                LecturaSensor.modulo,
                func.round(func.avg(LecturaSensor.temperatura), 2).label("avg"),
                func.round(func.min(LecturaSensor.temperatura), 2).label("min"),
                func.round(func.max(LecturaSensor.temperatura), 2).label("max"),
            ).group_by(LecturaSensor.modulo))).all()
            data = [{"modulo": r.modulo, "promedio": r.avg, "min": r.min, "max": r.max} for r in rows]
            return {"query": "Temperatura por módulo", "mode": "deterministico", "tools_used": [],
                    "respuesta": "Temperatura por módulo: " + "; ".join(
                        f"{d['modulo']} prom {d['promedio']}°C (min {d['min']}, max {d['max']})" for d in data),
                    "data": data}

        if "financ" in q or "valor" in q or "cop" in q or "dinero" in q or "economi" in q or "económ" in q:
            lotes = (await session.execute(select(Lote))).scalars().all()
            fin = economics.resumen_financiero(lotes)
            return {"query": "Resumen financiero", "mode": "deterministico", "tools_used": [],
                    "respuesta": f"Valor proyectado total: {fin['total_valor_proyectado_cop']:,} COP "
                                 f"(~${fin['total_valor_proyectado_usd']:,} USD). "
                                 f"Margen proyectado: {fin['total_margen_proyectado_cop']:,} COP.",
                    "data": fin}

        if "predic" in q or "rendimiento" in q or "proyec" in q:
            lotes = (await session.execute(select(Lote))).scalars().all()
            preds = [await prediction.predict_yield(session, l) for l in lotes]
            return {"query": "Predicción de rendimiento", "mode": "deterministico", "tools_used": [],
                    "respuesta": "Predicción por lote: " + "; ".join(
                        f"{p['lote']} → {p['kg_larva_predicho_final']}kg ({p['rendimiento_predicho_pct']}%, conf. {p['confianza']})"
                        for p in preds),
                    "data": preds}

        if "alerta" in q:
            resueltas = "resuelta" in q
            rows = (await session.execute(select(Alerta, Lote.codigo)
                    .join(Lote, Alerta.lote_id == Lote.id, isouter=True)
                    .where(Alerta.resuelta == resueltas).order_by(desc(Alerta.timestamp)).limit(20))).all()
            data = [{"lote": r.codigo, "variable": r.Alerta.variable, "severidad": r.Alerta.severidad,
                     "mensaje": r.Alerta.mensaje} for r in rows]
            return {"query": f"Alertas {'resueltas' if resueltas else 'activas'}", "mode": "deterministico",
                    "tools_used": [], "respuesta": f"{len(data)} alertas. " + " | ".join(d["mensaje"] for d in data[:5]),
                    "data": data}

        if "nh3" in q or "amoni" in q or "amoní" in q:
            rows = (await session.execute(select(
                LecturaSensor.modulo,
                func.round(func.max(LecturaSensor.nh3), 2).label("max"),
                func.round(func.avg(LecturaSensor.nh3), 2).label("avg"),
            ).group_by(LecturaSensor.modulo))).all()
            data = [{"modulo": r.modulo, "max_ppm": r.max, "promedio_ppm": r.avg} for r in rows]
            return {"query": "NH₃ por módulo", "mode": "deterministico", "tools_used": [],
                    "respuesta": "NH₃ por módulo: " + "; ".join(
                        f"{d['modulo']} max {d['max_ppm']} / prom {d['promedio_ppm']} ppm" for d in data),
                    "data": data}

        if "lote" in q:
            lotes = (await session.execute(select(Lote).order_by(desc(Lote.created_at)))).scalars().all()
            data = [{"codigo": l.codigo, "dia": l.dia_actual, "etapa": l.etapa_actual,
                     "status": l.status, "modulo": l.modulo, "kg_larva": l.kg_larva_real} for l in lotes]
            return {"query": "Lotes", "mode": "deterministico", "tools_used": [],
                    "respuesta": f"{len(data)} lotes: " + "; ".join(
                        f"{d['codigo']} (día {d['dia']}, {d['etapa']}, {d['status']})" for d in data),
                    "data": data}

        return {"query": query_text, "mode": "deterministico", "tools_used": [],
                "respuesta": "No entendí la consulta. Probá: 'valor financiero', 'predicción de rendimiento', "
                             "'temperatura por módulo', 'alertas activas', 'NH3 por módulo', 'estado de lotes'.",
                "data": []}
