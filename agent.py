"""
BioFactor — Kernel cognitivo del agente
Loop explícito: Percibir → Razonar → Decidir → Actuar → Aprender.
A diferencia del loop de reglas anterior, cada ciclo produce una DECISIÓN
auditable (AgentDecision), aplica el DIAL DE AUTONOMÍA (auto-ejecuta o pide
aprobación humana) y narra lo que hace al feed en vivo (SSE).
"""

import json
import uuid
import time
from datetime import datetime, timedelta
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc, and_, func

from database import (
    AsyncSessionLocal, Lote, LecturaSensor, Alerta, WorkflowLog,
    AgentDecision, Approval, AgentMemory, SystemConfig,
    LoteStatus, AlertSeverity, WorkflowStatus, StageEnum,
    DecisionOutcome, ApprovalStatus,
)
import events
import llm

# ─── THRESHOLDS ────────────────────────────────────────────────────────────────
THRESHOLDS = {
    "temperatura":  {"min": 24.0, "opt_min": 28.0, "opt_max": 32.0, "max": 36.0},
    "humedad":      {"min": 50.0, "opt_min": 60.0, "opt_max": 72.0, "max": 85.0},
    "nh3":          {"warn": 40.0, "critical": 55.0},
    "co2":          {"warn": 1500, "critical": 2500},
    "ph_sustrato":  {"min": 6.0, "max": 8.0},
}

# ─── WORKFLOW DEFINITIONS ─────────────────────────────────────────────────────
WORKFLOWS = {
    "alta_temperatura": {
        "name": "Alta temperatura → Ajuste de ventilación",
        "trigger": "temperatura > opt_max", "role": "proceso",
        "steps": ["Registrar alerta en DB", "Incrementar ventilación módulo +15%",
                  "Re-evaluar en 30 minutos", "Si persiste > 34°C → notificar operador jefe"],
    },
    "baja_humedad": {
        "name": "Humedad baja → Activar riego",
        "trigger": "humedad < opt_min", "role": "proceso",
        "steps": ["Registrar alerta en DB", "Activar sistema de nebulización 5 min",
                  "Esperar 15 min y re-medir", "Si no sube → notificar operador"],
    },
    "nh3_elevado": {
        "name": "NH₃ elevado → Extracción de aire",
        "trigger": "nh3 > warn", "role": "proceso",
        "steps": ["Registrar alerta NH₃", "Aumentar extracción de aire +20%",
                  "Abrir compuerta de ventilación módulo afectado", "Monitorear cada 10 min"],
    },
    "ciclo_completo": {
        "name": "Ciclo 14 días → Protocolo de cosecha",
        "trigger": "dia_actual >= dias_ciclo", "role": "operaciones",
        "steps": ["Marcar lote como COSECHA en DB", "Notificar a equipo de cosecha",
                  "Preparar equipos de cribado", "Registrar pesos reales larva y frass",
                  "Calcular rendimiento vs proyectado", "Despachar al procesamiento"],
    },
    "bajo_rendimiento": {
        "name": "Rendimiento < 85% → Análisis causas",
        "trigger": "rendimiento_pct < 85", "role": "financiero",
        "steps": ["Generar reporte diagnóstico del lote", "Correlacionar variables con rendimiento",
                  "Identificar días críticos", "Recomendar ajuste parámetros próximo lote"],
    },
    "analisis_diario": {
        "name": "Análisis diario de producción",
        "trigger": "scheduler 06:00", "role": "operaciones",
        "steps": ["Leer lecturas últimas 24h de todos los lotes", "Calcular promedios y desviaciones",
                  "Comparar vs benchmark histórico", "Generar resumen ejecutivo",
                  "Proyectar rendimiento final de cada lote"],
    },
}

_SEV_RANK = {"info": 0, "warning": 1, "critical": 2}
_COOLDOWN_S = 150  # no re-emitir la misma condición por lote antes de esto


class ScadaAgent:
    """Cerebro operativo: kernel cognitivo + ejecución de workflows + NL query."""

    def __init__(self):
        self._cooldown: dict[str, datetime] = {}

    # ─── DIAL DE AUTONOMÍA ──────────────────────────────────────────────────────

    async def get_autonomy_level(self) -> str:
        async with AsyncSessionLocal() as s:
            c = (await s.execute(select(SystemConfig).where(SystemConfig.clave == "autonomy_level"))).scalar_one_or_none()
            return c.valor if c else "warning"

    async def set_autonomy_level(self, level: str) -> str:
        level = level if level in _SEV_RANK else "warning"
        async with AsyncSessionLocal() as s:
            c = (await s.execute(select(SystemConfig).where(SystemConfig.clave == "autonomy_level"))).scalar_one_or_none()
            if c:
                c.valor = level
            else:
                s.add(SystemConfig(clave="autonomy_level", valor=level))
            await s.commit()
        await events.publish("autonomy", {"level": level})
        return level

    # ─── NL QUERY (tool-use real) ───────────────────────────────────────────────

    async def run_query(self, query_text: str) -> dict:
        return await llm.run_agent_query(query_text, deep=True)

    # ─── ANÁLISIS (compat /api/agent/analyze) ───────────────────────────────────

    async def analyze_latest_readings(self) -> list[dict]:
        findings_out = []
        async with AsyncSessionLocal() as session:
            lotes = (await session.execute(
                select(Lote).where(Lote.status.in_([LoteStatus.ACTIVO, LoteStatus.ALERTA]))
            )).scalars().all()
            for lote in lotes:
                reading = await self._latest_reading(session, lote.id)
                if not reading:
                    continue
                fnds = self._derive_findings(lote, reading)
                findings_out.append({
                    "lote": lote.codigo, "modulo": lote.modulo, "dia_ciclo": lote.dia_actual,
                    "issues": [f["titulo"] for f in fnds],
                    "suggested_workflows": list({f["workflow_key"] for f in fnds if f["workflow_key"]}),
                    "reading": {"temperatura": reading.temperatura, "humedad": reading.humedad,
                                "nh3": reading.nh3, "co2": reading.co2, "ph_sustrato": reading.ph_sustrato,
                                "timestamp": reading.timestamp.isoformat()},
                })
        return findings_out

    # ─── EL KERNEL: un ciclo cognitivo ──────────────────────────────────────────

    async def cognitive_cycle(self) -> dict:
        """Percibir → Razonar → Decidir → Actuar → Aprender. Un ciclo."""
        cycle_id = uuid.uuid4().hex[:12]
        autonomy = await self.get_autonomy_level()
        auto_rank = _SEV_RANK.get(autonomy, 1)
        emitted = 0

        async with AsyncSessionLocal() as session:
            lotes = (await session.execute(
                select(Lote).where(Lote.status.in_([LoteStatus.ACTIVO, LoteStatus.ALERTA]))
            )).scalars().all()

            for lote in lotes:
                try:
                    reading = await self._latest_reading(session, lote.id)
                    if not reading:
                        continue
                    # PERCIBIR + RAZONAR
                    for f in self._derive_findings(lote, reading):
                        key = f"{lote.codigo}:{f['variable']}"
                        last = self._cooldown.get(key)
                        if last and (datetime.utcnow() - last).total_seconds() < _COOLDOWN_S:
                            continue
                        self._cooldown[key] = datetime.utcnow()
                        emitted += 1
                        await self._handle_finding(session, cycle_id, lote, reading, f, auto_rank)
                except Exception as e:
                    print(f"[kernel] error en lote {lote.codigo}: {e}")

        return {"cycle_id": cycle_id, "decisiones_emitidas": emitted, "autonomy": autonomy}

    async def _handle_finding(self, session, cycle_id, lote, reading, f, auto_rank):
        """DECIDIR → ACTUAR → APRENDER para un hallazgo."""
        sev_rank = _SEV_RANK.get(f["severidad"], 1)
        percepcion = {
            "variable": f["variable"], "valor": f["valor"], "limite": f["limite"],
            "temperatura": reading.temperatura, "humedad": reading.humedad, "nh3": reading.nh3,
        }

        # RAZONAR (narración opcional vía LLM rápido)
        contexto = (f"Lote {lote.codigo} ({lote.modulo}), día {lote.dia_actual}/{lote.dias_ciclo}. "
                    f"{f['titulo']}. Valor {f['valor']} vs límite {f['limite']}. "
                    f"Acción candidata: {WORKFLOWS.get(f['workflow_key'], {}).get('name', '—')}.")
        narr = await llm.narrate(f["role"], contexto)
        razonamiento = narr["texto"] or f["razonamiento"]

        decide_auto = sev_rank <= auto_rank
        outcome = DecisionOutcome.AUTO_EXECUTED if decide_auto else DecisionOutcome.PENDING_APPROVAL

        decision = AgentDecision(
            timestamp=datetime.utcnow(), cycle_id=cycle_id, lote_codigo=lote.codigo,
            modulo=lote.modulo, role=f["role"], severidad=f["severidad"], titulo=f["titulo"],
            percepcion=json.dumps(percepcion, default=str), razonamiento=razonamiento,
            accion_propuesta=f["workflow_key"], outcome=outcome, confianza=f["confianza"],
            modelo=narr["modelo"], tokens=narr["tokens"],
        )
        session.add(decision)
        await session.flush()

        await events.publish("decision", {
            "id": decision.id, "lote": lote.codigo, "modulo": lote.modulo, "role": f["role"],
            "severidad": f["severidad"], "titulo": f["titulo"], "razonamiento": razonamiento,
            "accion": f["workflow_key"], "outcome": outcome,
        })

        if decide_auto and f["workflow_key"]:
            # ACTUAR
            result = await self.execute_workflow(
                f["workflow_key"], lote.codigo,
                {**percepcion, "modulo": lote.modulo, "triggered_by": "agente_ia_auto"})
            decision.resultado = json.dumps(result, default=str)  # APRENDER (registrar outcome)
            await session.commit()
        elif not decide_auto and f["workflow_key"]:
            # Pedir aprobación humana (si no hay ya una pendiente para esta acción)
            existing = (await session.execute(
                select(Approval).where(and_(
                    Approval.lote_codigo == lote.codigo,
                    Approval.workflow_key == f["workflow_key"],
                    Approval.status == ApprovalStatus.PENDING)))).scalar_one_or_none()
            if not existing:
                appr = Approval(
                    decision_id=decision.id, lote_codigo=lote.codigo, titulo=f["titulo"],
                    rationale=razonamiento, workflow_key=f["workflow_key"],
                    input_data=json.dumps({**percepcion, "modulo": lote.modulo}, default=str),
                    severidad=f["severidad"], status=ApprovalStatus.PENDING)
                session.add(appr)
                await session.commit()
                await events.publish("approval", {
                    "id": appr.id, "lote": lote.codigo, "titulo": f["titulo"],
                    "severidad": f["severidad"], "accion": f["workflow_key"], "rationale": razonamiento})
            else:
                await session.commit()
        else:
            await session.commit()

    def _derive_findings(self, lote, reading) -> list[dict]:
        """Reglas determinísticas → hallazgos con severidad y acción candidata."""
        out = []
        T, H, N = THRESHOLDS["temperatura"], THRESHOLDS["humedad"], THRESHOLDS["nh3"]

        if reading.temperatura >= T["max"]:
            out.append(dict(variable="temperatura", severidad="critical", role="proceso",
                            valor=reading.temperatura, limite=T["max"], workflow_key="alta_temperatura",
                            titulo=f"Temperatura CRÍTICA {reading.temperatura:.1f}°C en {lote.modulo}",
                            razonamiento="Temperatura en zona de daño térmico; riesgo de colapso de colonia.",
                            confianza=0.9))
        elif reading.temperatura > T["opt_max"]:
            out.append(dict(variable="temperatura", severidad="warning", role="proceso",
                            valor=reading.temperatura, limite=T["opt_max"], workflow_key="alta_temperatura",
                            titulo=f"Temperatura sobre óptimo {reading.temperatura:.1f}°C en {lote.modulo}",
                            razonamiento="Sobre el óptimo; ventilación +15% suele resolver en ~22 min.",
                            confianza=0.82))

        if reading.nh3 >= N["critical"]:
            out.append(dict(variable="nh3", severidad="critical", role="proceso",
                            valor=reading.nh3, limite=N["critical"], workflow_key="nh3_elevado",
                            titulo=f"NH₃ CRÍTICO {reading.nh3:.1f} ppm en {lote.modulo}",
                            razonamiento="NH₃ crítico indica putrefacción del sustrato; extracción urgente.",
                            confianza=0.88))
        elif reading.nh3 >= N["warn"]:
            out.append(dict(variable="nh3", severidad="warning", role="proceso",
                            valor=reading.nh3, limite=N["warn"], workflow_key="nh3_elevado",
                            titulo=f"NH₃ elevado {reading.nh3:.1f} ppm en {lote.modulo}",
                            razonamiento="NH₃ acercándose a zona insegura; aumentar extracción de aire.",
                            confianza=0.8))

        if reading.humedad < H["min"] or reading.humedad > H["max"]:
            out.append(dict(variable="humedad", severidad="warning", role="proceso",
                            valor=reading.humedad, limite=H["min"] if reading.humedad < H["min"] else H["max"],
                            workflow_key="baja_humedad",
                            titulo=f"Humedad fuera de rango {reading.humedad:.1f}% en {lote.modulo}",
                            razonamiento="Humedad fuera de banda segura; ajustar nebulización/ventilación.",
                            confianza=0.75))

        if lote.dia_actual >= lote.dias_ciclo and lote.status != LoteStatus.COSECHA:
            out.append(dict(variable="ciclo", severidad="info", role="operaciones",
                            valor=float(lote.dia_actual), limite=float(lote.dias_ciclo),
                            workflow_key="ciclo_completo",
                            titulo=f"Ciclo completo: {lote.codigo} día {lote.dia_actual}/{lote.dias_ciclo}",
                            razonamiento="Lote alcanzó el día final; iniciar protocolo de cosecha.",
                            confianza=0.95))
        return out

    async def _latest_reading(self, session, lote_id):
        return (await session.execute(
            select(LecturaSensor).where(LecturaSensor.lote_id == lote_id)
            .order_by(desc(LecturaSensor.timestamp)).limit(1))).scalar_one_or_none()

    # ─── APROBACIONES (HITL) ────────────────────────────────────────────────────

    async def resolve_approval(self, approval_id: int, approve: bool, by: str = "operador", nota: str = "") -> dict:
        async with AsyncSessionLocal() as session:
            appr = await session.get(Approval, approval_id)
            if not appr:
                return {"error": "aprobación no encontrada"}
            if appr.status != ApprovalStatus.PENDING:
                return {"error": f"aprobación ya {appr.status}"}

            appr.status = ApprovalStatus.APPROVED if approve else ApprovalStatus.REJECTED
            appr.resolved_at = datetime.utcnow()
            appr.resolved_by = by
            appr.nota_operador = nota

            result = None
            if approve:
                try:
                    data = json.loads(appr.input_data or "{}")
                except json.JSONDecodeError:
                    data = {}
                result = await self.execute_workflow(
                    appr.workflow_key, appr.lote_codigo,
                    {**data, "triggered_by": f"aprobado_por_{by}"})

            # actualizar la decisión vinculada
            if appr.decision_id:
                dec = await session.get(AgentDecision, appr.decision_id)
                if dec:
                    dec.outcome = DecisionOutcome.APPROVED if approve else DecisionOutcome.REJECTED
                    if result is not None:
                        dec.resultado = json.dumps(result, default=str)
            await session.commit()

        await events.publish("approval_resolved", {
            "id": approval_id, "approved": approve, "by": by, "lote": appr.lote_codigo})
        return {"status": appr.status, "executed": result is not None, "result": result}

    # ─── EJECUCIÓN DE WORKFLOWS ─────────────────────────────────────────────────

    async def execute_workflow(self, workflow_key: str, lote_codigo: str, input_data: dict) -> dict:
        if workflow_key not in WORKFLOWS:
            return {"error": f"Workflow '{workflow_key}' no existe"}
        wf = WORKFLOWS[workflow_key]
        start = datetime.utcnow()

        async with AsyncSessionLocal() as session:
            log = WorkflowLog(
                workflow_name=wf["name"], triggered_by=input_data.get("triggered_by", "agente_ia"),
                status=WorkflowStatus.RUNNING, lote_codigo=lote_codigo,
                input_data=json.dumps(input_data, default=str))
            session.add(log)
            await session.flush()
            log_id = log.id
            try:
                result = await self._run_workflow_logic(workflow_key, lote_codigo, input_data, session)
                log.status = WorkflowStatus.COMPLETED
                log.result = json.dumps(result, default=str)
                log.duration_ms = int((datetime.utcnow() - start).total_seconds() * 1000)
                await session.commit()
                await events.publish("workflow", {
                    "id": log_id, "name": wf["name"], "lote": lote_codigo,
                    "status": "completed", "triggered_by": log.triggered_by})
                return {"status": "completed", "workflow": wf["name"], "result": result, "log_id": log_id}
            except Exception as e:
                log.status = WorkflowStatus.FAILED
                log.result = str(e)
                await session.commit()
                return {"status": "failed", "error": str(e)}

    async def _run_workflow_logic(self, key, lote_codigo, data, session):
        if key == "alta_temperatura":
            lote = await session.scalar(select(Lote).where(Lote.codigo == lote_codigo))
            if lote:
                session.add(Alerta(
                    lote_id=lote.id, severidad=AlertSeverity.WARNING, variable="temperatura",
                    valor_actual=data.get("valor", data.get("temperatura", 0)),
                    valor_limite=THRESHOLDS["temperatura"]["opt_max"],
                    mensaje=f"Alta temperatura en {lote.modulo}. Ventilación aumentada 15%.",
                    accion_tomada="Ventilación aumentada automáticamente +15%"))
                lote.status = LoteStatus.ALERTA
                lote.updated_at = datetime.utcnow()
                await session.flush()
            return {"ventilacion_ajustada": "+15%", "alerta_creada": True, "modulo": data.get("modulo")}

        if key == "ciclo_completo":
            lote = await session.scalar(select(Lote).where(Lote.codigo == lote_codigo))
            if lote:
                lote.status = LoteStatus.COSECHA
                lote.etapa_actual = StageEnum.COSECHA
                lote.updated_at = datetime.utcnow()
                session.add(Alerta(
                    lote_id=lote.id, severidad=AlertSeverity.INFO, variable="ciclo",
                    valor_actual=float(lote.dia_actual), valor_limite=float(lote.dias_ciclo),
                    mensaje=f"Lote {lote_codigo} completó {lote.dias_ciclo} días. Protocolo de cosecha iniciado.",
                    resuelta=False, accion_tomada="Equipo de cosecha notificado. Cribado programado."))
                await session.flush()
            return {"lote_status": "cosecha", "equipo_notificado": True, "protocolo": "cribado_iniciado"}

        if key == "nh3_elevado":
            lote = await session.scalar(select(Lote).where(Lote.codigo == lote_codigo))
            if lote:
                session.add(Alerta(
                    lote_id=lote.id, severidad=AlertSeverity.WARNING, variable="nh3",
                    valor_actual=data.get("valor", data.get("nh3", 0)), valor_limite=THRESHOLDS["nh3"]["warn"],
                    mensaje=f"NH₃ elevado en {lote.modulo}. Extracción aumentada.",
                    accion_tomada="Extracción de aire +20%. Compuerta abierta."))
                await session.flush()
            return {"extraccion_ajustada": "+20%", "compuerta": "abierta", "alerta_creada": True}

        if key == "baja_humedad":
            lote = await session.scalar(select(Lote).where(Lote.codigo == lote_codigo))
            if lote:
                session.add(Alerta(
                    lote_id=lote.id, severidad=AlertSeverity.WARNING, variable="humedad",
                    valor_actual=data.get("valor", data.get("humedad", 0)),
                    valor_limite=THRESHOLDS["humedad"]["opt_min"],
                    mensaje=f"Humedad fuera de rango en {lote.modulo}. Nebulización activada.",
                    accion_tomada="Nebulización 5 min. Re-medición en 15 min."))
                await session.flush()
            return {"nebulizacion": "5min", "alerta_creada": True}

        if key == "analisis_diario":
            rows = (await session.execute(
                select(LecturaSensor.modulo,
                       func.avg(LecturaSensor.temperatura).label("temp_avg"),
                       func.avg(LecturaSensor.humedad).label("hum_avg"),
                       func.max(LecturaSensor.nh3).label("nh3_max"),
                       func.count(LecturaSensor.id).label("n"))
                .where(LecturaSensor.timestamp >= datetime.utcnow() - timedelta(hours=24))
                .group_by(LecturaSensor.modulo))).all()
            summary = [{"modulo": r.modulo, "temp_avg": round(r.temp_avg, 2),
                        "hum_avg": round(r.hum_avg, 2), "nh3_max": round(r.nh3_max, 2), "lecturas": r.n}
                       for r in rows]
            return {"fecha": datetime.utcnow().date().isoformat(), "modulos": summary, "lotes_activos": len(summary)}

        if key == "bajo_rendimiento":
            lote = await session.scalar(select(Lote).where(Lote.codigo == lote_codigo))
            if not lote:
                return {"error": "lote no encontrado"}
            rendimiento = (lote.kg_larva_real / lote.kg_larva_proy) * 100 if lote.kg_larva_proy else 0
            return {"lote": lote_codigo, "rendimiento_actual": round(rendimiento, 1),
                    "kg_larva_real": lote.kg_larva_real, "kg_larva_proy": lote.kg_larva_proy,
                    "diagnostico": "Revisar temperatura días 3-5 y humedad días 7-9",
                    "recomendacion": "Aumentar frecuencia de riego en próximo lote"}

        return {"status": "ok"}
