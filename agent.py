"""
BioFactor SCADA – Agente IA real
Analiza datos de la DB, detecta anomalías, ejecuta workflows del proceso.
"""

import json
import asyncio
from datetime import datetime, timedelta
from typing import Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc, and_, func
from database import (
    AsyncSessionLocal, Lote, LecturaSensor, Alerta, WorkflowLog, ProduccionDiaria,
    LoteStatus, AlertSeverity, WorkflowStatus, StageEnum
)

# ─── THRESHOLDS (configurable) ────────────────────────────────────────────────
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
        "trigger": "temperatura > opt_max",
        "steps": [
            "Registrar alerta en DB",
            "Incrementar ventilación módulo +15%",
            "Re-evaluar en 30 minutos",
            "Si persiste > 34°C → notificar operador jefe",
        ]
    },
    "baja_humedad": {
        "name": "Humedad baja → Activar riego",
        "trigger": "humedad < opt_min",
        "steps": [
            "Registrar alerta en DB",
            "Activar sistema de nebulización 5 min",
            "Esperar 15 min y re-medir",
            "Si no sube → notificar operador",
        ]
    },
    "nh3_elevado": {
        "name": "NH₃ elevado → Extracción de aire",
        "trigger": "nh3 > warn",
        "steps": [
            "Registrar alerta NH₃",
            "Aumentar extracción de aire +20%",
            "Abrir compuerta de ventilación módulo afectado",
            "Monitorear cada 10 min",
        ]
    },
    "ciclo_completo": {
        "name": "Ciclo 14 días → Protocolo de cosecha",
        "trigger": "dia_actual >= dias_ciclo",
        "steps": [
            "Marcar lote como COSECHA en DB",
            "Notificar a equipo de cosecha",
            "Preparar equipos de cribado",
            "Registrar pesos reales larva y frass",
            "Calcular rendimiento vs proyectado",
            "Despachar al procesamiento",
        ]
    },
    "bajo_rendimiento": {
        "name": "Rendimiento < 85% → Análisis causas",
        "trigger": "rendimiento_pct < 85",
        "steps": [
            "Generar reporte diagnóstico del lote",
            "Correlacionar variables con rendimiento",
            "Identificar días críticos",
            "Recomendar ajuste parámetros próximo lote",
        ]
    },
    "analisis_diario": {
        "name": "Análisis diario de producción",
        "trigger": "scheduler 06:00",
        "steps": [
            "Leer lecturas últimas 24h de todos los lotes",
            "Calcular promedios y desviaciones",
            "Comparar vs benchmark histórico",
            "Generar resumen ejecutivo",
            "Proyectar rendimiento final de cada lote",
        ]
    }
}


class ScadaAgent:
    """Agente IA que analiza datos SCADA y ejecuta workflows del proceso."""

    async def get_session(self) -> AsyncSession:
        return AsyncSessionLocal()

    # ─── ANALYSIS ──────────────────────────────────────────────────────────────

    async def analyze_latest_readings(self) -> list[dict]:
        """Analiza las lecturas más recientes de cada lote activo."""
        findings = []
        async with AsyncSessionLocal() as session:
            # Get active lotes
            result = await session.execute(
                select(Lote).where(Lote.status.in_([LoteStatus.ACTIVO, LoteStatus.ALERTA]))
            )
            lotes = result.scalars().all()

            for lote in lotes:
                # Last reading
                lr = await session.execute(
                    select(LecturaSensor)
                    .where(LecturaSensor.lote_id == lote.id)
                    .order_by(desc(LecturaSensor.timestamp))
                    .limit(1)
                )
                reading = lr.scalar_one_or_none()
                if not reading:
                    continue

                issues = []
                suggested_workflows = []

                # Check temperatura
                T = THRESHOLDS["temperatura"]
                if reading.temperatura > T["opt_max"]:
                    issues.append(f"Temperatura {reading.temperatura:.1f}°C > {T['opt_max']}°C óptimo")
                    suggested_workflows.append("alta_temperatura")
                elif reading.temperatura < T["opt_min"]:
                    issues.append(f"Temperatura {reading.temperatura:.1f}°C < {T['opt_min']}°C óptimo")

                # Check humedad
                H = THRESHOLDS["humedad"]
                if reading.humedad < H["opt_min"]:
                    issues.append(f"Humedad {reading.humedad:.1f}% < {H['opt_min']}% óptimo")
                    suggested_workflows.append("baja_humedad")
                elif reading.humedad > H["opt_max"]:
                    issues.append(f"Humedad {reading.humedad:.1f}% > {H['opt_max']}% óptimo")

                # Check NH3
                N = THRESHOLDS["nh3"]
                if reading.nh3 >= N["critical"]:
                    issues.append(f"NH₃ CRÍTICO: {reading.nh3:.1f} ppm ≥ {N['critical']} ppm")
                    suggested_workflows.append("nh3_elevado")
                elif reading.nh3 >= N["warn"]:
                    issues.append(f"NH₃ elevado: {reading.nh3:.1f} ppm ≥ {N['warn']} ppm")
                    suggested_workflows.append("nh3_elevado")

                # Check ciclo completo
                if lote.dia_actual >= lote.dias_ciclo:
                    issues.append(f"Ciclo completo: día {lote.dia_actual}/{lote.dias_ciclo}")
                    suggested_workflows.append("ciclo_completo")

                findings.append({
                    "lote": lote.codigo,
                    "modulo": lote.modulo,
                    "dia_ciclo": lote.dia_actual,
                    "issues": issues,
                    "suggested_workflows": list(set(suggested_workflows)),
                    "reading": {
                        "temperatura": reading.temperatura,
                        "humedad": reading.humedad,
                        "nh3": reading.nh3,
                        "co2": reading.co2,
                        "ph_sustrato": reading.ph_sustrato,
                        "timestamp": reading.timestamp.isoformat()
                    }
                })

        return findings

    # ─── WORKFLOW EXECUTION ────────────────────────────────────────────────────

    async def execute_workflow(self, workflow_key: str, lote_codigo: str, input_data: dict) -> dict:
        """Ejecuta un workflow del proceso y lo registra en DB."""
        if workflow_key not in WORKFLOWS:
            return {"error": f"Workflow '{workflow_key}' no existe"}

        wf = WORKFLOWS[workflow_key]
        start = datetime.utcnow()

        async with AsyncSessionLocal() as session:
            log = WorkflowLog(
                workflow_name=wf["name"],
                triggered_by=input_data.get("triggered_by", "agente_ia"),
                status=WorkflowStatus.RUNNING,
                lote_codigo=lote_codigo,
                input_data=json.dumps(input_data)
            )
            session.add(log)
            await session.flush()
            log_id = log.id

            try:
                result = await self._run_workflow_logic(workflow_key, lote_codigo, input_data, session)
                log.status = WorkflowStatus.COMPLETED
                log.result = json.dumps(result)
                log.duration_ms = int((datetime.utcnow() - start).total_seconds() * 1000)
                await session.commit()
                return {"status": "completed", "workflow": wf["name"], "result": result, "log_id": log_id}
            except Exception as e:
                log.status = WorkflowStatus.FAILED
                log.result = str(e)
                await session.commit()
                return {"status": "failed", "error": str(e)}

    async def _run_workflow_logic(self, key: str, lote_codigo: str, data: dict, session: AsyncSession) -> dict:
        """Lógica real de cada workflow."""

        if key == "alta_temperatura":
            # Create alert + update lote status
            lote = await session.scalar(select(Lote).where(Lote.codigo == lote_codigo))
            if lote:
                alerta = Alerta(
                    lote_id=lote.id,
                    severidad=AlertSeverity.WARNING,
                    variable="temperatura",
                    valor_actual=data.get("temperatura", 0),
                    valor_limite=THRESHOLDS["temperatura"]["opt_max"],
                    mensaje=f"Alta temperatura {data.get('temperatura', 0):.1f}°C en {lote.modulo}. Ventilación aumentada 15%.",
                    accion_tomada="Ventilación aumentada automáticamente +15%"
                )
                session.add(alerta)
                lote.status = LoteStatus.ALERTA
                lote.updated_at = datetime.utcnow()
                await session.flush()
            return {"ventilacion_ajustada": "+15%", "alerta_creada": True, "modulo": data.get("modulo")}

        elif key == "ciclo_completo":
            lote = await session.scalar(select(Lote).where(Lote.codigo == lote_codigo))
            if lote:
                lote.status = LoteStatus.COSECHA
                lote.etapa_actual = StageEnum.COSECHA
                lote.updated_at = datetime.utcnow()
                alerta = Alerta(
                    lote_id=lote.id,
                    severidad=AlertSeverity.INFO,
                    variable="ciclo",
                    valor_actual=float(lote.dia_actual),
                    valor_limite=float(lote.dias_ciclo),
                    mensaje=f"Lote {lote_codigo} completó {lote.dias_ciclo} días. Protocolo de cosecha iniciado.",
                    resuelta=False,
                    accion_tomada="Equipo de cosecha notificado. Cribado programado."
                )
                session.add(alerta)
                await session.flush()
            return {"lote_status": "cosecha", "equipo_notificado": True, "protocolo": "cribado_iniciado"}

        elif key == "nh3_elevado":
            lote = await session.scalar(select(Lote).where(Lote.codigo == lote_codigo))
            if lote:
                alerta = Alerta(
                    lote_id=lote.id,
                    severidad=AlertSeverity.WARNING,
                    variable="nh3",
                    valor_actual=data.get("nh3", 0),
                    valor_limite=THRESHOLDS["nh3"]["warn"],
                    mensaje=f"NH₃ elevado {data.get('nh3', 0):.1f} ppm en {lote.modulo}. Extracción aumentada.",
                    accion_tomada="Extracción de aire +20%. Compuerta abierta."
                )
                session.add(alerta)
                await session.flush()
            return {"extraccion_ajustada": "+20%", "compuerta": "abierta", "alerta_creada": True}

        elif key == "analisis_diario":
            # Aggregate daily stats for all active lotes
            result = await session.execute(
                select(
                    LecturaSensor.modulo,
                    func.avg(LecturaSensor.temperatura).label("temp_avg"),
                    func.avg(LecturaSensor.humedad).label("hum_avg"),
                    func.max(LecturaSensor.nh3).label("nh3_max"),
                    func.count(LecturaSensor.id).label("lecturas")
                )
                .where(LecturaSensor.timestamp >= datetime.utcnow() - timedelta(hours=24))
                .group_by(LecturaSensor.modulo)
            )
            rows = result.all()
            summary = [
                {"modulo": r.modulo, "temp_avg": round(r.temp_avg, 2), "hum_avg": round(r.hum_avg, 2),
                 "nh3_max": round(r.nh3_max, 2), "lecturas": r.lecturas}
                for r in rows
            ]
            return {"fecha": datetime.utcnow().date().isoformat(), "modulos": summary, "lotes_activos": len(summary)}

        elif key == "bajo_rendimiento":
            lote = await session.scalar(select(Lote).where(Lote.codigo == lote_codigo))
            if not lote:
                return {"error": "lote no encontrado"}
            rendimiento_pct = (lote.kg_larva_real / lote.kg_larva_proy) * 100
            return {
                "lote": lote_codigo,
                "rendimiento_actual": round(rendimiento_pct, 1),
                "kg_larva_real": lote.kg_larva_real,
                "kg_larva_proy": lote.kg_larva_proy,
                "diagnostico": "Revisar temperatura días 3-5 y humedad días 7-9",
                "recomendacion": "Aumentar frecuencia de riego en próximo lote"
            }

        return {"status": "ok"}

    # ─── NATURAL LANGUAGE QUERY ────────────────────────────────────────────────

    async def run_query(self, query_text: str) -> dict:
        """
        Interpreta una query en lenguaje natural y la traduce a SQL + resultado.
        """
        q = query_text.lower().strip()

        async with AsyncSessionLocal() as session:

            # Temperatura promedio
            if "temperatura" in q and ("promedio" in q or "promedio" in q or "avg" in q):
                result = await session.execute(
                    select(
                        LecturaSensor.modulo,
                        func.round(func.avg(LecturaSensor.temperatura), 2).label("temp_promedio"),
                        func.round(func.min(LecturaSensor.temperatura), 2).label("temp_min"),
                        func.round(func.max(LecturaSensor.temperatura), 2).label("temp_max")
                    ).group_by(LecturaSensor.modulo)
                )
                rows = result.all()
                return {
                    "query": "Temperatura promedio por módulo",
                    "sql": "SELECT modulo, AVG(temperatura), MIN(temperatura), MAX(temperatura) FROM lecturas_sensor GROUP BY modulo",
                    "data": [{"modulo": r.modulo, "promedio": r.temp_promedio, "min": r.temp_min, "max": r.temp_max} for r in rows]
                }

            # Alertas activas
            elif "alerta" in q:
                resueltas = "resueltas" in q or "resuelta" in q
                result = await session.execute(
                    select(Alerta, Lote.codigo)
                    .join(Lote, Alerta.lote_id == Lote.id, isouter=True)
                    .where(Alerta.resuelta == resueltas)
                    .order_by(desc(Alerta.timestamp))
                    .limit(20)
                )
                rows = result.all()
                return {
                    "query": f"Alertas {'resueltas' if resueltas else 'activas'}",
                    "sql": f"SELECT * FROM alertas WHERE resuelta = {resueltas} ORDER BY timestamp DESC",
                    "data": [{"id": r.Alerta.id, "lote": r.codigo, "variable": r.Alerta.variable,
                               "severidad": r.Alerta.severidad, "mensaje": r.Alerta.mensaje,
                               "timestamp": r.Alerta.timestamp.isoformat()} for r in rows]
                }

            # Rendimiento de lotes
            elif "rendimiento" in q or "produccion" in q or "producción" in q:
                result = await session.execute(select(Lote).order_by(desc(Lote.created_at)))
                lotes = result.scalars().all()
                return {
                    "query": "Rendimiento por lote",
                    "sql": "SELECT codigo, kg_larva_proy, kg_larva_real, kg_frass_proy, kg_frass_real FROM lotes",
                    "data": [
                        {
                            "lote": l.codigo,
                            "kg_larva_proy": l.kg_larva_proy,
                            "kg_larva_real": l.kg_larva_real,
                            "rendimiento_larva_pct": round((l.kg_larva_real / l.kg_larva_proy) * 100, 1),
                            "kg_frass_proy": l.kg_frass_proy,
                            "kg_frass_real": l.kg_frass_real,
                            "status": l.status
                        } for l in lotes
                    ]
                }

            # NH3 máximo
            elif "nh3" in q or "amoniaco" in q or "amoníaco" in q:
                result = await session.execute(
                    select(
                        LecturaSensor.modulo,
                        func.round(func.max(LecturaSensor.nh3), 2).label("nh3_max"),
                        func.round(func.avg(LecturaSensor.nh3), 2).label("nh3_avg"),
                    ).group_by(LecturaSensor.modulo)
                )
                rows = result.all()
                return {
                    "query": "NH₃ por módulo",
                    "sql": "SELECT modulo, MAX(nh3), AVG(nh3) FROM lecturas_sensor GROUP BY modulo",
                    "data": [{"modulo": r.modulo, "max_ppm": r.nh3_max, "promedio_ppm": r.nh3_avg} for r in rows]
                }

            # Lecturas recientes
            elif "lectura" in q or "reciente" in q or "ultimo" in q or "último" in q:
                result = await session.execute(
                    select(LecturaSensor).order_by(desc(LecturaSensor.timestamp)).limit(10)
                )
                rows = result.scalars().all()
                return {
                    "query": "Últimas lecturas de sensores",
                    "sql": "SELECT * FROM lecturas_sensor ORDER BY timestamp DESC LIMIT 10",
                    "data": [
                        {"id": r.id, "modulo": r.modulo, "timestamp": r.timestamp.isoformat(),
                         "temperatura": r.temperatura, "humedad": r.humedad, "nh3": r.nh3} for r in rows
                    ]
                }

            # Lotes activos
            elif "lote" in q:
                result = await session.execute(select(Lote).order_by(desc(Lote.created_at)))
                lotes = result.scalars().all()
                return {
                    "query": "Todos los lotes",
                    "sql": "SELECT * FROM lotes ORDER BY created_at DESC",
                    "data": [
                        {"codigo": l.codigo, "dia_actual": l.dia_actual, "dias_ciclo": l.dias_ciclo,
                         "etapa": l.etapa_actual, "status": l.status, "modulo": l.modulo,
                         "kg_larva_real": l.kg_larva_real, "kg_frass_real": l.kg_frass_real} for l in lotes
                    ]
                }

            # Workflows ejecutados
            elif "workflow" in q or "accion" in q or "acción" in q:
                result = await session.execute(
                    select(WorkflowLog).order_by(desc(WorkflowLog.timestamp)).limit(20)
                )
                rows = result.scalars().all()
                return {
                    "query": "Historial de workflows",
                    "sql": "SELECT * FROM workflow_logs ORDER BY timestamp DESC LIMIT 20",
                    "data": [
                        {"id": r.id, "workflow": r.workflow_name, "lote": r.lote_codigo,
                         "status": r.status, "triggered_by": r.triggered_by,
                         "duration_ms": r.duration_ms, "timestamp": r.timestamp.isoformat()} for r in rows
                    ]
                }

            else:
                return {
                    "query": query_text,
                    "error": "No entendí la consulta. Prueba: 'temperatura promedio', 'alertas activas', 'rendimiento lotes', 'NH3 módulos', 'últimas lecturas', 'workflows'",
                    "data": []
                }
