"""
BioFactor SCADA – Database layer
SQLite (file-based, portable, zero-config) via SQLAlchemy async.
Schema mirrors what a Postgres deployment would use – column types,
constraints and query patterns are identical.
"""

import asyncio
import random
import math
from datetime import datetime, timedelta
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, mapped_column, Mapped, relationship
from sqlalchemy import (
    String, Float, Integer, Boolean, DateTime, Text, ForeignKey,
    Enum as SAEnum, select, func, desc, and_, text
)
import enum

DATABASE_URL = "sqlite+aiosqlite:///./biofactor.db"

engine = create_async_engine(DATABASE_URL, echo=False)
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)

# ─── ENUMS ────────────────────────────────────────────────────────────────────

class StageEnum(str, enum.Enum):
    RECOLECCION   = "Recolección"
    RECEPCION     = "Recepción"
    REPRODUCCION  = "Reproducción"
    BIOCONVERSION = "Bioconversión"
    COSECHA       = "Cosecha"
    PROCESAMIENTO = "Procesamiento"
    DESPACHO      = "Despacho"

class LoteStatus(str, enum.Enum):
    ACTIVO   = "activo"
    COSECHA  = "cosecha"
    PROCESO  = "proceso"
    DESPACHADO = "despachado"
    ALERTA   = "alerta"

class AlertSeverity(str, enum.Enum):
    INFO    = "info"
    WARNING = "warning"
    CRITICAL = "critical"

class WorkflowStatus(str, enum.Enum):
    PENDING   = "pending"
    RUNNING   = "running"
    COMPLETED = "completed"
    FAILED    = "failed"

# ─── MODELS ──────────────────────────────────────────────────────────────────

class Base(DeclarativeBase):
    pass

class Lote(Base):
    __tablename__ = "lotes"

    id: Mapped[int]             = mapped_column(Integer, primary_key=True)
    codigo: Mapped[str]         = mapped_column(String(32), unique=True, nullable=False)
    fecha_inicio: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    dia_actual: Mapped[int]     = mapped_column(Integer, default=1)
    dias_ciclo: Mapped[int]     = mapped_column(Integer, default=14)
    kg_entrada: Mapped[float]   = mapped_column(Float, default=1000.0)
    kg_larva_proy: Mapped[float]  = mapped_column(Float, default=130.0)
    kg_frass_proy: Mapped[float]  = mapped_column(Float, default=450.0)
    kg_larva_real: Mapped[float]  = mapped_column(Float, default=0.0)
    kg_frass_real: Mapped[float]  = mapped_column(Float, default=0.0)
    etapa_actual: Mapped[str]   = mapped_column(String(64), default=StageEnum.BIOCONVERSION)
    status: Mapped[str]         = mapped_column(String(32), default=LoteStatus.ACTIVO)
    modulo: Mapped[str]         = mapped_column(String(16), default="MOD-01")
    notas: Mapped[str]          = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    lecturas: Mapped[list["LecturaSensor"]] = relationship("LecturaSensor", back_populates="lote", cascade="all, delete-orphan")
    alertas:  Mapped[list["Alerta"]]        = relationship("Alerta", back_populates="lote", cascade="all, delete-orphan")

class LecturaSensor(Base):
    __tablename__ = "lecturas_sensor"

    id: Mapped[int]             = mapped_column(Integer, primary_key=True)
    lote_id: Mapped[int]        = mapped_column(ForeignKey("lotes.id"), nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)
    temperatura: Mapped[float]  = mapped_column(Float, nullable=False)   # °C
    humedad: Mapped[float]      = mapped_column(Float, nullable=False)   # %
    nh3: Mapped[float]          = mapped_column(Float, nullable=False)   # ppm
    co2: Mapped[float]          = mapped_column(Float, nullable=False)   # ppm
    ph_sustrato: Mapped[float]  = mapped_column(Float, nullable=False)   # pH
    masa_larva_g: Mapped[float] = mapped_column(Float, nullable=False)   # gramos
    etapa: Mapped[str]          = mapped_column(String(64), nullable=False)
    modulo: Mapped[str]         = mapped_column(String(16), nullable=False)

    lote: Mapped["Lote"] = relationship("Lote", back_populates="lecturas")

class Alerta(Base):
    __tablename__ = "alertas"

    id: Mapped[int]              = mapped_column(Integer, primary_key=True)
    lote_id: Mapped[int]         = mapped_column(ForeignKey("lotes.id"), nullable=True)
    timestamp: Mapped[datetime]  = mapped_column(DateTime, default=datetime.utcnow, index=True)
    severidad: Mapped[str]       = mapped_column(String(16), nullable=False)
    variable: Mapped[str]        = mapped_column(String(64), nullable=False)
    mensaje: Mapped[str]         = mapped_column(Text, nullable=False)
    valor_actual: Mapped[float]  = mapped_column(Float, nullable=True)
    valor_limite: Mapped[float]  = mapped_column(Float, nullable=True)
    resuelta: Mapped[bool]       = mapped_column(Boolean, default=False)
    accion_tomada: Mapped[str]   = mapped_column(Text, default="")

    lote: Mapped["Lote"] = relationship("Lote", back_populates="alertas")

class WorkflowLog(Base):
    __tablename__ = "workflow_logs"

    id: Mapped[int]              = mapped_column(Integer, primary_key=True)
    timestamp: Mapped[datetime]  = mapped_column(DateTime, default=datetime.utcnow, index=True)
    workflow_name: Mapped[str]   = mapped_column(String(128), nullable=False)
    triggered_by: Mapped[str]    = mapped_column(String(64), default="agente_ia")
    status: Mapped[str]          = mapped_column(String(32), default=WorkflowStatus.PENDING)
    lote_codigo: Mapped[str]     = mapped_column(String(32), default="")
    input_data: Mapped[str]      = mapped_column(Text, default="")
    result: Mapped[str]          = mapped_column(Text, default="")
    duration_ms: Mapped[int]     = mapped_column(Integer, default=0)

class ProduccionDiaria(Base):
    __tablename__ = "produccion_diaria"

    id: Mapped[int]              = mapped_column(Integer, primary_key=True)
    fecha: Mapped[datetime]      = mapped_column(DateTime, nullable=False, index=True)
    lote_codigo: Mapped[str]     = mapped_column(String(32), nullable=False)
    dia_ciclo: Mapped[int]       = mapped_column(Integer, nullable=False)
    kg_larva_acum: Mapped[float] = mapped_column(Float, default=0.0)
    kg_frass_acum: Mapped[float] = mapped_column(Float, default=0.0)
    temp_promedio: Mapped[float] = mapped_column(Float, default=0.0)
    hum_promedio: Mapped[float]  = mapped_column(Float, default=0.0)
    nh3_max: Mapped[float]       = mapped_column(Float, default=0.0)
    alertas_count: Mapped[int]   = mapped_column(Integer, default=0)
    rendimiento_pct: Mapped[float] = mapped_column(Float, default=0.0)

# ─── INIT + SEED ──────────────────────────────────────────────────────────────

async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    await seed_data()

async def seed_data():
    async with AsyncSessionLocal() as session:
        # Check if already seeded
        result = await session.execute(select(func.count()).select_from(Lote))
        if result.scalar() > 0:
            return

        now = datetime.utcnow()

        # Create 3 lotes
        lotes_data = [
            dict(
                codigo="BSF-2026-06A",
                fecha_inicio=now - timedelta(days=7),
                dia_actual=8,
                etapa_actual=StageEnum.BIOCONVERSION,
                status=LoteStatus.ACTIVO,
                modulo="MOD-01",
                kg_larva_real=98.4,
                kg_frass_real=312.0,
                notas="Lote principal. Temperatura estable. Buen progreso."
            ),
            dict(
                codigo="BSF-2026-06B",
                fecha_inicio=now - timedelta(days=2),
                dia_actual=3,
                etapa_actual=StageEnum.REPRODUCCION,
                status=LoteStatus.ALERTA,
                modulo="MOD-02",
                kg_larva_real=11.2,
                kg_frass_real=28.0,
                notas="Alerta de temperatura detectada en módulo 02."
            ),
            dict(
                codigo="BSF-2026-05C",
                fecha_inicio=now - timedelta(days=13),
                dia_actual=14,
                etapa_actual=StageEnum.COSECHA,
                status=LoteStatus.COSECHA,
                modulo="MOD-03",
                kg_larva_real=128.0,
                kg_frass_real=441.0,
                notas="Ciclo completo. Listo para cosecha y procesamiento."
            ),
        ]

        lotes = []
        for ld in lotes_data:
            lote = Lote(**ld)
            session.add(lote)
            lotes.append(lote)

        await session.flush()

        # Seed sensor readings: last 8 days for lote A (2 readings/hour = 48/day = 384 total, we seed 200)
        random.seed(42)
        readings = []
        for lote in lotes:
            days_back = lote.dia_actual
            base_temp = 29.5 if lote.modulo == "MOD-01" else 31.2
            base_hum  = 66.0
            base_nh3  = 38.0
            for h in range(min(days_back * 6, 96)):  # 6 readings/day
                ts = now - timedelta(hours=(days_back * 6 - h) * 4)
                t  = base_temp + math.sin(h * 0.3) * 1.5 + random.gauss(0, 0.3)
                hu = base_hum  + math.sin(h * 0.2) * 3   + random.gauss(0, 0.5)
                nh = base_nh3  + h * 0.05 + random.gauss(0, 1.5)
                ph = 6.8 + random.gauss(0, 0.15)
                co2= 800 + h * 2 + random.gauss(0, 20)
                mass_g = (lote.kg_larva_real / (days_back * 6)) * h * 1000
                r = LecturaSensor(
                    lote_id=lote.id,
                    timestamp=ts,
                    temperatura=round(t, 2),
                    humedad=round(hu, 2),
                    nh3=round(nh, 2),
                    co2=round(co2, 1),
                    ph_sustrato=round(ph, 2),
                    masa_larva_g=round(mass_g, 1),
                    etapa=lote.etapa_actual,
                    modulo=lote.modulo
                )
                readings.append(r)
                session.add(r)

        # Seed alerts
        alerts_data = [
            dict(lote_id=lotes[1].id, severidad=AlertSeverity.WARNING,
                 variable="temperatura", valor_actual=32.8, valor_limite=32.0,
                 mensaje="Temperatura en MOD-02 supera límite óptimo (32.8°C > 32°C). Verificar ventilación.",
                 resuelta=False, accion_tomada=""),
            dict(lote_id=lotes[0].id, severidad=AlertSeverity.WARNING,
                 variable="nh3", valor_actual=42.0, valor_limite=40.0,
                 mensaje="NH₃ en MOD-01 llegando al límite seguro (42 ppm). Aumentar extracción de aire.",
                 resuelta=False, accion_tomada=""),
            dict(lote_id=lotes[2].id, severidad=AlertSeverity.INFO,
                 variable="ciclo", valor_actual=14.0, valor_limite=14.0,
                 mensaje="Lote BSF-05C completó ciclo de 14 días. Iniciar protocolo de cosecha.",
                 resuelta=True, accion_tomada="Cosecha programada para hoy 08:00"),
        ]
        for i, ad in enumerate(alerts_data):
            a = Alerta(**ad, timestamp=now - timedelta(minutes=30 - i*10))
            session.add(a)

        # Seed workflow logs
        wf_data = [
            dict(workflow_name="Alerta temperatura → Ajuste ventilación",
                 triggered_by="agente_ia", status=WorkflowStatus.COMPLETED,
                 lote_codigo="BSF-2026-06B",
                 input_data='{"temp": 32.8, "modulo": "MOD-02"}',
                 result='{"accion": "ventilacion_aumentada_15pct", "ts": "auto"}',
                 duration_ms=320),
            dict(workflow_name="Ciclo completo → Notificación cosecha",
                 triggered_by="agente_ia", status=WorkflowStatus.COMPLETED,
                 lote_codigo="BSF-2026-05C",
                 input_data='{"dia": 14, "lote": "BSF-2026-05C"}',
                 result='{"notificacion": "enviada", "responsable": "operador_01"}',
                 duration_ms=145),
            dict(workflow_name="Análisis diario de producción",
                 triggered_by="scheduler", status=WorkflowStatus.COMPLETED,
                 lote_codigo="ALL",
                 input_data='{"fecha": "2026-06-10"}',
                 result='{"lotes_analizados": 3, "alertas_generadas": 2}',
                 duration_ms=890),
        ]
        for wd in wf_data:
            wf = WorkflowLog(**wd, timestamp=now - timedelta(minutes=20))
            session.add(wf)

        # Seed produccion_diaria
        for lote in lotes:
            for d in range(1, lote.dia_actual + 1):
                fecha = lote.fecha_inicio + timedelta(days=d - 1)
                progress = d / 14
                pd_entry = ProduccionDiaria(
                    fecha=fecha,
                    lote_codigo=lote.codigo,
                    dia_ciclo=d,
                    kg_larva_acum=round(lote.kg_larva_real * progress * random.uniform(0.95, 1.05), 1),
                    kg_frass_acum=round(lote.kg_frass_real * progress * random.uniform(0.95, 1.05), 1),
                    temp_promedio=round(29.5 + random.gauss(0, 0.4), 2),
                    hum_promedio=round(66.0 + random.gauss(0, 1.0), 2),
                    nh3_max=round(35.0 + d * 0.4 + random.gauss(0, 1), 2),
                    alertas_count=1 if d in [3, 7, 10] else 0,
                    rendimiento_pct=round(min(100, progress * 100 * random.uniform(0.97, 1.02)), 1)
                )
                session.add(pd_entry)

        await session.commit()
        print("[DB] Seed completo.")
