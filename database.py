"""
BioFactor — Database layer (AI-native)
Dual-mode: usa PostgreSQL (asyncpg) si DATABASE_URL está seteada,
si no cae a SQLite (aiosqlite) — portable, zero-config para el demo.
El schema es idéntico en ambos motores; los tipos JSON se persisten
como Text (json.dumps) para máxima compatibilidad.
"""

import os
import re
import json
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

# ─── ENGINE (dual-mode) ────────────────────────────────────────────────────────

def _resolve_database_url() -> tuple[str, bool, dict]:
    """Resuelve la URL de conexión. Devuelve (url, is_sqlite, connect_args)."""
    raw = (os.getenv("DATABASE_URL") or "").strip()
    if not raw:
        # Ruta relativa al módulo (no al cwd) → robusto sin importar desde dónde se lance
        db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "biofactor.db")
        return f"sqlite+aiosqlite:///{db_path}", True, {}

    url = raw
    # Render/Heroku inyectan postgres:// o postgresql:// — normalizar a asyncpg
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql+asyncpg://", 1)
    elif url.startswith("postgresql://"):
        url = url.replace("postgresql://", "postgresql+asyncpg://", 1)

    connect_args: dict = {}
    # asyncpg no entiende ?sslmode= en la URL (eso es libpq) — traducir a ssl
    if "sslmode=" in url:
        url = re.sub(r"[?&]sslmode=[^&]+", "", url)
        connect_args["ssl"] = True
    return url, False, connect_args


DATABASE_URL, IS_SQLITE, _CONNECT_ARGS = _resolve_database_url()

_engine_kwargs: dict = {"echo": False}
if not IS_SQLITE:
    _engine_kwargs.update(pool_size=10, max_overflow=20, pool_pre_ping=True)
if _CONNECT_ARGS:
    _engine_kwargs["connect_args"] = _CONNECT_ARGS

engine = create_async_engine(DATABASE_URL, **_engine_kwargs)
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)

DB_BACKEND = "sqlite" if IS_SQLITE else "postgres"

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

class DecisionOutcome(str, enum.Enum):
    AUTO_EXECUTED    = "auto_executed"     # el dial permitió ejecución autónoma
    PENDING_APPROVAL = "pending_approval"  # requiere humano
    APPROVED         = "approved"
    REJECTED         = "rejected"
    OBSERVED         = "observed"          # solo observación, sin acción

class ApprovalStatus(str, enum.Enum):
    PENDING  = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED  = "expired"

# ─── MODELS ──────────────────────────────────────────────────────────────────

class Base(DeclarativeBase):
    pass

class Planta(Base):
    """Scaffold multi-tenant: una planta de bioconversión (base del modelo SaaS)."""
    __tablename__ = "plantas"

    id: Mapped[int]       = mapped_column(Integer, primary_key=True)
    nombre: Mapped[str]   = mapped_column(String(128), nullable=False)
    ciudad: Mapped[str]   = mapped_column(String(64), default="Montería")
    pais: Mapped[str]     = mapped_column(String(32), default="Colombia")
    slug: Mapped[str]     = mapped_column(String(32), unique=True, nullable=False)
    activa: Mapped[bool]  = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

class Lote(Base):
    __tablename__ = "lotes"

    id: Mapped[int]             = mapped_column(Integer, primary_key=True)
    planta_id: Mapped[int]      = mapped_column(ForeignKey("plantas.id"), nullable=True)
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

# ─── MODELOS AGÉNTICOS (núcleo AI-native) ──────────────────────────────────────

class AgentDecision(Base):
    """Traza de razonamiento del kernel: percepción → razonamiento → acción.
    Es el registro auditable que hace visible la autonomía del agente."""
    __tablename__ = "agent_decisions"

    id: Mapped[int]             = mapped_column(Integer, primary_key=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)
    cycle_id: Mapped[str]       = mapped_column(String(36), default="", index=True)
    lote_codigo: Mapped[str]    = mapped_column(String(32), default="")
    modulo: Mapped[str]         = mapped_column(String(16), default="")
    role: Mapped[str]           = mapped_column(String(24), default="proceso")  # proceso|financiero|calidad|operaciones
    severidad: Mapped[str]      = mapped_column(String(16), default=AlertSeverity.INFO)
    titulo: Mapped[str]         = mapped_column(String(160), default="")
    percepcion: Mapped[str]     = mapped_column(Text, default="")   # JSON: señales observadas
    razonamiento: Mapped[str]   = mapped_column(Text, default="")   # texto: hipótesis/causa raíz
    accion_propuesta: Mapped[str] = mapped_column(String(64), default="")  # workflow_key
    outcome: Mapped[str]        = mapped_column(String(24), default=DecisionOutcome.OBSERVED)
    confianza: Mapped[float]    = mapped_column(Float, default=0.0)  # 0..1
    modelo: Mapped[str]         = mapped_column(String(32), default="deterministico")
    tokens: Mapped[int]         = mapped_column(Integer, default=0)
    latencia_ms: Mapped[int]    = mapped_column(Integer, default=0)
    resultado: Mapped[str]      = mapped_column(Text, default="")   # JSON: outcome real (se llena luego)

class Approval(Base):
    """Cola de aprobación humana (human-in-the-loop). El agente propone
    intervenciones críticas; el operador aprueba/rechaza."""
    __tablename__ = "approvals"

    id: Mapped[int]             = mapped_column(Integer, primary_key=True)
    decision_id: Mapped[int]    = mapped_column(ForeignKey("agent_decisions.id"), nullable=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)
    lote_codigo: Mapped[str]    = mapped_column(String(32), default="")
    titulo: Mapped[str]         = mapped_column(String(160), default="")
    rationale: Mapped[str]      = mapped_column(Text, default="")
    workflow_key: Mapped[str]   = mapped_column(String(64), default="")
    input_data: Mapped[str]     = mapped_column(Text, default="")   # JSON
    severidad: Mapped[str]      = mapped_column(String(16), default=AlertSeverity.CRITICAL)
    status: Mapped[str]         = mapped_column(String(16), default=ApprovalStatus.PENDING, index=True)
    resolved_at: Mapped[datetime] = mapped_column(DateTime, nullable=True)
    resolved_by: Mapped[str]    = mapped_column(String(64), default="")
    nota_operador: Mapped[str]  = mapped_column(Text, default="")

class AgentMemory(Base):
    """Memoria / aprendizajes del agente. Cierra el loop: registra qué
    intervenciones funcionaron y ajusta baselines por módulo."""
    __tablename__ = "agent_memory"

    id: Mapped[int]             = mapped_column(Integer, primary_key=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)
    scope: Mapped[str]          = mapped_column(String(32), default="global")  # global|MOD-01|lote:CODE
    clave: Mapped[str]          = mapped_column(String(64), default="")
    valor: Mapped[str]          = mapped_column(Text, default="")   # JSON o texto
    nota: Mapped[str]           = mapped_column(Text, default="")

class SystemConfig(Base):
    """Configuración del sistema (key/value). Incluye el dial de autonomía."""
    __tablename__ = "system_config"

    id: Mapped[int]   = mapped_column(Integer, primary_key=True)
    clave: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    valor: Mapped[str] = mapped_column(Text, default="")
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

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

        # Planta default (multi-tenant scaffold)
        planta = Planta(nombre="BioFactor Montería", ciudad="Montería",
                        pais="Colombia", slug="monteria", activa=True)
        session.add(planta)
        await session.flush()

        # Config inicial: dial de autonomía.
        # autonomy_level: severidad MÁXIMA que el agente ejecuta solo.
        #   "info" = nada auto (todo a aprobación) | "warning" = rutinas auto, críticas a aprobación | "critical" = todo auto
        session.add(SystemConfig(clave="autonomy_level", valor="warning"))
        session.add(SystemConfig(clave="llm_enabled", valor="auto"))  # auto = usa LLM si hay key

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
            lote = Lote(planta_id=planta.id, **ld)
            session.add(lote)
            lotes.append(lote)

        await session.flush()

        # Seed sensor readings
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

        # Seed memoria del agente (aprendizajes iniciales — da contexto al kernel)
        memorias = [
            dict(scope="MOD-02", clave="bias_temperatura",
                 valor=json.dumps({"offset_c": 1.2}),
                 nota="MOD-02 corre ~1.2°C más caliente que MOD-01 por ubicación. Compensar al evaluar."),
            dict(scope="global", clave="intervencion_ventilacion",
                 valor=json.dumps({"delta_pct": 15, "tiempo_resolucion_min": 22}),
                 nota="Ventilación +15% históricamente baja temperatura ~1.5°C en ~22 min."),
            dict(scope="global", clave="ratio_frass_larva",
                 valor=json.dumps({"ratio": 3.46}),
                 nota="Relación histórica frass:larva ≈ 3.46 en lotes completados."),
        ]
        for m in memorias:
            session.add(AgentMemory(**m, timestamp=now - timedelta(days=1)))

        # Seed un par de decisiones del agente (para que el feed no arranque vacío)
        decisiones = [
            dict(lote_codigo="BSF-2026-06B", modulo="MOD-02", role="proceso",
                 severidad=AlertSeverity.WARNING, titulo="Temperatura sobre óptimo en MOD-02",
                 percepcion=json.dumps({"temperatura": 32.8, "tendencia": "+0.3°C/30min", "umbral": 32.0}),
                 razonamiento="Temperatura 0.8°C sobre el óptimo con tendencia al alza. Memoria: MOD-02 corre +1.2°C; ventilación +15% resuelve en ~22min. Severidad warning → dentro del nivel de autonomía actual.",
                 accion_propuesta="alta_temperatura", outcome=DecisionOutcome.AUTO_EXECUTED,
                 confianza=0.82, modelo="deterministico"),
            dict(lote_codigo="BSF-2026-05C", modulo="MOD-03", role="operaciones",
                 severidad=AlertSeverity.INFO, titulo="Ciclo de 14 días completo",
                 percepcion=json.dumps({"dia_actual": 14, "dias_ciclo": 14}),
                 razonamiento="Lote alcanzó el día 14. Protocolo de cosecha aplica. Acción informativa, no requiere intervención crítica.",
                 accion_propuesta="ciclo_completo", outcome=DecisionOutcome.AUTO_EXECUTED,
                 confianza=0.95, modelo="deterministico"),
        ]
        for i, dec in enumerate(decisiones):
            session.add(AgentDecision(**dec, timestamp=now - timedelta(minutes=18 - i*5),
                                      cycle_id="seed"))

        await session.commit()
        print(f"[DB] Seed completo. Backend={DB_BACKEND}")
