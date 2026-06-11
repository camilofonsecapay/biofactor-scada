# BioFactor SCADA

Sistema de monitoreo inteligente para bioconversión de residuos orgánicos con Mosca Soldado Negra (BSF).

## Stack
- **Backend:** FastAPI + SQLAlchemy async + SQLite
- **Agente IA:** Análisis en tiempo real, 6 workflows automatizados
- **Frontend:** HTML/CSS/JS vanilla — UI corporativa

## Endpoints principales
- `GET /` — Dashboard SCADA
- `GET /api/dashboard` — KPIs + lotes + alertas
- `GET /api/lotes` — Lista de lotes
- `POST /api/lotes` — Crear lote
- `GET /api/lecturas` — Lecturas de sensores (con filtros)
- `GET /api/alertas` — Alertas activas/resueltas
- `POST /api/workflows/execute` — Ejecutar workflow IA
- `POST /api/agent/query` — Query en lenguaje natural
- `POST /api/agent/analyze` — Análisis completo en tiempo real
- `GET /docs` — Swagger UI

## Variables SCADA monitoreadas
- Temperatura (28–32°C óptimo BSF)
- Humedad del sustrato (60–70%)
- NH₃ / Calidad de aire (< 50 ppm)
- Ciclos de producción (trazabilidad lote a lote)

## Deploy
Railway: `uvicorn main:app --host 0.0.0.0 --port $PORT`
