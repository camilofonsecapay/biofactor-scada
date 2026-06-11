"""
BioFactor — Motor predictivo
1) predict_yield: predice kg de larva al cierre del ciclo (día 14) usando
   el histórico de lotes en produccion_diaria + proyección lineal.
2) simulate_intervention: what-if biológico — estima el impacto de un
   cambio de condiciones (temp/humedad/NH₃) sobre rendimiento y economía.

Pure-Python (sin numpy) para máxima portabilidad en el deploy.
"""

from sqlalchemy import select
from economics import PRECIOS, TRM_REFERENCIA

# Respuesta del rendimiento de larva a cada variable: curva suave (cuadrática)
# centrada en el óptimo. Cualquier desviación del óptimo reduce el rendimiento,
# de forma progresiva — así el simulador muestra un gradiente realista.
RESPUESTA = {
    "temperatura": {"opt": 30.0, "k": 0.020, "piso": 0.10},
    "humedad":     {"opt": 66.0, "k": 0.0015, "piso": 0.30},
    "nh3":         {"opt": 25.0, "k": 0.0010, "piso": 0.20},
}


def _factor_variable(nombre: str, valor: float) -> float:
    """Factor de rendimiento (0..1) para una variable dada su desviación del óptimo."""
    r = RESPUESTA.get(nombre)
    if not r or valor is None:
        return 1.0
    dev = valor - r["opt"]
    return max(r["piso"], 1.0 - r["k"] * dev * dev)


async def predict_yield(session, lote) -> dict:
    """Predice el rendimiento final del lote. Disponible desde día 1."""
    from database import ProduccionDiaria  # import local para evitar ciclos

    # Histórico: rendimiento final de cada lote en produccion_diaria
    r = await session.execute(select(ProduccionDiaria).order_by(ProduccionDiaria.fecha))
    rows = r.scalars().all()

    # Agrupar por lote_codigo y tomar el rendimiento del último día registrado
    final_por_lote: dict[str, float] = {}
    maxdia_por_lote: dict[str, int] = {}
    for p in rows:
        if p.lote_codigo == lote.codigo:
            continue  # no usar el propio lote
        if p.dia_ciclo >= maxdia_por_lote.get(p.lote_codigo, 0):
            maxdia_por_lote[p.lote_codigo] = p.dia_ciclo
            final_por_lote[p.lote_codigo] = p.rendimiento_pct

    ratios = [v / 100.0 for v in final_por_lote.values() if v]

    if len(ratios) >= 2:
        ratio_prom = sum(ratios) / len(ratios)
        pred_hist = lote.kg_larva_proy * ratio_prom
        confianza = "alta" if len(ratios) >= 5 else "media"
    else:
        pred_hist = None
        confianza = "baja"

    # Proyección lineal desde el avance actual
    progress = lote.dia_actual / lote.dias_ciclo if lote.dias_ciclo > 0 else 0.0
    progress = max(progress, 0.01)
    pred_lineal = lote.kg_larva_real / progress

    # Blend: si hay historia, ponderar histórico + lineal por avance del ciclo
    if pred_hist is not None:
        w = min(progress, 1.0)  # a más avanzado el ciclo, más peso a lo observado
        pred_final = pred_lineal * w + pred_hist * (1 - w)
    else:
        pred_final = pred_lineal

    rendimiento_pred = (pred_final / lote.kg_larva_proy * 100) if lote.kg_larva_proy else 0.0
    dias_restantes = max(0, lote.dias_ciclo - lote.dia_actual)
    valor_pred = pred_final * PRECIOS["larva_fresca"]

    return {
        "lote": lote.codigo,
        "dia_actual": lote.dia_actual,
        "dias_restantes": dias_restantes,
        "kg_larva_actual": round(lote.kg_larva_real, 1),
        "kg_larva_predicho_final": round(pred_final, 1),
        "kg_frass_predicho_final": round(pred_final * 3.46, 1),  # ratio histórico BSF
        "rendimiento_predicho_pct": round(rendimiento_pred, 1),
        "confianza": confianza,
        "n_lotes_historicos": len(ratios),
        "valor_predicho_cop": round(valor_pred),
        "valor_predicho_usd": round(valor_pred / TRM_REFERENCIA),
        "alerta": "bajo_rendimiento" if pred_final < lote.kg_larva_proy * 0.85 else None,
    }


def simulate_intervention(lote, reading: dict, cambios: dict) -> dict:
    """
    What-if: dado el estado actual (reading) y cambios propuestos
    (ej. {"temperatura": +2}), estima el impacto en rendimiento y economía.
    """
    variables = ["temperatura", "humedad", "nh3"]
    base_vals = {v: (reading or {}).get(v) for v in variables}
    new_vals = dict(base_vals)
    for k, delta in (cambios or {}).items():
        if k in new_vals and new_vals[k] is not None:
            new_vals[k] = new_vals[k] + delta

    factor_base = 1.0
    factor_new = 1.0
    for v in variables:
        factor_base *= _factor_variable(v, base_vals[v])
        factor_new  *= _factor_variable(v, new_vals[v])

    # Proyección de larva final bajo cada escenario
    progress = lote.dia_actual / lote.dias_ciclo if lote.dias_ciclo > 0 else 0.0
    progress = max(progress, 0.01)
    larva_proj_base = (lote.kg_larva_real / progress) * 1.0  # ya refleja condiciones pasadas
    # El factor relativo aplica al rendimiento de los días restantes
    dias_restantes = max(0, lote.dias_ciclo - lote.dia_actual)
    frac_restante = dias_restantes / lote.dias_ciclo if lote.dias_ciclo else 0.0

    rel = (factor_new / factor_base) if factor_base > 0 else 1.0
    larva_proj_new = larva_proj_base * (1 - frac_restante + frac_restante * rel)

    delta_kg = larva_proj_new - larva_proj_base
    delta_cop = delta_kg * PRECIOS["larva_fresca"]

    return {
        "lote": lote.codigo,
        "cambios": cambios,
        "estado_base": base_vals,
        "estado_simulado": new_vals,
        "factor_rendimiento_base": round(factor_base, 3),
        "factor_rendimiento_simulado": round(factor_new, 3),
        "kg_larva_proyectado_base": round(larva_proj_base, 1),
        "kg_larva_proyectado_simulado": round(larva_proj_new, 1),
        "delta_kg_larva": round(delta_kg, 1),
        "delta_valor_cop": round(delta_cop),
        "recomendacion": (
            "Favorable: el cambio mejora el rendimiento proyectado." if delta_kg > 0.5
            else "Desfavorable: el cambio reduce el rendimiento proyectado." if delta_kg < -0.5
            else "Impacto marginal sobre el rendimiento."
        ),
    }
