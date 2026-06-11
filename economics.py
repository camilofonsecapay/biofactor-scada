"""
BioFactor — Motor económico
Convierte el estado biológico del proceso en valor económico:
ingreso acumulado, proyección al cierre del ciclo, desglose y margen.
Es lo que hace que el dashboard hable el idioma de un inversor, no solo
el de un sensor.
"""

# Precios de referencia (COP). Frass certificado puede valer 1200-1800/kg.
PRECIOS = {
    "larva_fresca":  4950,   # COP/kg
    "harina_bsf":   12100,   # COP/kg
    "frass":          550,   # COP/kg
    "frass_certificado": 1500,
    "gate_fee":     70000,   # COP/tonelada de residuo recibida
}

# Balance de masa típico por tonelada de residuo procesada.
BALANCE_MASA = {
    "entrada_residuo_kg": 1000,
    "larva_output_kg":     130,   # 13% conversión
    "frass_output_kg":     450,   # 45% conversión
    "perdidas_pct":         42,   # humedad + CO2 + calor
}

# Costo operativo aproximado por tonelada procesada (energía, mano de obra,
# sustrato complementario). Referencia para margen.
COSTO_OPERATIVO_POR_TON = 95000  # COP/ton de residuo

TRM_REFERENCIA = 4200  # COP/USD


def valor_lote(lote) -> dict:
    """Calcula el valor económico (acumulado y proyectado) de un lote."""
    p = PRECIOS

    # Ingreso ya generado con kg reales
    ingreso_larva = lote.kg_larva_real * p["larva_fresca"]
    ingreso_frass = lote.kg_frass_real * p["frass"]
    gate_fee_lote = (lote.kg_entrada / 1000.0) * p["gate_fee"]
    ingreso_acum = ingreso_larva + ingreso_frass + gate_fee_lote

    # Proyección al final del ciclo (extrapolación por avance del ciclo)
    progress = lote.dia_actual / lote.dias_ciclo if lote.dias_ciclo > 0 else 0.0
    progress = max(progress, 0.01)
    larva_final = lote.kg_larva_real / progress
    frass_final = lote.kg_frass_real / progress

    valor_proy = (larva_final * p["larva_fresca"]
                  + frass_final * p["frass"]
                  + gate_fee_lote)

    costo_op = (lote.kg_entrada / 1000.0) * COSTO_OPERATIVO_POR_TON
    margen_proy = valor_proy - costo_op
    margen_pct = (margen_proy / valor_proy * 100) if valor_proy > 0 else 0.0

    rendimiento_pct = (lote.kg_larva_real / lote.kg_larva_proy * 100) if lote.kg_larva_proy else 0.0

    return {
        "lote": lote.codigo,
        "dia": lote.dia_actual,
        "dias_ciclo": lote.dias_ciclo,
        "ingreso_acumulado_cop": round(ingreso_acum),
        "proyeccion_final_cop": round(valor_proy),
        "proyeccion_final_usd": round(valor_proy / TRM_REFERENCIA),
        "costo_operativo_cop": round(costo_op),
        "margen_proyectado_cop": round(margen_proy),
        "margen_pct": round(margen_pct, 1),
        "desglose": {
            "larva_cop": round(ingreso_larva),
            "frass_cop": round(ingreso_frass),
            "gate_fee_cop": round(gate_fee_lote),
        },
        "kg_larva_real": lote.kg_larva_real,
        "kg_larva_proy_final": round(larva_final, 1),
        "kg_frass_proy_final": round(frass_final, 1),
        "rendimiento_vs_proyectado_pct": round(rendimiento_pct, 1),
        "upside_certificacion_cop": round(frass_final * (p["frass_certificado"] - p["frass"])),
    }


def resumen_financiero(lotes) -> dict:
    """Agrega el valor económico de todos los lotes."""
    detalle = [valor_lote(l) for l in lotes]
    total_proy = sum(d["proyeccion_final_cop"] for d in detalle)
    total_acum = sum(d["ingreso_acumulado_cop"] for d in detalle)
    total_margen = sum(d["margen_proyectado_cop"] for d in detalle)
    total_upside = sum(d["upside_certificacion_cop"] for d in detalle)
    return {
        "total_valor_proyectado_cop": round(total_proy),
        "total_valor_proyectado_usd": round(total_proy / TRM_REFERENCIA),
        "total_ingreso_acumulado_cop": round(total_acum),
        "total_margen_proyectado_cop": round(total_margen),
        "upside_frass_certificado_cop": round(total_upside),
        "trm_referencia": TRM_REFERENCIA,
        "precios_referencia": PRECIOS,
        "lotes": detalle,
    }
