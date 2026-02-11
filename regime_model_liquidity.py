"""
================================================================================
REGIME MODEL: LIQUIDITY — regime_model_liquidity.py
================================================================================
Proyecto: Análisis de contexto económico-financiero para asignación dinámica
          en el S&P 500.

Propósito: Clasificar el régimen de política monetaria y liquidez usando
           indicadores de tipos de interés, curva de rendimientos e inflación.

Filosofía:
    Este modelo responde a la pregunta: "¿La política monetaria está
    facilitando o restringiendo las condiciones financieras?"

    La política monetaria de la Reserva Federal es el principal determinante
    de las condiciones de liquidez en los mercados estadounidenses. Cuando
    la Fed es acomodaticia (tipos bajos, tipo real negativo, curva
    empinada), los activos de riesgo tienden a beneficiarse. Cuando es
    restrictiva (tipos altos y subiendo, tipo real positivo, curva plana
    o invertida), las condiciones se endurecen.

    El régimen monetario opera en un horizonte distinto al macro y al
    financiero: los cambios de política monetaria tardan 6-18 meses en
    transmitirse a la economía real (el "lag monetario" clásico de
    Friedman). Por eso tiene sentido como modelo independiente.

Indicadores utilizados (del indicators.py):
    ┌────────────────────────────────┬────────┬──────────────────────────────┐
    │ Indicador                      │ Dir.   │ Justificación                │
    ├────────────────────────────────┼────────┼──────────────────────────────┤
    │ mon_real_rate                  │ -      │ Tipo real alto = restrictivo. │
    │                                │        │ Tipo real negativo = estímulo│
    │ mon_fedfunds_diff_12m         │ -      │ Fed subiendo tipos =         │
    │                                │        │ endurecimiento               │
    │ mon_yield_curve_level          │ +      │ Curva empinada = expectativas│
    │                                │        │ de crecimiento; invertida =  │
    │                                │        │ restricción excesiva         │
    │ mon_yield_curve_diff_6m        │ +      │ Curva empinándose = mejora;  │
    │                                │        │ aplanándose = deterioro      │
    │ infl_cpi_accel_6m             │ -      │ Inflación acelerando fuerza  │
    │                                │        │ a la Fed a restringir        │
    │ infl_breakeven_3m_change      │ -      │ Expectativas de inflación    │
    │                                │        │ al alza = más restricción    │
    │                                │        │ esperada                     │
    └────────────────────────────────┴────────┴──────────────────────────────┘

    Lógica de selección:
    - El tipo real (FEDFUNDS - CPI YoY) es la medida más directa de si
      la política es genuinamente restrictiva o expansiva. Un tipo real
      negativo significa que la Fed está por debajo de la inflación
      (estimulando); positivo = por encima (restringiendo).
    - La variación del Fed Funds en 12 meses captura la DIRECCIÓN de la
      política (hiking vs cutting cycle).
    - La curva de tipos refleja las expectativas del mercado sobre la
      política futura: una curva invertida indica que el mercado espera
      que la Fed ha ido demasiado lejos.
    - La inflación se incluye porque es el principal determinante de la
      reacción de la Fed. Si la inflación acelera, la Fed probablemente
      endurecerá; si desacelera, relajará.

Regímenes de salida:
    1 = ACOMODATICIO — Tipos bajos/bajando, tipo real negativo, curva sana.
    0 = NEUTRAL      — Transición, señales mixtas.
   -1 = RESTRICTIVO  — Tipos altos/subiendo, tipo real positivo, inflación
                       presionando.

Metodología:
    Composite z-score expansivo con dirección y umbrales simétricos de ±0.5σ.
    Idéntica al modelo macro (ver regime_model_macro.py para justificación).

Autor: Mauro Calvo Pérez y Jorge Fernández Beloso
Fecha: 2026-02
================================================================================
"""

import logging
from pathlib import Path

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# 1. CONFIGURACIÓN DEL MODELO
# ══════════════════════════════════════════════════════════════════════════════

MODEL_NAME = "liquidity"

# Indicadores requeridos y su dirección
# +1: valor alto → condiciones acomodaticias (liquidez abundante)
# -1: valor alto → condiciones restrictivas (liquidez escasa)
INDICATORS: dict[str, int] = {
    "mon_real_rate":              -1,  # Tipo real alto = restrictivo
    "mon_fedfunds_diff_12m":     -1,  # Fed subiendo tipos = endurecimiento
    "mon_yield_curve_level":     +1,  # Curva empinada = condiciones sanas
    "mon_yield_curve_diff_6m":   +1,  # Curva empinándose = mejora
    "infl_cpi_accel_6m":         -1,  # Inflación acelerando = presión restrictiva
    "infl_breakeven_3m_change":  -1,  # Expectativas inflación al alza = más restricción
}

# Parámetros de clasificación
ACCOMMODATIVE_THRESHOLD = +0.5
RESTRICTIVE_THRESHOLD = -0.5
MIN_PERIODS = 24

# Etiquetas de régimen
REGIME_LABELS = {
    1: "accommodative",
    0: "neutral",
    -1: "restrictive",
}


# ══════════════════════════════════════════════════════════════════════════════
# 2. FUNCIONES AUXILIARES
# ══════════════════════════════════════════════════════════════════════════════

def _expanding_zscore(series: pd.Series, min_periods: int = 24) -> pd.Series:
    """
    Z-score expansivo: usa solo datos hasta la fecha actual.

    z_t = (X_t - mean(X_1..X_t)) / std(X_1..X_t)

    Sin look-ahead bias. Los primeros min_periods valores son NaN.
    """
    expanding_mean = series.expanding(min_periods=min_periods).mean()
    expanding_std = series.expanding(min_periods=min_periods).std(ddof=1)
    expanding_std = expanding_std.replace(0, np.nan)
    return (series - expanding_mean) / expanding_std


# ══════════════════════════════════════════════════════════════════════════════
# 3. FUNCIÓN PRINCIPAL DEL MODELO
# ══════════════════════════════════════════════════════════════════════════════

def classify_regime(
    indicators: pd.DataFrame,
    min_periods: int = MIN_PERIODS,
    accommodative_threshold: float = ACCOMMODATIVE_THRESHOLD,
    restrictive_threshold: float = RESTRICTIVE_THRESHOLD,
) -> pd.DataFrame:
    """
    Clasifica el régimen de política monetaria/liquidez para cada fecha.

    Parámetros
    ----------
    indicators : pd.DataFrame
        DataFrame con los indicadores del indicators.py.
    min_periods : int
        Mínimo de meses para estadísticas expansivas.
    accommodative_threshold : float
        Umbral del composite para clasificar como acomodaticio.
    restrictive_threshold : float
        Umbral (negativo) para clasificar como restrictivo.

    Retorna
    -------
    pd.DataFrame con columnas:
        - regime_liquidity       : int (-1, 0, 1)
        - regime_liquidity_label : str
        - regime_liquidity_score : float composite z-score
    """
    logger.info(f"[{MODEL_NAME}] Clasificando régimen de liquidez/política monetaria...")

    # --- Verificar indicadores ---
    available = [col for col in INDICATORS if col in indicators.columns]
    missing = [col for col in INDICATORS if col not in indicators.columns]

    if missing:
        logger.warning(f"[{MODEL_NAME}] Indicadores faltantes: {missing}")
    if not available:
        raise ValueError(
            f"[{MODEL_NAME}] Ningún indicador disponible. "
            f"Se requieren: {list(INDICATORS.keys())}"
        )

    logger.info(f"[{MODEL_NAME}] Usando {len(available)}/{len(INDICATORS)} indicadores: {available}")

    # --- Calcular z-scores dirigidos ---
    directed_zscores = pd.DataFrame(index=indicators.index)

    for col in available:
        direction = INDICATORS[col]
        raw_series = indicators[col]

        zscore = _expanding_zscore(raw_series, min_periods=min_periods)
        directed_zscores[col] = zscore * direction

        n_valid = zscore.notna().sum()
        logger.info(
            f"[{MODEL_NAME}]   {col:>35s} (dir={direction:+d}): "
            f"{n_valid} obs válidas"
        )

    # --- Composite score ---
    composite = directed_zscores.mean(axis=1)

    # --- Clasificar régimen ---
    regime = pd.Series(0, index=indicators.index, dtype=int, name="regime_liquidity")
    regime[composite > accommodative_threshold] = 1
    regime[composite < restrictive_threshold] = -1
    regime[composite.isna()] = np.nan

    # --- Output ---
    result = pd.DataFrame(index=indicators.index)
    result["regime_liquidity"] = regime
    result["regime_liquidity_label"] = regime.map(REGIME_LABELS)
    result["regime_liquidity_score"] = composite

    # --- Log resumen ---
    valid_regimes = regime.dropna()
    if len(valid_regimes) > 0:
        counts = valid_regimes.value_counts().sort_index()
        total = len(valid_regimes)
        logger.info(f"[{MODEL_NAME}] Clasificación completada ({total} meses válidos):")
        for val, count in counts.items():
            label = REGIME_LABELS.get(int(val), "unknown")
            pct = 100 * count / total
            logger.info(f"[{MODEL_NAME}]   {int(val):+d} ({label:>14s}): {count:>4d} meses ({pct:5.1f}%)")

    return result


# ══════════════════════════════════════════════════════════════════════════════
# 4. INTERFAZ ESTÁNDAR
# ══════════════════════════════════════════════════════════════════════════════

def get_regime_series(indicators: pd.DataFrame) -> pd.Series:
    """
    Interfaz estándar: devuelve solo la serie de régimen.

    Punto de entrada para regime_selector.py.
    """
    result = classify_regime(indicators)
    return result["regime_liquidity"]