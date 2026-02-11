"""
================================================================================
REGIME MODEL: FINANCIAL — regime_model_financial.py
================================================================================
Proyecto: Análisis de contexto económico-financiero para asignación dinámica
          en el S&P 500.

Propósito: Clasificar el régimen de condiciones financieras usando indicadores
           de volatilidad, crédito y estrés de mercado.

Filosofía:
    Este modelo responde a la pregunta: "¿Las condiciones financieras son
    favorables o adversas para activos de riesgo?" A diferencia del modelo
    macro (que mira la economía real con retraso), este modelo usa señales
    de MERCADO que se actualizan en tiempo real y reflejan las expectativas
    y el posicionamiento de los inversores.

    La distinción risk-on / risk-off es el marco más utilizado por
    inversores institucionales para describir regímenes de mercado
    (Ilmanen, 2011; Ang, 2014). Se basa en la idea de que hay periodos
    donde el apetito por riesgo es elevado (spreads bajos, VIX bajo,
    momentum positivo) y periodos de aversión al riesgo (VIX alto,
    spreads amplios, flight-to-quality).

Indicadores utilizados (del indicators.py):
    ┌──────────────────────────────────────┬────────┬──────────────────────┐
    │ Indicador                            │ Dir.   │ Justificación        │
    ├──────────────────────────────────────┼────────┼──────────────────────┤
    │ vol_vix_zscore_24m                   │ -      │ VIX alto = estrés    │
    │ vol_implied_vs_realized_6m           │ -      │ Prima riesgo alta=   │
    │                                      │        │ aversión al riesgo   │
    │ credit_hy_oas_zscore_24m             │ -      │ Spread alto = estrés │
    │ credit_hy_oas_3m_change              │ -      │ Spread subiendo=     │
    │                                      │        │ deterioro            │
    │ credit_riskon_riskoff_mom_6m         │ +      │ HYG/TLT subiendo =  │
    │                                      │        │ risk-on              │
    │ trend_drawdown                       │ +      │ Drawdown profundo =  │
    │                                      │        │ estrés severo        │
    └──────────────────────────────────────┴────────┴──────────────────────┘

    Lógica de selección:
    - VIX z-score (no nivel bruto) porque el "nivel normal" del VIX
      cambia a lo largo del tiempo. Un VIX de 20 era alto en 2017 pero
      bajo en 2020. El z-score adapta esto.
    - HY OAS z-score por la misma razón: los spreads "normales" varían.
    - Se incluye el drawdown de SPY como indicador de estrés porque
      drawdowns severos (-20%+) cambian el régimen independientemente
      de lo que digan volatilidad o crédito.
    - Se incluye el variance risk premium (implied vs realized) porque
      una prima elevada indica que el mercado anticipa más riesgo del
      observado recientemente.

Regímenes de salida:
    1 = RISK-ON   — Volatilidad baja, spreads ajustados, momentum favorable.
    0 = NEUTRAL   — Condiciones mixtas, transición.
   -1 = RISK-OFF  — Volatilidad elevada, spreads amplios, estrés.

Metodología:
    Idéntica al modelo macro: composite z-score expansivo con dirección,
    umbrales simétricos de ±0.5σ. Ver regime_model_macro.py para la
    justificación detallada del enfoque.

    Diferencia clave: algunos indicadores ya son z-scores (vol_vix_zscore_24m,
    credit_hy_oas_zscore_24m). Aun así, se les aplica el z-score expansivo
    encima para normalizar su distribución a lo largo de toda la muestra.
    Esto es correcto porque los z-scores rolling de 24m del indicators.py
    miden "extremos relativos a 2 años", mientras que aquí queremos
    "extremos relativos a toda la historia disponible".

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

MODEL_NAME = "financial"

# Indicadores requeridos y su dirección
# +1: valor alto → condiciones favorables (risk-on)
# -1: valor alto → condiciones desfavorables (risk-off)
INDICATORS: dict[str, int] = {
    "vol_vix_zscore_24m":            -1,  # VIX anómalamente alto = estrés
    "vol_implied_vs_realized_6m":    -1,  # Prima de riesgo alta = aversión
    "credit_hy_oas_zscore_24m":      -1,  # Spread HY elevado = estrés crédito
    "credit_hy_oas_3m_change":       -1,  # Spread subiendo = deterioro
    "credit_riskon_riskoff_mom_6m":  +1,  # HYG/TLT subiendo = risk-on
    "trend_drawdown":                +1,  # Drawdown profundo (muy negativo) = estrés
}

# Parámetros de clasificación
RISKON_THRESHOLD = +0.5
RISKOFF_THRESHOLD = -0.5
MIN_PERIODS = 24

# Etiquetas de régimen
REGIME_LABELS = {
    1: "risk_on",
    0: "neutral",
    -1: "risk_off",
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
    riskon_threshold: float = RISKON_THRESHOLD,
    riskoff_threshold: float = RISKOFF_THRESHOLD,
) -> pd.DataFrame:
    """
    Clasifica el régimen de condiciones financieras para cada fecha.

    Parámetros
    ----------
    indicators : pd.DataFrame
        DataFrame con los indicadores del indicators.py.
    min_periods : int
        Mínimo de meses para estadísticas expansivas.
    riskon_threshold : float
        Umbral del composite para clasificar como risk-on.
    riskoff_threshold : float
        Umbral (negativo) para clasificar como risk-off.

    Retorna
    -------
    pd.DataFrame con columnas:
        - regime_financial       : int (-1, 0, 1)
        - regime_financial_label : str
        - regime_financial_score : float composite z-score
    """
    logger.info(f"[{MODEL_NAME}] Clasificando régimen de condiciones financieras...")

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
            f"[{MODEL_NAME}]   {col:>40s} (dir={direction:+d}): "
            f"{n_valid} obs válidas"
        )

    # --- Composite score ---
    composite = directed_zscores.mean(axis=1)

    # --- Clasificar régimen ---
    regime = pd.Series(0, index=indicators.index, dtype=int, name="regime_financial")
    regime[composite > riskon_threshold] = 1
    regime[composite < riskoff_threshold] = -1
    regime[composite.isna()] = np.nan

    # --- Output ---
    result = pd.DataFrame(index=indicators.index)
    result["regime_financial"] = regime
    result["regime_financial_label"] = regime.map(REGIME_LABELS)
    result["regime_financial_score"] = composite

    # --- Log resumen ---
    valid_regimes = regime.dropna()
    if len(valid_regimes) > 0:
        counts = valid_regimes.value_counts().sort_index()
        total = len(valid_regimes)
        logger.info(f"[{MODEL_NAME}] Clasificación completada ({total} meses válidos):")
        for val, count in counts.items():
            label = REGIME_LABELS.get(int(val), "unknown")
            pct = 100 * count / total
            logger.info(f"[{MODEL_NAME}]   {int(val):+d} ({label:>12s}): {count:>4d} meses ({pct:5.1f}%)")

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
    return result["regime_financial"]