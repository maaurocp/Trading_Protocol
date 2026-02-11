"""
================================================================================
REGIME MODEL: MACRO — regime_model_macro.py
================================================================================
Proyecto: Análisis de contexto económico-financiero para asignación dinámica
          en el S&P 500.

Propósito: Clasificar el régimen macroeconómico usando exclusivamente
           indicadores de ciclo económico real.

Filosofía:
    Este modelo responde a la pregunta: "¿En qué fase del ciclo económico
    estamos?" La respuesta se basa en los dos pilares que el NBER Business
    Cycle Dating Committee utiliza para definir recesiones: actividad
    productiva (producción industrial) y mercado laboral (desempleo).
    Se añade la curva de tipos como indicador adelantado complementario.

Indicadores utilizados (del indicators.py):
    ┌──────────────────────────────┬────────────────┬──────────────────────┐
    │ Indicador                    │ Dirección      │ Justificación        │
    ├──────────────────────────────┼────────────────┼──────────────────────┤
    │ cycle_indpro_yoy             │ + (alto=bueno) │ Crecimiento actividad│
    │ cycle_indpro_accel           │ + (alto=bueno) │ Momento de inflexión │
    │ cycle_unemployment_yoy_diff  │ - (alto=malo)  │ Deterioro laboral    │
    │ cycle_unemployment_3m_diff   │ - (alto=malo)  │ Deterioro reciente   │
    │ mon_yield_curve_level        │ + (alto=bueno) │ Curva normal=sana    │
    └──────────────────────────────┴────────────────┴──────────────────────┘

    Dirección:
    - "+" significa que valores altos indican condiciones favorables (expansión).
    - "-" significa que valores altos indican deterioro (contracción).

Regímenes de salida:
    1 = EXPANSIÓN   — Economía en crecimiento, empleo mejorando, curva sana.
    0 = NEUTRAL     — Señales mixtas, transición, desaceleración moderada.
   -1 = CONTRACCIÓN — Economía en deterioro, empleo empeorando, estrés.

Metodología — Composite Z-Score con ventana expansiva:

    1. Para cada indicador, se calcula un z-score EXPANSIVO (expanding):
       z_t = (X_t - mean(X_1..X_t)) / std(X_1..X_t)
       Esto usa únicamente información pasada → sin look-ahead bias.
       Se requiere un mínimo de 24 meses para que la estadística sea estable.

    2. Se aplica la DIRECCIÓN: indicadores donde "alto = malo" (desempleo)
       se multiplican por -1. Así, un z-score positivo SIEMPRE significa
       "condiciones favorables".

    3. Se promedian los z-scores dirigidos en un composite score.

    4. Se clasifica con umbrales FIJOS y SIMÉTRICOS:
       - Composite > +0.5  →  EXPANSIÓN (1)
       - Composite < -0.5  →  CONTRACCIÓN (-1)
       - Resto             →  NEUTRAL (0)

       El umbral de ±0.5σ es una convención estándar en investigación
       cuantitativa (medio sigma). No se ha optimizado mirando resultados
       históricos; es un punto de corte razonable que separa condiciones
       moderadamente por encima/debajo de la media histórica.

Validación disponible:
    cycle_nber_recession se incluye en el output para comparación ex-post,
    pero NO se usa como input del modelo.

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

# Nombre identificativo del modelo (usado por regime_selector.py)
MODEL_NAME = "macro"

# Indicadores requeridos y su dirección
# Dirección +1: valor alto → condiciones favorables (expansión)
# Dirección -1: valor alto → condiciones desfavorables (contracción)
INDICATORS: dict[str, int] = {
    "cycle_indpro_yoy":            +1,  # Crecimiento industrial positivo = expansión
    "cycle_indpro_accel":          +1,  # Aceleración = economía ganando fuerza
    "cycle_unemployment_yoy_diff": -1,  # Desempleo subiendo = deterioro
    "cycle_unemployment_3m_diff":  -1,  # Deterioro laboral reciente
    "mon_yield_curve_level":       +1,  # Curva positiva = expectativas sanas
}

# Parámetros de clasificación
EXPANSION_THRESHOLD = +0.5   # Composite z > +0.5σ → Expansión
CONTRACTION_THRESHOLD = -0.5  # Composite z < -0.5σ → Contracción
MIN_PERIODS = 24  # Mínimo de meses para calcular estadísticas expansivas

# Etiquetas de régimen
REGIME_LABELS = {
    1: "expansion",
    0: "neutral",
    -1: "contraction",
}

# Columna de validación (NO es input del modelo)
VALIDATION_COL = "cycle_nber_recession"


# ══════════════════════════════════════════════════════════════════════════════
# 2. FUNCIONES AUXILIARES
# ══════════════════════════════════════════════════════════════════════════════

def _expanding_zscore(series: pd.Series, min_periods: int = 24) -> pd.Series:
    """
    Calcula el z-score expansivo de una serie.

    Para cada fecha t, el z-score se calcula usando SOLO datos desde
    el inicio hasta t (no datos futuros):

        z_t = (X_t - mean(X_1..X_t)) / std(X_1..X_t)

    Los primeros `min_periods` valores serán NaN porque no hay
    suficientes datos para una estimación estable.

    Parámetros
    ----------
    series : pd.Series
        Serie temporal del indicador.
    min_periods : int
        Número mínimo de observaciones antes de calcular.

    Retorna
    -------
    pd.Series : z-scores expansivos.
    """
    expanding_mean = series.expanding(min_periods=min_periods).mean()
    expanding_std = series.expanding(min_periods=min_periods).std(ddof=1)

    # Proteger contra std = 0 (serie constante)
    expanding_std = expanding_std.replace(0, np.nan)

    return (series - expanding_mean) / expanding_std


# ══════════════════════════════════════════════════════════════════════════════
# 3. FUNCIÓN PRINCIPAL DEL MODELO
# ══════════════════════════════════════════════════════════════════════════════

def classify_regime(
    indicators: pd.DataFrame,
    min_periods: int = MIN_PERIODS,
    expansion_threshold: float = EXPANSION_THRESHOLD,
    contraction_threshold: float = CONTRACTION_THRESHOLD,
) -> pd.DataFrame:
    """
    Clasifica el régimen macroeconómico para cada fecha.

    Parámetros
    ----------
    indicators : pd.DataFrame
        DataFrame con los indicadores del indicators.py.
        Debe contener las columnas definidas en INDICATORS.
    min_periods : int
        Mínimo de meses para estadísticas expansivas.
    expansion_threshold : float
        Umbral del composite z-score para clasificar como expansión.
    contraction_threshold : float
        Umbral (negativo) para clasificar como contracción.

    Retorna
    -------
    pd.DataFrame con columnas:
        - regime_macro       : int (-1, 0, 1) régimen clasificado
        - regime_macro_label : str etiqueta legible
        - regime_macro_score : float composite z-score (para diagnóstico)
    """
    logger.info(f"[{MODEL_NAME}] Clasificando régimen macroeconómico...")

    # --- Verificar disponibilidad de indicadores ---
    available = [col for col in INDICATORS if col in indicators.columns]
    missing = [col for col in INDICATORS if col not in indicators.columns]

    if missing:
        logger.warning(f"[{MODEL_NAME}] Indicadores faltantes: {missing}")
    if not available:
        raise ValueError(f"[{MODEL_NAME}] Ningún indicador disponible. Se requieren: {list(INDICATORS.keys())}")

    logger.info(f"[{MODEL_NAME}] Usando {len(available)}/{len(INDICATORS)} indicadores: {available}")

    # --- Calcular z-scores dirigidos ---
    directed_zscores = pd.DataFrame(index=indicators.index)

    for col in available:
        direction = INDICATORS[col]
        raw_series = indicators[col]

        # Z-score expansivo (sin look-ahead)
        zscore = _expanding_zscore(raw_series, min_periods=min_periods)

        # Aplicar dirección: multiplicar por +1 o -1
        # Resultado: z positivo SIEMPRE = condiciones favorables
        directed_zscores[col] = zscore * direction

        # Log de diagnóstico
        n_valid = zscore.notna().sum()
        logger.info(
            f"[{MODEL_NAME}]   {col:>35s} (dir={direction:+d}): "
            f"{n_valid} obs válidas"
        )

    # --- Composite score: promedio de z-scores dirigidos ---
    # Se usa la media (no suma) para que el composite sea comparable
    # independientemente del número de indicadores disponibles.
    composite = directed_zscores.mean(axis=1)

    # --- Clasificar régimen ---
    regime = pd.Series(0, index=indicators.index, dtype=int, name="regime_macro")
    regime[composite > expansion_threshold] = 1
    regime[composite < contraction_threshold] = -1

    # Donde el composite es NaN (periodos iniciales), el régimen es NaN
    regime[composite.isna()] = np.nan

    # --- Construir output ---
    result = pd.DataFrame(index=indicators.index)
    result["regime_macro"] = regime
    result["regime_macro_label"] = regime.map(REGIME_LABELS)
    result["regime_macro_score"] = composite

    # Añadir NBER para validación si disponible (NO como input)
    if VALIDATION_COL in indicators.columns:
        result["nber_recession_validation"] = indicators[VALIDATION_COL]

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
# 4. INTERFAZ ESTÁNDAR (usada por regime_selector.py)
# ══════════════════════════════════════════════════════════════════════════════

def get_regime_series(indicators: pd.DataFrame) -> pd.Series:
    """
    Interfaz estándar: devuelve solo la serie de régimen.

    Esta función es el punto de entrada que usa regime_selector.py.
    Todos los modelos de régimen deben implementar esta función con
    la misma firma.

    Parámetros
    ----------
    indicators : pd.DataFrame
        Dataset completo de indicadores.

    Retorna
    -------
    pd.Series : serie temporal de régimen (int: -1, 0, 1).
    """
    result = classify_regime(indicators)
    return result["regime_macro"]