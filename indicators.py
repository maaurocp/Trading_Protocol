"""
================================================================================
INDICATORS MODULE — indicators.py
================================================================================
Proyecto: Análisis de contexto económico-financiero para asignación dinámica
          en el S&P 500.

Propósito: PASO 3 — Construcción de un universo amplio de indicadores
           económicos y financieros que describan el contexto de mercado.

           Este módulo NO define estrategia, NO genera señales, NO selecciona
           indicadores y NO evalúa performance. Es un laboratorio estable de
           indicadores reutilizable para múltiples versiones futuras del sistema.

Filosofía de diseño:

    1. CONTEXTO, NO PREDICCIÓN: Los indicadores miden el estado del entorno
       económico-financiero. No intentan predecir la dirección de precios.
       La pregunta no es "¿subirá SPY?" sino "¿en qué tipo de entorno
       estamos operando?"

    2. UNIVERSO AMPLIO, SIN FILTRADO: Se generan múltiples variantes
       razonables (distintas ventanas, transformaciones) para que la
       selección se realice en fases posteriores con criterios explícitos.

    3. INTERPRETABILIDAD: Cada indicador tiene significado económico
       documentado. No se incluyen indicadores de caja negra.

    4. SIN LOOK-AHEAD BIAS: Todo cálculo usa únicamente información
       pasada y presente. Las series macroeconómicas con retraso de
       publicación se documentan explícitamente.

    5. TRAZABILIDAD: Cada indicador tiene metadatos completos (categoría,
       fuente, lag natural, limitaciones).

Categorías:
    1. Tendencia de mercado
    2. Volatilidad y riesgo
    3. Valoración relativa
    4. Ciclo económico
    5. Política monetaria
    6. Estrés financiero y crédito
    7. Inflación y expectativas
    8. Amplitud / participación cross-asset

Entrada:  data/processed/market_monthly.csv
          data/processed/macro_monthly.csv
Salida:   data/indicators/
          ├── indicators_full.csv
          └── indicators_metadata.csv

Autor: Mauro Calvo Pérez y Jorge Fernández Beloso
Fecha: 2026-02
================================================================================
"""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN DE LOGGING
# ──────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# 1. CONFIGURACIÓN Y RUTAS
# ══════════════════════════════════════════════════════════════════════════════

PROCESSED_DATA_DIR = Path("data/processed")
INDICATORS_DIR = Path("data/indicators")

# Columnas esperadas de los datasets procesados (del preprocessing.py)
MARKET_COLS = ["SPY", "VIX", "TLT", "TIP", "LQD", "HYG", "GLD"]
MACRO_COLS = [
    "CPI", "UNRATE", "FEDFUNDS", "DFF", "T10Y2Y",
    "GS10", "GS2", "INDPRO", "USREC", "T10YIE", "HY_OAS",
]


# ══════════════════════════════════════════════════════════════════════════════
# 2. CARGA DE DATOS
# ══════════════════════════════════════════════════════════════════════════════

def load_processed_data(
    processed_dir: Path = PROCESSED_DATA_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Carga los datasets procesados del Paso 2.

    Lee market_monthly.csv y macro_monthly.csv, verifica la presencia de
    las columnas esperadas y reporta cualquier discrepancia.

    Retorna
    -------
    tuple[pd.DataFrame, pd.DataFrame] : (market, macro)
    """
    logger.info("Cargando datos procesados...")

    market_path = processed_dir / "market_monthly.csv"
    macro_path = processed_dir / "macro_monthly.csv"

    market = pd.read_csv(market_path, index_col=0, parse_dates=True)
    macro = pd.read_csv(macro_path, index_col=0, parse_dates=True)

    # Verificar columnas
    for col in MARKET_COLS:
        if col not in market.columns:
            logger.warning(f"  ⚠ Columna de mercado ausente: {col}")
    for col in MACRO_COLS:
        if col not in macro.columns:
            logger.warning(f"  ⚠ Columna macro ausente: {col}")

    logger.info(
        f"  Market: {market.shape[0]} meses × {market.shape[1]} cols | "
        f"{market.index.min().date()} → {market.index.max().date()}"
    )
    logger.info(
        f"  Macro:  {macro.shape[0]} meses × {macro.shape[1]} cols | "
        f"{macro.index.min().date()} → {macro.index.max().date()}"
    )

    return market, macro


# ══════════════════════════════════════════════════════════════════════════════
# 3. FUNCIONES AUXILIARES DE CÁLCULO
# ══════════════════════════════════════════════════════════════════════════════
# Estas funciones implementan las transformaciones atómicas usadas por los
# constructores de indicadores. Todas operan sobre pd.Series y devuelven
# pd.Series, preservando el índice temporal.
#
# IMPORTANTE sobre look-ahead bias:
# - Todas las funciones rolling usan min_periods=window para no producir
#   valores con datos insuficientes (evita estimaciones sesgadas al inicio).
# - Ninguna función usa ventanas centradas; todas son "trailing" (backward).
# - Los retornos se calculan con shift(n) que mira hacia atrás n periodos.

def pct_return(series: pd.Series, periods: int = 1) -> pd.Series:
    """
    Retorno porcentual sobre N periodos.

    Cálculo: (P_t / P_{t-n}) - 1

    Usa información pasada exclusivamente (shift mira hacia atrás).
    Los primeros N valores serán NaN por construcción.
    """
    return series.pct_change(periods=periods)


def log_return(series: pd.Series, periods: int = 1) -> pd.Series:
    """
    Retorno logarítmico sobre N periodos.

    Cálculo: ln(P_t / P_{t-n})

    Preferido en finanzas cuantitativas por su aditividad temporal
    y mejor comportamiento estadístico para retornos extremos.
    """
    return np.log(series / series.shift(periods))


def rolling_mean(series: pd.Series, window: int) -> pd.Series:
    """
    Media móvil trailing de N periodos.

    min_periods=window garantiza que no se calcula con menos datos de
    los requeridos (los primeros window-1 valores serán NaN).
    """
    return series.rolling(window=window, min_periods=window).mean()


def rolling_std(series: pd.Series, window: int) -> pd.Series:
    """
    Desviación estándar móvil trailing de N periodos.

    Mide la dispersión/volatilidad en una ventana retrospectiva.
    Usa ddof=1 (estimador insesgado de la desviación estándar muestral).
    """
    return series.rolling(window=window, min_periods=window).std(ddof=1)


def rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    """
    Z-score móvil: cuántas desviaciones estándar está el valor actual
    respecto a su media móvil.

    Cálculo: (X_t - mean_window) / std_window

    Útil para medir si un indicador está en niveles extremos respecto
    a su propio historial reciente. No es normalización global (eso
    introduciría look-ahead bias); es normalización rolling local.
    """
    mean = rolling_mean(series, window)
    std = rolling_std(series, window)
    return (series - mean) / std


def yoy_change(series: pd.Series) -> pd.Series:
    """
    Cambio year-over-year (12 meses).

    Para series que representan niveles (CPI, INDPRO), el YoY es
    el cambio porcentual: (X_t / X_{t-12}) - 1

    Los primeros 12 valores serán NaN.
    """
    return series.pct_change(periods=12)


def yoy_diff(series: pd.Series) -> pd.Series:
    """
    Diferencia year-over-year (12 meses) en nivel absoluto.

    Para series que ya están en unidades de tasa (FEDFUNDS, UNRATE),
    la diferencia absoluta es más interpretable que el cambio porcentual.

    Cálculo: X_t - X_{t-12}
    """
    return series - series.shift(12)


def drawdown_from_peak(series: pd.Series) -> pd.Series:
    """
    Drawdown actual respecto al máximo histórico acumulado.

    Cálculo: (P_t / max(P_0..P_t)) - 1

    Siempre ≤ 0. Un valor de -0.20 significa que el precio está un 20%
    por debajo de su máximo histórico hasta esa fecha.

    cummax() solo mira hacia atrás → sin look-ahead bias.
    """
    cumulative_max = series.cummax()
    return (series / cumulative_max) - 1


def relative_ratio(series_a: pd.Series, series_b: pd.Series) -> pd.Series:
    """
    Ratio entre dos series: A / B.

    Útil para medir valor relativo entre activos (ej: SPY/TLT).
    """
    return series_a / series_b


# ══════════════════════════════════════════════════════════════════════════════
# 4. ALMACÉN DE METADATOS
# ══════════════════════════════════════════════════════════════════════════════
# Cada indicador se registra en esta lista con sus metadatos completos.
# Esto permite generar automáticamente el archivo indicators_metadata.csv
# y facilita la documentación, filtrado y selección en fases posteriores.

_metadata_registry: list[dict] = []


def _register(
    name: str,
    category: str,
    description: str,
    source: str,
    natural_lag: str,
    limitations: str,
) -> None:
    """Registra los metadatos de un indicador en el almacén global."""
    _metadata_registry.append({
        "indicator": name,
        "category": category,
        "description": description,
        "source": source,
        "frequency": "monthly",
        "natural_lag": natural_lag,
        "limitations": limitations,
    })


# ══════════════════════════════════════════════════════════════════════════════
# 5. CONSTRUCTORES DE INDICADORES POR CATEGORÍA
# ══════════════════════════════════════════════════════════════════════════════
# Cada función recibe los DataFrames de mercado y/o macro y devuelve un
# DataFrame con las columnas de indicadores de su categoría.
#
# Convención de nombres:
#   {categoria}_{concepto}_{ventana/detalle}
#   Ejemplos: trend_momentum_6m, vol_realized_12m, credit_hy_oas_zscore_24m


# ──────────────────────────────────────────────────────────────────────────────
# 5.1  TENDENCIA DE MERCADO
# ──────────────────────────────────────────────────────────────────────────────
# Estos indicadores miden la dirección y fuerza del movimiento del mercado
# de renta variable en distintos horizontes temporales.
#
# Justificación económica:
# La tendencia de mercado refleja la evaluación agregada de los inversores
# sobre las perspectivas económicas. Los mercados en tendencia alcista
# sostenida típicamente coinciden con expansión económica y bajo estrés
# financiero. La persistencia del momentum está bien documentada en la
# literatura académica (Jegadeesh & Titman, 1993; Asness et al., 2013).
#
# Múltiples ventanas temporales capturan dinámicas diferentes:
# - Corto plazo (1-3m): reacción a eventos y sentimiento.
# - Medio plazo (6m): tendencia táctica.
# - Largo plazo (12m): tendencia cíclica/secular.
# ──────────────────────────────────────────────────────────────────────────────

def build_trend_indicators(market: pd.DataFrame) -> pd.DataFrame:
    """Construye indicadores de tendencia de mercado."""

    logger.info("  Categoría 1: Tendencia de mercado")
    df = pd.DataFrame(index=market.index)
    spy = market["SPY"]

    # --- Momentum (retorno acumulado) en distintas ventanas ---
    # Mide cuánto ha subido/bajado SPY en los últimos N meses.
    # Un momentum positivo indica tendencia alcista; negativo, bajista.
    for months in [1, 3, 6, 12]:
        col = f"trend_momentum_{months}m"
        df[col] = pct_return(spy, periods=months)
        _register(
            name=col,
            category="trend",
            description=f"Retorno acumulado del S&P 500 en los últimos {months} meses",
            source="SPY (yfinance)",
            natural_lag="0 (precio de cierre de fin de mes)",
            limitations="Basado en un solo ETF; no captura breadth del mercado",
        )

    # --- Precio relativo a media móvil ---
    # Mide si el mercado cotiza por encima o por debajo de su tendencia.
    # Ratio > 1 → por encima de la media (bullish); < 1 → por debajo (bearish).
    # Se usa ratio (no diferencia) para que sea comparable entre periodos
    # con distintos niveles de precio.
    for months in [6, 12]:
        col = f"trend_price_vs_ma_{months}m"
        ma = rolling_mean(spy, window=months)
        df[col] = spy / ma
        _register(
            name=col,
            category="trend",
            description=(
                f"Ratio del precio SPY respecto a su media móvil de {months} meses. "
                f">1 = por encima de tendencia"
            ),
            source="SPY (yfinance)",
            natural_lag="0",
            limitations="Media móvil simple; no pondera datos recientes más que antiguos",
        )

    # --- Drawdown desde máximo histórico ---
    # Mide cuánto ha caído el mercado desde su pico.
    # Un drawdown profundo indica estrés severo o mercado bajista.
    # Siempre ≤ 0; valor de 0 = en máximos históricos.
    col = "trend_drawdown"
    df[col] = drawdown_from_peak(spy)
    _register(
        name=col,
        category="trend",
        description="Drawdown del S&P 500 desde su máximo histórico acumulado",
        source="SPY (yfinance)",
        natural_lag="0",
        limitations="Asimétrico (solo mide caídas). Sensible a burbujas previas",
    )

    # --- Aceleración del momentum ---
    # Cambio en el momentum de 6 meses respecto al mes anterior.
    # Positivo = momentum acelerando; negativo = desacelerando.
    # Captura puntos de inflexión antes de que el momentum cambie de signo.
    col = "trend_momentum_accel"
    mom_6m = pct_return(spy, periods=6)
    df[col] = mom_6m - mom_6m.shift(1)
    _register(
        name=col,
        category="trend",
        description=(
            "Aceleración del momentum de 6 meses (cambio MoM del retorno 6m). "
            "Positivo = tendencia ganando fuerza"
        ),
        source="SPY (yfinance)",
        natural_lag="0",
        limitations="Derivada segunda; puede ser ruidosa en mercados laterales",
    )

    logger.info(f"    → {len(df.columns)} indicadores generados")
    return df


# ──────────────────────────────────────────────────────────────────────────────
# 5.2  VOLATILIDAD Y RIESGO
# ──────────────────────────────────────────────────────────────────────────────
# Estos indicadores miden la incertidumbre y el riesgo percibido en el
# mercado, tanto implícito (VIX) como realizado (vol histórica de SPY).
#
# Justificación económica:
# La volatilidad es un indicador fundamental del régimen de mercado.
# Regímenes de alta volatilidad coinciden con crisis, recesiones y
# drawdowns severos. La distinción entre volatilidad implícita (forward-
# looking, basada en opciones) y realizada (backward-looking) proporciona
# información sobre las expectativas del mercado vs. la realidad reciente.
# El "spread" entre ambas (variance risk premium) es un indicador de
# aversión al riesgo documentado en la literatura (Bollerslev et al., 2009).
# ──────────────────────────────────────────────────────────────────────────────

def build_volatility_indicators(market: pd.DataFrame) -> pd.DataFrame:
    """Construye indicadores de volatilidad y riesgo."""

    logger.info("  Categoría 2: Volatilidad y riesgo")
    df = pd.DataFrame(index=market.index)
    spy = market["SPY"]
    vix = market["VIX"]

    # --- VIX: nivel y dinámica ---
    # El VIX ya es un indicador de volatilidad implícita (media mensual
    # del preprocesado). Incluimos el nivel, su cambio y su z-score.

    # Nivel del VIX
    col = "vol_vix_level"
    df[col] = vix
    _register(
        name=col,
        category="volatility",
        description="Nivel del VIX (media mensual de volatilidad implícita a 30 días)",
        source="VIX (yfinance)",
        natural_lag="0 (media del mes completo)",
        limitations="Derivado de opciones del S&P 500; puede distorsionarse por flujos",
    )

    # Cambio mensual del VIX
    col = "vol_vix_mom_change"
    df[col] = vix - vix.shift(1)
    _register(
        name=col,
        category="volatility",
        description="Cambio absoluto mensual del VIX (nivel actual - nivel mes anterior)",
        source="VIX (yfinance)",
        natural_lag="0",
        limitations="Cambio absoluto; la magnitud depende del nivel base del VIX",
    )

    # Z-score del VIX en ventana de 24 meses
    # Mide si el VIX está en niveles extremos respecto a su historial
    # reciente de 2 años. Un z-score > 2 sugiere estrés anómalo.
    col = "vol_vix_zscore_24m"
    df[col] = rolling_zscore(vix, window=24)
    _register(
        name=col,
        category="volatility",
        description=(
            "Z-score del VIX en ventana de 24 meses. "
            "Mide si la volatilidad implícita está en niveles extremos"
        ),
        source="VIX (yfinance)",
        natural_lag="0",
        limitations="Sensible a periodos anómalos dentro de la ventana (ej: COVID en 2020)",
    )

    # --- Volatilidad realizada del S&P 500 ---
    # Desviación estándar de los retornos mensuales de SPY en ventanas
    # de 3, 6 y 12 meses. Mide la volatilidad histórica efectiva.
    spy_ret = log_return(spy, periods=1)

    for months in [3, 6, 12]:
        col = f"vol_realized_{months}m"
        df[col] = rolling_std(spy_ret, window=months)
        _register(
            name=col,
            category="volatility",
            description=(
                f"Volatilidad realizada del S&P 500: desviación estándar de "
                f"retornos log mensuales en ventana de {months} meses"
            ),
            source="SPY (yfinance)",
            natural_lag="0",
            limitations=(
                f"Backward-looking (usa {months} meses pasados). "
                "No captura volatilidad intra-mes (datos mensuales)"
            ),
        )

    # --- Spread implícita vs realizada (Variance Risk Premium proxy) ---
    # VIX - volatilidad realizada anualizada.
    # VIX está en unidades anualizadas (%), así que anualizamos la vol
    # realizada de 6m multiplicando por sqrt(12).
    # Un spread alto indica que el mercado espera más riesgo del
    # que ha ocurrido recientemente → mayor aversión al riesgo.
    col = "vol_implied_vs_realized_6m"
    realized_annualized = rolling_std(spy_ret, window=6) * np.sqrt(12) * 100
    df[col] = vix - realized_annualized
    _register(
        name=col,
        category="volatility",
        description=(
            "Spread entre volatilidad implícita (VIX) y realizada anualizada (6m). "
            "Proxy del variance risk premium. Positivo = mercado espera más riesgo"
        ),
        source="VIX + SPY (yfinance)",
        natural_lag="0",
        limitations=(
            "VIX es media mensual, vol realizada es de datos mensuales. "
            "Comparación aproximada por diferencia de granularidad"
        ),
    )

    logger.info(f"    → {len(df.columns)} indicadores generados")
    return df


# ──────────────────────────────────────────────────────────────────────────────
# 5.3  VALORACIÓN RELATIVA
# ──────────────────────────────────────────────────────────────────────────────
# Estos indicadores miden el posicionamiento relativo del mercado de renta
# variable frente a otras clases de activos y referentes de tipo de interés.
#
# Justificación económica:
# La valoración absoluta del mercado (P/E, CAPE, etc.) requiere datos de
# beneficios empresariales que no están disponibles en este proyecto.
# Sin embargo, el valor RELATIVO entre activos proporciona información
# importante sobre las preferencias de riesgo de los inversores y el
# atractivo comparativo de la renta variable. El ratio equity/bonds
# (SPY/TLT) refleja la asignación implícita del mercado entre crecimiento
# y seguridad. El ratio equity/oro (SPY/GLD) mide la preferencia por
# activos reales vs. financieros.
#
# LIMITACIÓN PRINCIPAL: No disponemos de P/E, CAPE, ni earnings yield.
# Los indicadores aquí son de valor relativo cross-asset, no de
# valoración fundamental del S&P 500.
# ──────────────────────────────────────────────────────────────────────────────

def build_valuation_indicators(
    market: pd.DataFrame, macro: pd.DataFrame,
) -> pd.DataFrame:
    """Construye indicadores de valoración relativa."""

    logger.info("  Categoría 3: Valoración relativa")
    df = pd.DataFrame(index=market.index)
    spy = market["SPY"]

    # --- Ratio equity/bonds (SPY/TLT) ---
    # Un ratio ascendente indica que la renta variable se abarata
    # relativamente a los bonos (o que los inversores prefieren riesgo).
    # Z-score para contextualizar respecto al historial reciente.
    if "TLT" in market.columns:
        ratio_eq_bond = relative_ratio(spy, market["TLT"])

        col = "val_equity_bond_ratio"
        df[col] = ratio_eq_bond
        _register(
            name=col,
            category="valuation",
            description="Ratio SPY/TLT — valor relativo de RV vs bonos largo plazo",
            source="SPY + TLT (yfinance)",
            natural_lag="0",
            limitations="No mide valoración absoluta; depende de oferta/demanda de ambos",
        )

        col = "val_equity_bond_zscore_24m"
        df[col] = rolling_zscore(ratio_eq_bond, window=24)
        _register(
            name=col,
            category="valuation",
            description=(
                "Z-score 24m del ratio SPY/TLT. "
                "Valores extremos sugieren posicionamiento cross-asset inusual"
            ),
            source="SPY + TLT (yfinance)",
            natural_lag="0",
            limitations="Ventana de 24m puede no capturar cambios de régimen",
        )

    # --- Ratio equity/oro (SPY/GLD) ---
    # El oro compite con la renta variable como reserva de valor.
    # Un ratio descendente indica preferencia por activos refugio.
    if "GLD" in market.columns:
        ratio_eq_gold = relative_ratio(spy, market["GLD"])

        col = "val_equity_gold_ratio"
        df[col] = ratio_eq_gold
        _register(
            name=col,
            category="valuation",
            description="Ratio SPY/GLD — valor relativo de RV vs oro",
            source="SPY + GLD (yfinance)",
            natural_lag="0",
            limitations="GLD disponible solo desde 2004; serie corta",
        )

        col = "val_equity_gold_momentum_12m"
        df[col] = pct_return(ratio_eq_gold, periods=12)
        _register(
            name=col,
            category="valuation",
            description=(
                "Momentum 12m del ratio SPY/GLD. "
                "Positivo = RV ganando terreno vs oro en el último año"
            ),
            source="SPY + GLD (yfinance)",
            natural_lag="0",
            limitations="Sensible a shocks específicos del oro (geopolítica, USD)",
        )

    # --- Tipo real a largo plazo: GS10 - Breakeven Inflation ---
    # Proxy del rendimiento real del bono a 10 años.
    # Tipos reales altos encarecen el capital y reducen el atractivo
    # relativo de la renta variable. Tipos reales negativos favorecen
    # activos de riesgo.
    if "GS10" in macro.columns and "T10YIE" in macro.columns:
        col = "val_real_yield_10y"
        df[col] = macro["GS10"] - macro["T10YIE"]
        _register(
            name=col,
            category="valuation",
            description=(
                "Tipo de interés real a 10 años: GS10 - Breakeven Inflation 10Y. "
                "Proxy del coste real del capital a largo plazo"
            ),
            source="GS10 + T10YIE (FRED)",
            natural_lag="0 (datos de mercado)",
            limitations="T10YIE solo disponible desde 2003",
        )

    # --- Excess yield: rendimiento del bono vs retorno reciente de RV ---
    # Compara el rendimiento nominal del bono 10Y con el retorno
    # anualizado de SPY en los últimos 12 meses. No es un "equity risk
    # premium" formal (necesitaría earnings yield), sino un proxy crudo
    # de atractivo relativo basado en retornos recientes.
    if "GS10" in macro.columns:
        spy_ret_12m_annualized = pct_return(spy, periods=12) * 100  # en %
        col = "val_bond_yield_vs_spy_ret"
        df[col] = macro["GS10"] - spy_ret_12m_annualized
        _register(
            name=col,
            category="valuation",
            description=(
                "GS10 - retorno 12m de SPY (%). "
                "Positivo = bonos ofrecen más que lo que RV ha dado recientemente"
            ),
            source="GS10 (FRED) + SPY (yfinance)",
            natural_lag="0",
            limitations=(
                "Comparación asimétrica: yield (forward) vs retorno (backward). "
                "No es equity risk premium formal"
            ),
        )

    logger.info(f"    → {len(df.columns)} indicadores generados")
    return df


# ──────────────────────────────────────────────────────────────────────────────
# 5.4  CICLO ECONÓMICO
# ──────────────────────────────────────────────────────────────────────────────
# Estos indicadores miden el estado y la dirección del ciclo de actividad
# económica real.
#
# Justificación económica:
# El ciclo económico es el determinante fundamental de los beneficios
# empresariales y, por extensión, de los retornos del mercado a medio y
# largo plazo. La producción industrial y el empleo son los dos pilares
# del NBER Business Cycle Dating Committee para definir recesiones.
# Incluimos la variable USREC como referencia de validación, pero con
# la advertencia explícita de que NO puede usarse como señal prospectiva
# debido a su retraso de publicación.
# ──────────────────────────────────────────────────────────────────────────────

def build_cycle_indicators(macro: pd.DataFrame) -> pd.DataFrame:
    """Construye indicadores de ciclo económico."""

    logger.info("  Categoría 4: Ciclo económico")
    df = pd.DataFrame(index=macro.index)

    # --- Producción industrial: crecimiento year-over-year ---
    # Medida central de la actividad económica real.
    # YoY suaviza estacionalidad. Valores negativos = contracción.
    if "INDPRO" in macro.columns:
        indpro = macro["INDPRO"]

        col = "cycle_indpro_yoy"
        df[col] = yoy_change(indpro)
        _register(
            name=col,
            category="cycle",
            description=(
                "Crecimiento YoY de la producción industrial. "
                "Negativo = contracción del sector industrial"
            ),
            source="INDPRO (FRED)",
            natural_lag="~1-2 meses (publicación con retraso + revisiones)",
            limitations="Solo sector industrial/manufacturero; no captura servicios",
        )

        # Momentum de 3 y 6 meses de INDPRO
        for months in [3, 6]:
            col = f"cycle_indpro_mom_{months}m"
            df[col] = pct_return(indpro, periods=months)
            _register(
                name=col,
                category="cycle",
                description=f"Retorno de producción industrial en {months} meses",
                source="INDPRO (FRED)",
                natural_lag="~1-2 meses",
                limitations="Revisiones retroactivas del dato",
            )

        # Aceleración: cambio en el YoY
        # Positivo = la economía no solo crece, sino que se acelera.
        col = "cycle_indpro_accel"
        indpro_yoy = yoy_change(indpro)
        df[col] = indpro_yoy - indpro_yoy.shift(3)
        _register(
            name=col,
            category="cycle",
            description=(
                "Aceleración de la producción industrial: cambio en YoY vs 3 meses atrás. "
                "Positivo = actividad acelerando"
            ),
            source="INDPRO (FRED)",
            natural_lag="~1-2 meses",
            limitations="Derivada segunda; amplifica ruido",
        )

    # --- Desempleo: nivel, dirección y dinámica ---
    if "UNRATE" in macro.columns:
        unrate = macro["UNRATE"]

        # Nivel de desempleo
        col = "cycle_unemployment_level"
        df[col] = unrate
        _register(
            name=col,
            category="cycle",
            description="Tasa de desempleo U-3 (nivel)",
            source="UNRATE (FRED)",
            natural_lag="~1 mes (publicación primer viernes del mes siguiente)",
            limitations="Indicador retrasado del ciclo; sube después de que empieza la recesión",
        )

        # Cambio YoY del desempleo (en puntos porcentuales)
        col = "cycle_unemployment_yoy_diff"
        df[col] = yoy_diff(unrate)
        _register(
            name=col,
            category="cycle",
            description=(
                "Cambio YoY del desempleo en puntos porcentuales. "
                "Positivo = desempleo subiendo vs hace un año"
            ),
            source="UNRATE (FRED)",
            natural_lag="~1 mes",
            limitations="Retrasado; cuando sube mucho la recesión ya está en curso",
        )

        # Cambio de 3 meses (dirección reciente)
        col = "cycle_unemployment_3m_diff"
        df[col] = unrate - unrate.shift(3)
        _register(
            name=col,
            category="cycle",
            description=(
                "Cambio del desempleo en 3 meses (pp). "
                "Captura deterioro rápido del mercado laboral"
            ),
            source="UNRATE (FRED)",
            natural_lag="~1 mes",
            limitations="Puede ser volátil por cambios estacionales",
        )

    # --- USREC: indicador de recesión NBER (solo para validación) ---
    if "USREC" in macro.columns:
        col = "cycle_nber_recession"
        df[col] = macro["USREC"]
        _register(
            name=col,
            category="cycle",
            description=(
                "Indicador binario de recesión NBER (1=recesión, 0=expansión). "
                "⚠ SOLO PARA VALIDACIÓN EX-POST"
            ),
            source="USREC (FRED)",
            natural_lag="6-18 MESES (declaración con retraso extremo del NBER)",
            limitations=(
                "NO usar como señal prospectiva. El NBER declara recesiones "
                "meses o años después de que empiezan. Solo válido para backtesting"
            ),
        )

    logger.info(f"    → {len(df.columns)} indicadores generados")
    return df


# ──────────────────────────────────────────────────────────────────────────────
# 5.5  POLÍTICA MONETARIA
# ──────────────────────────────────────────────────────────────────────────────
# Estos indicadores miden el estado y la dirección de la política monetaria
# de la Reserva Federal, así como la curva de tipos de interés.
#
# Justificación económica:
# La política monetaria es el principal mecanismo de control del ciclo
# económico y tiene impacto directo en la valoración de activos (vía
# tipos de descuento) y en las condiciones financieras. La curva de tipos
# es uno de los predictores de recesión más documentados en la literatura
# (Estrella & Mishkin, 1996; Adrian et al., 2019). El tipo real de la Fed
# (ajustado por inflación) determina si la política es realmente
# restrictiva o expansiva.
# ──────────────────────────────────────────────────────────────────────────────

def build_monetary_indicators(macro: pd.DataFrame) -> pd.DataFrame:
    """Construye indicadores de política monetaria."""

    logger.info("  Categoría 5: Política monetaria")
    df = pd.DataFrame(index=macro.index)

    # --- Fed Funds Rate: nivel y dirección ---
    if "FEDFUNDS" in macro.columns:
        ff = macro["FEDFUNDS"]

        col = "mon_fedfunds_level"
        df[col] = ff
        _register(
            name=col,
            category="monetary",
            description="Tipo de interés de los Fed Funds (nivel mensual)",
            source="FEDFUNDS (FRED)",
            natural_lag="0 (dato oficial del periodo)",
            limitations="Media mensual; no captura cambios intra-mes por reuniones FOMC",
        )

        # Cambio en 6 y 12 meses (dirección de la política)
        for months in [6, 12]:
            col = f"mon_fedfunds_diff_{months}m"
            df[col] = ff - ff.shift(months)
            _register(
                name=col,
                category="monetary",
                description=(
                    f"Cambio en Fed Funds en {months} meses (pp). "
                    f"Positivo = endurecimiento; negativo = relajación"
                ),
                source="FEDFUNDS (FRED)",
                natural_lag="0",
                limitations="No captura expectativas futuras de política (forward guidance)",
            )

    # --- Tipo real de la Fed: FEDFUNDS - CPI YoY ---
    # El tipo real determina si la política es genuinamente restrictiva
    # (tipo real positivo) o acomodaticia (tipo real negativo).
    # CPI YoY como proxy de inflación actual.
    if "FEDFUNDS" in macro.columns and "CPI" in macro.columns:
        cpi_yoy = yoy_change(macro["CPI"]) * 100  # En porcentaje
        col = "mon_real_rate"
        df[col] = macro["FEDFUNDS"] - cpi_yoy
        _register(
            name=col,
            category="monetary",
            description=(
                "Tipo de interés real de la Fed: FEDFUNDS - CPI YoY (%). "
                "Positivo = política restrictiva; negativo = acomodaticia"
            ),
            source="FEDFUNDS + CPI (FRED)",
            natural_lag="~1 mes (por publicación del CPI)",
            limitations=(
                "CPI se publica con retraso y se revisa. "
                "El tipo real ex-ante (con expectativas) sería más preciso"
            ),
        )

    # --- Curva de tipos: T10Y2Y ---
    # El spread entre el bono a 10 años y el de 2 años es el indicador
    # clásico de la curva. Una inversión (valores negativos) precede
    # históricamente a recesiones con 6-18 meses de antelación.
    if "T10Y2Y" in macro.columns:
        curve = macro["T10Y2Y"]

        col = "mon_yield_curve_level"
        df[col] = curve
        _register(
            name=col,
            category="monetary",
            description=(
                "Spread de la curva de tipos 10Y-2Y. "
                "Negativo = curva invertida (señal clásica pre-recesión)"
            ),
            source="T10Y2Y (FRED)",
            natural_lag="0 (dato de mercado)",
            limitations="La inversión puede durar poco o mucho antes de la recesión",
        )

        # Dirección de la curva (cambio en 3 y 6 meses)
        for months in [3, 6]:
            col = f"mon_yield_curve_diff_{months}m"
            df[col] = curve - curve.shift(months)
            _register(
                name=col,
                category="monetary",
                description=(
                    f"Cambio en el spread 10Y-2Y en {months} meses (pp). "
                    f"Negativo = curva aplanándose/invirtiéndose"
                ),
                source="T10Y2Y (FRED)",
                natural_lag="0",
                limitations="Cambio de pendiente, no nivel; ambos son informativos",
            )

    # --- Nivel del bono a 10 años ---
    if "GS10" in macro.columns:
        col = "mon_gs10_level"
        df[col] = macro["GS10"]
        _register(
            name=col,
            category="monetary",
            description="Rendimiento del Treasury 10Y — referencia de largo plazo",
            source="GS10 (FRED)",
            natural_lag="0",
            limitations="Influido por factores globales (no solo política de la Fed)",
        )

    logger.info(f"    → {len(df.columns)} indicadores generados")
    return df


# ──────────────────────────────────────────────────────────────────────────────
# 5.6  ESTRÉS FINANCIERO Y CRÉDITO
# ──────────────────────────────────────────────────────────────────────────────
# Estos indicadores miden las condiciones del mercado de crédito y el
# nivel de estrés financiero sistémico.
#
# Justificación económica:
# Los spreads de crédito reflejan la percepción del riesgo de impago
# del sector corporativo. Un ensanchamiento de spreads indica
# endurecimiento de las condiciones financieras y suele preceder a
# desaceleraciones económicas. El spread HY es especialmente sensible
# porque la deuda de menor calidad es la primera en sufrir cuando las
# condiciones se deterioran. La relación entre HYG y TLT (risk-on vs
# risk-off) captura los flujos de rotación entre activos de riesgo
# y activos refugio.
# ──────────────────────────────────────────────────────────────────────────────

def build_credit_indicators(
    market: pd.DataFrame, macro: pd.DataFrame,
) -> pd.DataFrame:
    """Construye indicadores de estrés financiero y crédito."""

    logger.info("  Categoría 6: Estrés financiero y crédito")
    df = pd.DataFrame(index=market.index)

    # --- HY OAS: nivel, dirección y extremos ---
    if "HY_OAS" in macro.columns:
        hy_oas = macro["HY_OAS"]

        col = "credit_hy_oas_level"
        df[col] = hy_oas
        _register(
            name=col,
            category="credit",
            description=(
                "Spread de crédito HY option-adjusted (bps sobre Treasuries). "
                "Mayor = más estrés crediticio"
            ),
            source="BAMLH0A0HYM2 (FRED / ICE BofA)",
            natural_lag="0 (dato de mercado)",
            limitations="Disponible desde 1996; composición del índice cambia con el tiempo",
        )

        # Cambio en 3 meses (dirección reciente del estrés)
        col = "credit_hy_oas_3m_change"
        df[col] = hy_oas - hy_oas.shift(3)
        _register(
            name=col,
            category="credit",
            description=(
                "Cambio en HY OAS en 3 meses. "
                "Positivo = estrés crediticio en aumento"
            ),
            source="BAMLH0A0HYM2 (FRED)",
            natural_lag="0",
            limitations="Spread puede ser volátil por factores de liquidez",
        )

        # Z-score 24m del HY OAS
        col = "credit_hy_oas_zscore_24m"
        df[col] = rolling_zscore(hy_oas, window=24)
        _register(
            name=col,
            category="credit",
            description=(
                "Z-score 24m del spread HY. "
                "Valores altos = estrés crediticio inusualmente elevado"
            ),
            source="BAMLH0A0HYM2 (FRED)",
            natural_lag="0",       
            limitations="Sensible a periodos de crisis dentro de la ventana",
        )

    # --- Ratio HYG/LQD: calidad del crédito ---
    # Un ratio descendente indica que la deuda HY cae más que la IG,
    # señal de flight-to-quality dentro del mercado de crédito.
    if "HYG" in market.columns and "LQD" in market.columns:
        ratio_hy_ig = relative_ratio(market["HYG"], market["LQD"])

        col = "credit_hy_ig_ratio"
        df[col] = ratio_hy_ig
        _register(
            name=col,
            category="credit",
            description="Ratio HYG/LQD — calidad del crédito (HY relativo a IG)",
            source="HYG + LQD (yfinance)",
            natural_lag="0",
            limitations="HYG disponible desde 2007; no cubre crisis pre-2007",
        )

        col = "credit_hy_ig_momentum_6m"
        df[col] = pct_return(ratio_hy_ig, periods=6)
        _register(
            name=col,
            category="credit",
            description=(
                "Momentum 6m del ratio HYG/LQD. "
                "Negativo = flight-to-quality en crédito"
            ),
            source="HYG + LQD (yfinance)",
            natural_lag="0",
            limitations="Serie corta (desde 2007)",
        )

    # --- Ratio risk-on/risk-off: HYG/TLT ---
    # Captura la rotación entre activos de riesgo (HY bonds) y
    # activos refugio (Treasuries largo plazo).
    if "HYG" in market.columns and "TLT" in market.columns:
        ratio_risk = relative_ratio(market["HYG"], market["TLT"])

        col = "credit_riskon_riskoff_ratio"
        df[col] = ratio_risk
        _register(
            name=col,
            category="credit",
            description=(
                "Ratio HYG/TLT — risk-on vs risk-off. "
                "Descendente = flight to quality"
            ),
            source="HYG + TLT (yfinance)",
            natural_lag="0",
            limitations="Ambos ETFs influidos por tipos de interés, no solo por riesgo",
        )

        col = "credit_riskon_riskoff_mom_6m"
        df[col] = pct_return(ratio_risk, periods=6)
        _register(
            name=col,
            category="credit",
            description="Momentum 6m del ratio risk-on/off (HYG/TLT)",
            source="HYG + TLT (yfinance)",
            natural_lag="0",
            limitations="Serie corta; sesgada por tipo de interés a largo plazo",
        )

    logger.info(f"    → {len(df.columns)} indicadores generados")
    return df


# ──────────────────────────────────────────────────────────────────────────────
# 5.7  INFLACIÓN Y EXPECTATIVAS
# ──────────────────────────────────────────────────────────────────────────────
# Estos indicadores miden el nivel, la dirección y las expectativas de
# inflación, tanto realizada (CPI) como implícita en el mercado (breakevens).
#
# Justificación económica:
# La inflación determina el poder adquisitivo de los retornos, la
# reacción de la Fed (y por tanto los tipos), y el descuento de flujos
# futuros. La inflación elevada y acelerada suele ser negativa para
# la renta variable (Fed restrictiva, compresión de múltiplos). La
# desinflación o inflación moderada suele acompañar mercados alcistas.
# Las expectativas de inflación del mercado (breakevens) son forward-
# looking y complementan la inflación realizada (backward-looking).
# ──────────────────────────────────────────────────────────────────────────────

def build_inflation_indicators(macro: pd.DataFrame) -> pd.DataFrame:
    """Construye indicadores de inflación y expectativas."""

    logger.info("  Categoría 7: Inflación y expectativas")
    df = pd.DataFrame(index=macro.index)

    # --- Inflación realizada (CPI) ---
    if "CPI" in macro.columns:
        cpi = macro["CPI"]

        # CPI Year-over-Year
        # El indicador más estándar y comparable de inflación.
        col = "infl_cpi_yoy"
        df[col] = yoy_change(cpi)
        _register(
            name=col,
            category="inflation",
            description="Inflación CPI year-over-year (tasa anualizada)",
            source="CPIAUCSL (FRED)",
            natural_lag="~2 semanas (publicación mediados del mes siguiente)",
            limitations="Dato revisable; no desestacionalizado en origen (CPIAUCSL sí lo es)",
        )

        # CPI Month-over-Month (tasa mensual, no anualizada)
        col = "infl_cpi_mom"
        df[col] = pct_return(cpi, periods=1)
        _register(
            name=col,
            category="inflation",
            description="Inflación CPI mes a mes (tasa mensual)",
            source="CPIAUCSL (FRED)",
            natural_lag="~2 semanas",
            limitations="Puede ser volátil por componentes estacionales (energía, alimentos)",
        )

        # Aceleración de la inflación: cambio en CPI YoY vs 6 meses atrás
        # Positivo = inflación acelerando; negativo = desacelerando.
        # Este indicador es especialmente relevante porque la Fed reacciona
        # más a la DIRECCIÓN de la inflación que a su nivel absoluto.
        col = "infl_cpi_accel_6m"
        cpi_yoy = yoy_change(cpi)
        df[col] = cpi_yoy - cpi_yoy.shift(6)
        _register(
            name=col,
            category="inflation",
            description=(
                "Aceleración de la inflación: CPI YoY actual vs 6 meses atrás. "
                "Positivo = inflación acelerando"
            ),
            source="CPIAUCSL (FRED)",
            natural_lag="~2 semanas",
            limitations="Derivada segunda; ruidosa en periodos de transición",
        )

        # CPI tendencia de medio plazo: media móvil 6m del MoM
        col = "infl_cpi_trend_6m"
        cpi_mom = pct_return(cpi, periods=1)
        df[col] = rolling_mean(cpi_mom, window=6)
        _register(
            name=col,
            category="inflation",
            description=(
                "Tendencia inflacionaria de 6 meses: media móvil del CPI MoM. "
                "Suaviza ruido mensual"
            ),
            source="CPIAUCSL (FRED)",
            natural_lag="~2 semanas",
            limitations="Suavizado introduce lag adicional en la señal",
        )

    # --- Expectativas de inflación del mercado (Breakeven) ---
    if "T10YIE" in macro.columns:
        bie = macro["T10YIE"]

        col = "infl_breakeven_10y"
        df[col] = bie
        _register(
            name=col,
            category="inflation",
            description=(
                "Breakeven inflation 10Y — expectativa de inflación del mercado "
                "implícita en el spread entre bonos nominales y TIPS"
            ),
            source="T10YIE (FRED)",
            natural_lag="0 (dato de mercado)",
            limitations="Contaminada por prima de liquidez de los TIPS; desde 2003",
        )

        # Cambio en 3 meses (dirección de las expectativas)
        col = "infl_breakeven_3m_change"
        df[col] = bie - bie.shift(3)
        _register(
            name=col,
            category="inflation",
            description=(
                "Cambio en breakeven inflation 10Y en 3 meses. "
                "Positivo = expectativas de inflación al alza"
            ),
            source="T10YIE (FRED)",
            natural_lag="0",
            limitations="Prima de liquidez puede distorsionar cambios",
        )

    # --- Sorpresa inflacionaria: CPI YoY vs Breakeven ---
    # Si la inflación realizada supera las expectativas, indica
    # sorpresa inflacionaria (potencialmente negativa para activos).
    if "CPI" in macro.columns and "T10YIE" in macro.columns:
        col = "infl_surprise"
        cpi_yoy_pct = yoy_change(macro["CPI"]) * 100  # En %
        df[col] = cpi_yoy_pct - macro["T10YIE"]
        _register(
            name=col,
            category="inflation",
            description=(
                "Sorpresa inflacionaria: CPI YoY (%) - Breakeven 10Y (%). "
                "Positivo = inflación realizada supera expectativas"
            ),
            source="CPI + T10YIE (FRED)",
            natural_lag="~2 semanas (por CPI)",
            limitations=(
                "Comparación imperfecta: CPI es backward (últimos 12m), "
                "breakeven es forward (próximos 10Y)"
            ),
        )

    logger.info(f"    → {len(df.columns)} indicadores generados")
    return df


# ──────────────────────────────────────────────────────────────────────────────
# 5.8  AMPLITUD / PARTICIPACIÓN CROSS-ASSET
# ──────────────────────────────────────────────────────────────────────────────
# Con solo el SPY como representante de renta variable, no es posible
# calcular indicadores de amplitud intra-equity (advance/decline, % por
# encima de MA, new highs/lows, etc.). Eso requeriría datos de componentes
# individuales del índice.
#
# Sin embargo, SÍ podemos medir la "amplitud cross-asset": cuántas clases
# de activos distintas participan en una tendencia. Esto es análogo a la
# amplitud de mercado pero entre asset classes.
#
# Justificación económica:
# Cuando múltiples clases de activos de riesgo suben simultáneamente
# (RV, crédito, oro), la tendencia alcista tiene mayor "profundidad"
# y suele ser más sostenible. Cuando solo sube un activo y el resto
# cae, la dispersión aumenta y la tendencia es más frágil. Este concepto
# se relaciona con la "risk-on breadth" y los flujos institucionales
# cross-asset.
#
# LIMITACIÓN: Estos indicadores son proxies cross-asset, NO amplitud
# intra-equity. No sustituyen a indicadores como el advance-decline line.
# ──────────────────────────────────────────────────────────────────────────────

def build_breadth_indicators(market: pd.DataFrame) -> pd.DataFrame:
    """Construye indicadores de amplitud/participación cross-asset."""

    logger.info("  Categoría 8: Amplitud cross-asset")
    df = pd.DataFrame(index=market.index)

    # Activos disponibles para medir amplitud cross-asset
    risk_assets = ["SPY", "TLT", "TIP", "LQD", "HYG", "GLD"]
    available = [a for a in risk_assets if a in market.columns]

    if len(available) < 3:
        logger.warning("  ⚠ Insuficientes activos para amplitud cross-asset")
        return df

    # Calcular retornos mensuales de todos los activos disponibles
    returns = pd.DataFrame({
        asset: pct_return(market[asset], periods=1) for asset in available
    })

    # --- Número de activos con retorno positivo (1 mes) ---
    # Cuenta cuántos activos cerraron el mes en positivo.
    # Valores altos = rally generalizado; valores bajos = estrés amplio.
    col = "breadth_positive_assets_1m"
    df[col] = (returns > 0).sum(axis=1)
    _register(
        name=col,
        category="breadth",
        description=(
            f"Nº de activos ({', '.join(available)}) con retorno mensual positivo. "
            f"Rango [0, {len(available)}]"
        ),
        source="yfinance (múltiples ETFs)",
        natural_lag="0",
        limitations="Solo cross-asset; no mide amplitud intra-equity",
    )

    # --- Fracción de activos con momentum positivo (6m) ---
    # Versión a medio plazo: cuenta activos con retorno 6m > 0.
    returns_6m = pd.DataFrame({
        asset: pct_return(market[asset], periods=6) for asset in available
    })

    col = "breadth_positive_mom6m_fraction"
    df[col] = (returns_6m > 0).sum(axis=1) / len(available)
    _register(
        name=col,
        category="breadth",
        description=(
            "Fracción de activos cross-asset con momentum 6m positivo. "
            "1.0 = todos en tendencia alcista; 0.0 = todos en caída"
        ),
        source="yfinance (múltiples ETFs)",
        natural_lag="0",
        limitations="Solo 6-7 activos; muestra pequeña para amplitud robusta",
    )

    # --- Dispersión de retornos cross-asset (1m) ---
    # Desviación estándar de los retornos mensuales entre activos.
    # Alta dispersión = baja correlación entre activos, régimen mixto.
    # Baja dispersión con retornos altos = rally coordinado.
    col = "breadth_return_dispersion_1m"
    df[col] = returns.std(axis=1, ddof=1)
    _register(
        name=col,
        category="breadth",
        description=(
            "Dispersión (std) de retornos mensuales entre activos cross-asset. "
            "Alta = régimen divergente; baja = movimiento coordinado"
        ),
        source="yfinance (múltiples ETFs)",
        natural_lag="0",
        limitations="Sensible a la composición del universo de activos",
    )

    # --- Correlación media rolling entre activos (12m) ---
    # Mide cuán correlacionados están los activos entre sí.
    # Correlaciones altas suelen indicar risk-on/off extremo.
    col = "breadth_avg_corr_12m"
    rolling_corrs = returns.rolling(window=12, min_periods=12).corr()

    # Extraer correlación media de la parte triangular superior
    # (excluye diagonal y pares duplicados).
    n_assets = len(available)
    mean_corrs = []
    for date in returns.index:
        try:
            corr_matrix = rolling_corrs.loc[date]
            if isinstance(corr_matrix, pd.DataFrame) and corr_matrix.shape == (n_assets, n_assets):
                # Tomar la parte triangular superior sin diagonal
                upper_triangle = corr_matrix.values[np.triu_indices(n_assets, k=1)]
                mean_corrs.append(np.nanmean(upper_triangle))
            else:
                mean_corrs.append(np.nan)
        except (KeyError, ValueError):
            mean_corrs.append(np.nan)

    df[col] = pd.Series(mean_corrs, index=returns.index)
    _register(
        name=col,
        category="breadth",
        description=(
            "Correlación media rolling 12m entre activos cross-asset. "
            "Alta = movimientos muy coordinados (risk-on/off extremo)"
        ),
        source="yfinance (múltiples ETFs)",
        natural_lag="0",
        limitations=(
            "Solo 6-7 activos; la correlación rolling con 12 meses es "
            "estimación ruidosa. Asume estabilidad dentro de la ventana"
        ),
    )

    # --- Retorno medio cross-asset (proxy de risk appetite) ---
    # Retorno medio de todos los activos en los últimos 3 meses.
    # Un retorno medio positivo alto = apetito de riesgo generalizado.
    returns_3m = pd.DataFrame({
        asset: pct_return(market[asset], periods=3) for asset in available
    })

    col = "breadth_avg_return_3m"
    df[col] = returns_3m.mean(axis=1)
    _register(
        name=col,
        category="breadth",
        description=(
            "Retorno medio 3m cross-asset. "
            "Proxy de apetito de riesgo global del mercado"
        ),
        source="yfinance (múltiples ETFs)",
        natural_lag="0",
        limitations="Incluye activos con diferente perfil de riesgo (TLT vs SPY)",
    )

    logger.info(f"    → {len(df.columns)} indicadores generados")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 6. ORQUESTADOR DE INDICADORES
# ══════════════════════════════════════════════════════════════════════════════

def build_all_indicators(
    market: pd.DataFrame,
    macro: pd.DataFrame,
) -> pd.DataFrame:
    """
    Construye el universo completo de indicadores.

    Ejecuta secuencialmente los 8 constructores de categorías y
    combina todos los indicadores en un único DataFrame.

    Los indicadores se calculan sobre el eje temporal común que resulta
    del outer join de mercado y macro. Cada indicador tendrá NaNs al
    inicio según su ventana de cálculo y la disponibilidad de sus datos
    fuente — esto es esperado y se documenta en los metadatos.

    Parámetros
    ----------
    market : pd.DataFrame
        Datos de mercado mensuales (del preprocessing.py).
    macro : pd.DataFrame
        Datos macroeconómicos mensuales (del preprocessing.py).

    Retorna
    -------
    pd.DataFrame : todos los indicadores, una columna por indicador.
    """
    logger.info("=" * 70)
    logger.info("CONSTRUCCIÓN DEL UNIVERSO DE INDICADORES")
    logger.info("=" * 70)

    # Limpiar registry de ejecuciones anteriores
    _metadata_registry.clear()

    # Ejecutar cada categoría
    categories = [
        build_trend_indicators(market),
        build_volatility_indicators(market),
        build_valuation_indicators(market, macro),
        build_cycle_indicators(macro),
        build_monetary_indicators(macro),
        build_credit_indicators(market, macro),
        build_inflation_indicators(macro),
        build_breadth_indicators(market),
    ]

    # Combinar todos los indicadores
    # Usamos concat con outer join sobre el índice temporal.
    all_indicators = pd.concat(categories, axis=1)
    all_indicators.index.name = "date"
    all_indicators = all_indicators.sort_index()

    # Informe final
    logger.info("")
    logger.info("─" * 70)
    logger.info(f"UNIVERSO COMPLETO: {all_indicators.shape[1]} indicadores × "
                f"{all_indicators.shape[0]} meses")
    logger.info(f"Rango: {all_indicators.index.min().date()} → "
                f"{all_indicators.index.max().date()}")

    # Resumen de NaNs por indicador
    nan_summary = all_indicators.isna().sum().sort_values(ascending=False)
    logger.info(f"\nTop 10 indicadores con más NaNs:")
    for col, n_nan in nan_summary.head(10).items():
        n_valid = all_indicators[col].notna().sum()
        first = all_indicators[col].first_valid_index()
        logger.info(
            f"    {col:<40s}: {n_valid:>4d} válidos, {n_nan:>4d} NaNs | "
            f"desde {first.date() if first else 'N/A'}"
        )

    return all_indicators


def build_metadata_dataframe() -> pd.DataFrame:
    """
    Convierte el registro de metadatos en un DataFrame estructurado.

    Debe llamarse DESPUÉS de build_all_indicators() para que el
    registry esté completo.

    Retorna
    -------
    pd.DataFrame : tabla de metadatos con una fila por indicador.
    """
    if not _metadata_registry:
        logger.warning("Registry vacío. ¿Se ejecutó build_all_indicators()?")
        return pd.DataFrame()

    metadata_df = pd.DataFrame(_metadata_registry)
    metadata_df = metadata_df.set_index("indicator")

    logger.info(f"\nMetadatos: {len(metadata_df)} indicadores documentados")
    logger.info("Indicadores por categoría:")
    for cat, count in metadata_df["category"].value_counts().sort_index().items():
        logger.info(f"    {cat:<20s}: {count}")

    return metadata_df


# ══════════════════════════════════════════════════════════════════════════════
# 7. PERSISTENCIA
# ══════════════════════════════════════════════════════════════════════════════

def save_indicators(
    indicators_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    output_dir: Path = INDICATORS_DIR,
) -> dict[str, Path]:
    """
    Guarda indicadores y metadatos en disco.

    Archivos generados:
    - indicators_full.csv     → Todos los indicadores (filas = meses)
    - indicators_metadata.csv → Metadatos (filas = indicadores)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    files = {}

    path_ind = output_dir / "indicators_full.csv"
    indicators_df.to_csv(path_ind)
    files["indicators_full"] = path_ind
    logger.info(f"  ✓ {path_ind.name}: {indicators_df.shape[0]} filas × "
                f"{indicators_df.shape[1]} cols")

    path_meta = output_dir / "indicators_metadata.csv"
    metadata_df.to_csv(path_meta)
    files["indicators_metadata"] = path_meta
    logger.info(f"  ✓ {path_meta.name}: {metadata_df.shape[0]} indicadores documentados")

    return files


# ══════════════════════════════════════════════════════════════════════════════
# 8. FUNCIÓN PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

def run_indicators(
    processed_dir: Path = PROCESSED_DATA_DIR,
    output_dir: Path = INDICATORS_DIR,
) -> dict:
    """
    Ejecuta el pipeline completo del Paso 3.

    1. Carga datos procesados (del Paso 2).
    2. Construye el universo completo de indicadores.
    3. Genera tabla de metadatos.
    4. Guarda todo en disco.

    Parámetros
    ----------
    processed_dir : Path
        Directorio con market_monthly.csv y macro_monthly.csv.
    output_dir : Path
        Directorio de salida para indicadores.

    Retorna
    -------
    dict : con claves 'indicators', 'metadata', 'files'.
    """
    logger.info("╔══════════════════════════════════════════════════════════════╗")
    logger.info("║   INDICATORS — Paso 3: Universo de indicadores             ║")
    logger.info("║   Estrategia de asignación dinámica S&P 500                ║")
    logger.info("╚══════════════════════════════════════════════════════════════╝")
    logger.info(f"Entrada: {processed_dir}")
    logger.info(f"Salida:  {output_dir}")
    logger.info("")

    # --- Cargar datos ---
    market, macro = load_processed_data(processed_dir)

    # --- Construir indicadores ---
    indicators_df = build_all_indicators(market, macro)

    # --- Metadatos ---
    metadata_df = build_metadata_dataframe()

    # --- Guardar ---
    logger.info("")
    logger.info("=" * 70)
    logger.info(f"GUARDANDO → {output_dir}")
    logger.info("=" * 70)
    saved_files = save_indicators(indicators_df, metadata_df, output_dir)

    # --- Resumen ---
    logger.info("")
    logger.info("=" * 70)
    logger.info("RESUMEN DEL PASO 3")
    logger.info("=" * 70)
    logger.info(f"  Total indicadores: {indicators_df.shape[1]}")
    logger.info(f"  Meses cubiertos:   {indicators_df.shape[0]}")
    logger.info(f"  Categorías:        {metadata_df['category'].nunique()}")
    for cat, count in metadata_df["category"].value_counts().sort_index().items():
        logger.info(f"      {cat}: {count}")
    logger.info(f"  Archivos: {list(saved_files.keys())}")
    logger.info("=" * 70)

    return {
        "indicators": indicators_df,
        "metadata": metadata_df,
        "files": saved_files,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 9. EJECUCIÓN DIRECTA (CLI)
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    """
    Ejecución desde línea de comandos.

    Uso:
        python indicators.py

    Requisitos previos:
        - Haber ejecutado data_loader.py (Paso 1).
        - Haber ejecutado preprocessing.py (Paso 2).
        - Los archivos market_monthly.csv y macro_monthly.csv deben
          existir en data/processed/.
    """
    results = run_indicators()