"""
================================================================================
DATA LOADER MODULE — data_loader.py
================================================================================
Proyecto: Análisis de contexto económico-financiero para asignación dinámica
          en el S&P 500.

Propósito: Módulo de adquisición de datos (data gathering). Descarga datos
           crudos ("raw") desde fuentes públicas y gratuitas, sin
           transformaciones ni imputaciones. Los datos se guardan tal cual
           llegan de la fuente para ser procesados en fases posteriores.

Fuentes:
    1. yfinance  — Datos de mercado (precios, volatilidad, proxies macro).
    2. FRED API  — Datos macroeconómicos oficiales de la Reserva Federal
                   de St. Louis (via fredapi).

Autor: Mauro Calvo Pérez y Jorge Fernández Beloso
Fecha: 2026-02
================================================================================
"""

import os
import datetime
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf
from fredapi import Fred

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
# 1. PARÁMETROS GLOBALES DE CONFIGURACIÓN
# ══════════════════════════════════════════════════════════════════════════════
# Todas las variables configurables están centralizadas aquí para facilitar
# su modificación sin tocar la lógica del módulo.

# --- Ruta de almacenamiento ---------------------------------------------------
RAW_DATA_DIR = Path("data/raw")

# --- Rango temporal por defecto -----------------------------------------------
# Se usa un inicio amplio (1990) para disponer de varios ciclos económicos
# completos. Cada fuente puede tener disponibilidad distinta; yfinance
# devolverá datos desde la fecha más antigua disponible para cada ticker.
DEFAULT_START_DATE = "1990-01-01"
DEFAULT_END_DATE = datetime.date.today().isoformat()

# --- Tickers de yfinance -----------------------------------------------------
# Cada ticker cumple un rol específico en la estrategia de asignación:
#
# SPY   → SPDR S&P 500 ETF Trust.
#          Proxy principal del mercado de renta variable estadounidense.
#          Representa el 70 % permanente + el 30 % táctico de la cartera.
#          Disponible desde 1993-01-29.
#
# ^VIX  → CBOE Volatility Index.
#          Mide la volatilidad implícita esperada a 30 días del S&P 500.
#          Señal clave de régimen de mercado (risk-on / risk-off).
#          Disponible desde 1990-01-02 (solo precios, no es invertible).
#
# TLT   → iShares 20+ Year Treasury Bond ETF.
#          Proxy de bonos soberanos de largo plazo (duración ~17 años).
#          Captura expectativas de tipos de interés y flight-to-quality.
#          Disponible desde 2002-07-30.
#
# TIP   → iShares TIPS Bond ETF.
#          Proxy de bonos ligados a la inflación (Treasury Inflation-Protected).
#          Refleja expectativas reales de inflación del mercado.
#          Disponible desde 2003-12-05.
#
# LQD   → iShares iBoxx $ Investment Grade Corporate Bond ETF.
#          Proxy del mercado de crédito corporativo investment grade.
#          Captura spreads de crédito y apetito por riesgo en renta fija.
#          Disponible desde 2002-07-26.
#
# HYG   → iShares iBoxx $ High Yield Corporate Bond ETF.
#          Proxy de deuda corporativa high yield (alto rendimiento).
#          Spread HYG-LQD mide estrés en crédito de menor calidad.
#          Disponible desde 2007-04-11.
#
# GLD   → SPDR Gold Shares ETF.
#          Proxy del precio del oro como activo refugio.
#          Señal de incertidumbre macro y cobertura inflacionaria.
#          Disponible desde 2004-11-18.
#
# Limitaciones de yfinance:
#   - Los datos más antiguos pueden tener gaps o ajustes retrospectivos.
#   - La profundidad histórica varía por ticker (SPY desde 1993, HYG desde 2007).
#   - yfinance no es una fuente oficial; depende del scraping de Yahoo Finance.
#   - Puede haber interrupciones temporales del servicio o cambios de API.
#   - Los precios "Adj Close" incorporan dividendos y splits, pero pueden
#     recalcularse retroactivamente por Yahoo sin previo aviso.

YFINANCE_TICKERS: dict[str, str] = {
    "SPY":  "S&P 500 ETF — activo principal de la estrategia",
    "^VIX": "CBOE Volatility Index — régimen de volatilidad",
    "TLT":  "Bonos largo plazo 20Y+ — tipos de interés y refugio",
    "TIP":  "TIPS — expectativas de inflación real",
    "LQD":  "Crédito investment grade — spreads de crédito",
    "HYG":  "Crédito high yield — estrés crediticio",
    "GLD":  "Oro — activo refugio e inflación",
}

# --- Series de FRED -----------------------------------------------------------
# Cada serie tiene un código único en la base de datos de FRED.
#
# CPIAUCSL  → Consumer Price Index for All Urban Consumers (CPI).
#              Medida principal de inflación. Frecuencia mensual.
#              Serie no desestacionalizada disponible desde 1947.
#
# UNRATE    → Civilian Unemployment Rate.
#              Tasa de desempleo oficial (U-3). Frecuencia mensual.
#              Indicador retrasado del ciclo económico.
#              Disponible desde 1948.
#
# FEDFUNDS  → Effective Federal Funds Rate.
#              Tipo de interés interbancario a un día (overnight).
#              Instrumento principal de política monetaria de la Fed.
#              Frecuencia mensual (media del periodo). Disponible desde 1954.
#
# DFF       → Effective Federal Funds Rate (diaria).
#              Misma serie que FEDFUNDS pero en frecuencia diaria.
#              Útil para análisis más granulares de política monetaria.
#              Disponible desde 1954.
#
# T10Y2Y    → 10-Year Treasury Constant Maturity Minus 2-Year.
#              Spread de la curva de tipos. Una inversión (valores negativos)
#              históricamente precede recesiones con 6-18 meses de antelación.
#              Frecuencia diaria. Disponible desde 1976.
#
# GS10      → 10-Year Treasury Constant Maturity Rate.
#              Tipo de interés nominal del bono a 10 años.
#              Referencia para valoración de activos y coste de capital.
#              Frecuencia mensual. Disponible desde 1953.
#
# GS2       → 2-Year Treasury Constant Maturity Rate.
#              Tipo de interés a corto plazo del Tesoro.
#              Sensible a expectativas de política monetaria de la Fed.
#              Frecuencia mensual. Disponible desde 1976.
#
# INDPRO    → Industrial Production Index.
#              Mide la producción real del sector industrial, minero y
#              utilities. Indicador coincidente del ciclo económico.
#              Frecuencia mensual. Disponible desde 1919.
#
# USREC     → NBER Based Recession Indicators.
#              Variable binaria: 1 = recesión, 0 = expansión.
#              Definida por el NBER Business Cycle Dating Committee.
#              IMPORTANTE: las fechas de recesión se publican con retraso
#              significativo (a veces más de un año), por lo que esta serie
#              NO puede usarse como señal en tiempo real. Solo sirve para
#              backtesting y validación histórica.
#              Frecuencia mensual. Disponible desde 1854.
#
# T10YIE    → 10-Year Breakeven Inflation Rate.
#              Diferencia entre el rendimiento nominal del bono a 10 años
#              y el rendimiento del TIPS a 10 años. Proxy de las
#              expectativas de inflación del mercado.
#              Frecuencia diaria. Disponible desde 2003.
#
# BAMLH0A0HYM2  → ICE BofA US High Yield Option-Adjusted Spread.
#              Spread de crédito high yield sobre Treasuries.
#              Mide el estrés en el mercado de crédito de menor calidad.
#              Frecuencia diaria. Disponible desde 1996.
#
# Limitaciones de FRED:
#   - Muchas series son mensuales o trimestrales; no se pueden granularizar.
#   - Las series macroeconómicas se revisan retroactivamente (el dato publicado
#     en una fecha puede cambiar meses después). Esto introduce look-ahead
#     bias si no se usa la serie "vintage" o "real-time".
#   - fredapi requiere una API key gratuita (registro en fred.stlouisfed.org).
#   - Algunas series se descontinúan o cambian de código sin aviso.
#   - Los datos de recesión (USREC) tienen retraso de publicación.

FRED_SERIES: dict[str, str] = {
    "CPIAUCSL":       "CPI — inflación general (mensual)",
    "UNRATE":         "Tasa de desempleo U-3 (mensual)",
    "FEDFUNDS":       "Fed Funds Rate — tipo oficial mensual",
    "DFF":            "Fed Funds Rate — tipo oficial diario",
    "T10Y2Y":         "Spread curva 10Y-2Y — inversión de curva (diario)",
    "GS10":           "Rendimiento Treasury 10Y (mensual)",
    "GS2":            "Rendimiento Treasury 2Y (mensual)",
    "INDPRO":         "Producción industrial (mensual)",
    "USREC":          "Indicador recesión NBER (mensual, con retraso)",
    "T10YIE":         "Breakeven inflation 10Y (diario)",
    "BAMLH0A0HYM2":  "Spread high yield OAS (diario)",
}


# ══════════════════════════════════════════════════════════════════════════════
# 2. FUNCIONES AUXILIARES
# ══════════════════════════════════════════════════════════════════════════════

def _ensure_output_dir(directory: Path) -> None:
    """Crea el directorio de salida si no existe."""
    directory.mkdir(parents=True, exist_ok=True)
    logger.info(f"Directorio de salida verificado: {directory}")


def _save_dataframe(
    df: pd.DataFrame,
    filename: str,
    directory: Path,
    file_format: str = "csv",
) -> Path:
    """
    Guarda un DataFrame en disco.

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame a guardar.
    filename : str
        Nombre del archivo (sin extensión).
    directory : Path
        Carpeta de destino.
    file_format : str
        'csv' o 'parquet'. Por defecto 'csv' para máxima portabilidad.

    Retorna
    -------
    Path : ruta completa del archivo guardado.
    """
    _ensure_output_dir(directory)

    if file_format == "parquet":
        filepath = directory / f"{filename}.parquet"
        df.to_parquet(filepath, engine="pyarrow")
    else:
        filepath = directory / f"{filename}.csv"
        df.to_csv(filepath)

    logger.info(f"Guardado: {filepath}  ({len(df)} filas, {len(df.columns)} cols)")
    return filepath


# ══════════════════════════════════════════════════════════════════════════════
# 3. DESCARGA DESDE YFINANCE
# ══════════════════════════════════════════════════════════════════════════════

def download_yfinance_data(
    tickers: Optional[dict[str, str]] = None,
    start_date: str = DEFAULT_START_DATE,
    end_date: str = DEFAULT_END_DATE,
    output_dir: Path = RAW_DATA_DIR,
    file_format: str = "csv",
) -> dict[str, pd.DataFrame]:
    """
    Descarga datos históricos diarios desde Yahoo Finance via yfinance.

    Se descargan todos los campos OHLCV + Adj Close para cada ticker.
    No se aplican transformaciones, filtros ni imputaciones.

    Parámetros
    ----------
    tickers : dict[str, str], opcional
        Diccionario {ticker: descripción}. Por defecto usa YFINANCE_TICKERS.
    start_date : str
        Fecha de inicio en formato 'YYYY-MM-DD'.
    end_date : str
        Fecha de fin en formato 'YYYY-MM-DD'.
    output_dir : Path
        Directorio donde se guardan los archivos.
    file_format : str
        'csv' o 'parquet'.

    Retorna
    -------
    dict[str, pd.DataFrame]
        Diccionario {ticker: DataFrame} con los datos descargados.
    """
    if tickers is None:
        tickers = YFINANCE_TICKERS

    logger.info("=" * 70)
    logger.info("INICIO: Descarga de datos de mercado desde yfinance")
    logger.info(f"Periodo: {start_date} → {end_date}")
    logger.info(f"Tickers: {list(tickers.keys())}")
    logger.info("=" * 70)

    results: dict[str, pd.DataFrame] = {}

    for ticker, description in tickers.items():
        logger.info(f"Descargando {ticker} — {description}...")

        try:
            # yf.download devuelve datos OHLCV + Adj Close.
            # auto_adjust=False para conservar tanto Close como Adj Close.
            # actions=True incluye dividendos y splits como columnas extra.
            df = yf.download(
                tickers=ticker,
                start=start_date,
                end=end_date,
                auto_adjust=False,
                actions=True,
                progress=False,
            )

            if df.empty:
                logger.warning(f"  ⚠ Sin datos para {ticker}. Saltando.")
                continue

            # Si yf.download devuelve MultiIndex en columnas (puede ocurrir
            # incluso con un solo ticker en algunas versiones), aplanar.
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            # Metadatos básicos del dataset descargado
            logger.info(
                f"  ✓ {ticker}: {len(df)} filas, "
                f"desde {df.index.min().date()} hasta {df.index.max().date()}"
            )

            # Guardar archivo individual por ticker
            # El nombre de archivo reemplaza caracteres problemáticos (^VIX → VIX)
            safe_name = ticker.replace("^", "").replace("/", "_")
            _save_dataframe(df, f"yf_{safe_name}", output_dir, file_format)

            results[ticker] = df

        except Exception as e:
            logger.error(f"  ✗ Error descargando {ticker}: {e}")
            continue

    logger.info(f"yfinance: {len(results)}/{len(tickers)} tickers descargados.")
    return results


# ══════════════════════════════════════════════════════════════════════════════
# 4. DESCARGA DESDE FRED
# ══════════════════════════════════════════════════════════════════════════════

def download_fred_data(
    fred_api_key: str,
    series: Optional[dict[str, str]] = None,
    start_date: str = DEFAULT_START_DATE,
    end_date: str = DEFAULT_END_DATE,
    output_dir: Path = RAW_DATA_DIR,
    file_format: str = "csv",
) -> dict[str, pd.Series]:
    """
    Descarga series macroeconómicas desde FRED (Federal Reserve Economic Data).

    Cada serie se descarga individualmente y se guarda como archivo separado.
    Adicionalmente se genera un DataFrame consolidado con todas las series.

    Parámetros
    ----------
    fred_api_key : str
        API key de FRED. Se obtiene gratuitamente registrándose en
        https://fred.stlouisfed.org/docs/api/api_key.html
    series : dict[str, str], opcional
        Diccionario {código_serie: descripción}. Por defecto usa FRED_SERIES.
    start_date : str
        Fecha de inicio.
    end_date : str
        Fecha de fin.
    output_dir : Path
        Directorio de salida.
    file_format : str
        'csv' o 'parquet'.

    Retorna
    -------
    dict[str, pd.Series]
        Diccionario {código_serie: Serie} con los datos descargados.
    """
    if series is None:
        series = FRED_SERIES

    logger.info("=" * 70)
    logger.info("INICIO: Descarga de datos macroeconómicos desde FRED")
    logger.info(f"Periodo: {start_date} → {end_date}")
    logger.info(f"Series: {list(series.keys())}")
    logger.info("=" * 70)

    # Inicializar cliente de FRED
    try:
        fred = Fred(api_key=os.getenv("FRED_API_KEY"))
        logger.info("Conexión con FRED API establecida.")
    except Exception as e:
        logger.error(f"Error conectando con FRED API: {e}")
        raise

    results: dict[str, pd.Series] = {}

    for series_id, description in series.items():
        logger.info(f"Descargando {series_id} — {description}...")

        try:
            data = fred.get_series(
                series_id=series_id,
                observation_start=start_date,
                observation_end=end_date,
            )

            if data is None or data.empty:
                logger.warning(f"  ⚠ Sin datos para {series_id}. Saltando.")
                continue

            # Asignar nombre a la serie para identificarla al consolidar
            data.name = series_id
            data.index.name = "date"

            logger.info(
                f"  ✓ {series_id}: {len(data)} observaciones, "
                f"desde {data.index.min().date()} hasta {data.index.max().date()}"
            )

            # Guardar serie individual
            df_single = data.to_frame()
            _save_dataframe(df_single, f"fred_{series_id}", output_dir, file_format)

            results[series_id] = data

        except Exception as e:
            logger.error(f"  ✗ Error descargando {series_id}: {e}")
            continue

    # ── Consolidar todas las series FRED en un solo archivo ──────────────
    # Esto facilita el uso posterior sin necesidad de cargar archivos
    # individuales. Se usa un outer join para no perder observaciones
    # de series con distinta frecuencia (diaria vs mensual).
    if results:
        consolidated = pd.DataFrame(results)
        consolidated.index.name = "date"
        _save_dataframe(consolidated, "fred_consolidated", output_dir, file_format)
        logger.info(
            f"Archivo consolidado FRED: {len(consolidated)} filas, "
            f"{len(consolidated.columns)} series."
        )

    logger.info(f"FRED: {len(results)}/{len(series)} series descargadas.")
    return results


# ══════════════════════════════════════════════════════════════════════════════
# 5. FUNCIÓN PRINCIPAL — ORQUESTADOR
# ══════════════════════════════════════════════════════════════════════════════

def download_all(
    fred_api_key: str,
    start_date: str = DEFAULT_START_DATE,
    end_date: str = DEFAULT_END_DATE,
    output_dir: Path = RAW_DATA_DIR,
    file_format: str = "csv",
) -> dict:
    """
    Ejecuta la descarga completa de datos de mercado y macroeconómicos.

    Es el punto de entrada principal del módulo. Llama secuencialmente
    a las funciones de yfinance y FRED.

    Parámetros
    ----------
    fred_api_key : str
        API key de FRED.
    start_date : str
        Fecha de inicio global.
    end_date : str
        Fecha de fin global.
    output_dir : Path
        Directorio de salida para todos los archivos.
    file_format : str
        'csv' o 'parquet'.

    Retorna
    -------
    dict
        Diccionario con claves 'yfinance' y 'fred', cada una conteniendo
        los datos descargados de su fuente respectiva.
    """
    logger.info("╔══════════════════════════════════════════════════════════════╗")
    logger.info("║   DATA LOADER — Módulo de adquisición de datos             ║")
    logger.info("║   Estrategia de asignación dinámica S&P 500                ║")
    logger.info("╚══════════════════════════════════════════════════════════════╝")
    logger.info(f"Periodo global: {start_date} → {end_date}")
    logger.info(f"Directorio de salida: {output_dir}")
    logger.info(f"Formato de archivo: {file_format}")
    logger.info("")

    # --- Paso 1: Datos de mercado (yfinance) ---
    yf_data = download_yfinance_data(
        start_date=start_date,
        end_date=end_date,
        output_dir=output_dir,
        file_format=file_format,
    )

    # --- Paso 2: Datos macroeconómicos (FRED) ---
    fred_data = download_fred_data(
        fred_api_key=fred_api_key,
        start_date=start_date,
        end_date=end_date,
        output_dir=output_dir,
        file_format=file_format,
    )

    # --- Resumen final ---
    logger.info("")
    logger.info("=" * 70)
    logger.info("RESUMEN DE DESCARGA")
    logger.info("=" * 70)
    logger.info(f"  yfinance : {len(yf_data)} tickers descargados")
    logger.info(f"  FRED     : {len(fred_data)} series descargadas")
    logger.info(f"  Archivos guardados en: {output_dir.resolve()}")
    logger.info("=" * 70)

    return {
        "yfinance": yf_data,
        "fred": fred_data,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 6. EJECUCIÓN DIRECTA (CLI)
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    """
    Ejecución desde línea de comandos.

    Uso:
        python data_loader.py

    Requisitos previos:
        1. Instalar dependencias:
           pip install yfinance fredapi pandas pyarrow

        2. Configurar la API key de FRED:
           - Opción A: Variable de entorno FRED_API_KEY
           - Opción B: Modificar directamente la variable a continuación

    La API key de FRED es gratuita. Se obtiene en:
        https://fred.stlouisfed.org/docs/api/api_key.html
    """
    # --- Obtener API key de FRED -----------------------------------------------
    # Prioridad: variable de entorno > valor hardcoded (solo para testing)
    FRED_API_KEY = os.environ.get("FRED_API_KEY", "")

    if not FRED_API_KEY:
        logger.error(
            "No se encontró FRED_API_KEY. "
            "Configúrala como variable de entorno o edita este archivo.\n"
            "  export FRED_API_KEY='tu_api_key_aqui'\n"
            "  Registro gratuito: https://fred.stlouisfed.org/docs/api/api_key.html"
        )
        # Aún así se ejecuta la descarga de yfinance
        logger.info("Continuando solo con la descarga de yfinance...")
        download_yfinance_data()
    else:
        download_all(fred_api_key=FRED_API_KEY)