"""
================================================================================
PREPROCESSING MODULE — preprocessing.py
================================================================================
Proyecto: Análisis de contexto económico-financiero para asignación dinámica
          en el S&P 500.

Propósito: PASO 2 — Validación, limpieza básica y alineación temporal de los
           datos crudos descargados en el Paso 1 (data_loader.py).

           Este módulo transforma datos raw heterogéneos (distintas frecuencias,
           coberturas y convenciones) en datasets mensuales alineados, listos
           para la construcción de features en fases posteriores.

Flujo conceptual:
    1. AUDITORÍA   → Leer cada archivo raw, detectar frecuencia real,
                     documentar cobertura temporal, NaNs y anomalías.
    2. NORMALIZACIÓN → Estandarizar índices temporales (timezone-naive,
                     DatetimeIndex, sin duplicados).
    3. RESAMPLING  → Convertir series diarias a mensuales con método
                     justificado para cada tipo de dato.
    4. ALINEACIÓN  → Merge en un eje temporal común (mes calendario).
    5. PERSISTENCIA → Guardar datasets procesados en data/processed/.

Decisiones metodológicas clave:

    ► Resampling de precios de mercado (yfinance):
      Se usa el ÚLTIMO valor disponible del mes calendario ("last business day").
      Justificación: en finanzas, los precios de cierre de fin de mes son el
      estándar para análisis mensual porque representan el valor de liquidación
      real del periodo. Usar media mensual suavizaría artificialmente la
      volatilidad y distorsionaría los retornos calculables en fases posteriores.

    ► Resampling de VIX:
      Se usa la MEDIA mensual (no el último valor).
      Justificación: el VIX es un indicador de régimen, no un precio de activo.
      La media mensual captura mejor el nivel de volatilidad implícita
      prevalente durante todo el mes, no solo el snapshot del último día.

    ► Resampling de series FRED diarias (DFF, T10Y2Y, T10YIE, BAMLH0A0HYM2):
      Se usa el ÚLTIMO valor disponible del mes.
      Justificación: las series FRED diarias representan tasas o spreads que
      se publican con la información disponible hasta esa fecha. Tomar el
      último valor del mes es consistente con usar la información más
      actualizada al cierre del periodo, sin introducir look-ahead bias.

    ► Series FRED ya mensuales (CPIAUCSL, UNRATE, FEDFUNDS, GS10, GS2,
      INDPRO, USREC):
      Se mantienen tal cual. Solo se normaliza el índice temporal al último
      día del mes calendario para que sea alineable con el resto.

    ► Prevención de look-ahead bias:
      - No se aplica forward-fill entre meses. Si un dato no existe para
        un mes, queda como NaN.
      - No se usa información de meses futuros para rellenar meses pasados.
      - El resampling usa solo datos del propio mes (no ventanas centradas).
      - USREC se marca explícitamente como serie con retraso de publicación
        (no apta para señales en tiempo real).

    ► Cobertura temporal:
      Cada serie tiene distinta fecha de inicio. Se documenta explícitamente
      y NO se recorta al mínimo común denominador en esta fase (se perdería
      información valiosa de series largas como VIX o CPI). El recorte a un
      rango común es decisión de las fases posteriores según la necesidad.

Entrada:  data/raw/  (archivos CSV del data_loader.py)
Salida:   data/processed/
          ├── market_monthly.csv
          ├── macro_monthly.csv
          ├── combined_monthly_raw.csv
          └── audit_report.csv

Autor: Mauro Calvo Pérez y Jorge Fernández Beloso
Fecha: 2026-02
================================================================================
"""

import logging
from pathlib import Path
from typing import Optional
import pandas as pd

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
# 1. CONFIGURACIÓN Y MAPEOS
# ══════════════════════════════════════════════════════════════════════════════

RAW_DATA_DIR = Path("data/raw")
PROCESSED_DATA_DIR = Path("data/processed")

# ---------------------------------------------------------------------------
# Archivos de mercado generados por data_loader.py
# Mapeo: nombre_archivo → { columna principal a extraer, método de resampling }
#
# "column": columna que se extraerá como serie principal para el dataset
#           mensual. Se usa "Adj Close" para precios (incorpora dividendos
#           y splits) y "Close" para VIX (que no es un activo negociable).
#
# "resample_method": cómo agregar de diario a mensual.
#   - "last"  → último valor del mes (estándar para precios y tasas).
#   - "mean"  → media del mes (apropiado para indicadores de régimen).
# ---------------------------------------------------------------------------
MARKET_FILES: dict[str, dict] = {
    "yf_SPY": {
        "column": "Adj Close",
        "resample_method": "last",
        "output_name": "SPY",
        "description": "S&P 500 ETF — precio ajustado fin de mes",
    },
    "yf_VIX": {
        "column": "Close",
        "resample_method": "mean",
        "output_name": "VIX",
        "description": "CBOE VIX — media mensual de volatilidad implícita",
    },
    "yf_TLT": {
        "column": "Adj Close",
        "resample_method": "last",
        "output_name": "TLT",
        "description": "Treasury 20Y+ ETF — precio ajustado fin de mes",
    },
    "yf_TIP": {
        "column": "Adj Close",
        "resample_method": "last",
        "output_name": "TIP",
        "description": "TIPS ETF — precio ajustado fin de mes",
    },
    "yf_LQD": {
        "column": "Adj Close",
        "resample_method": "last",
        "output_name": "LQD",
        "description": "IG Corporate Bond ETF — precio ajustado fin de mes",
    },
    "yf_HYG": {
        "column": "Adj Close",
        "resample_method": "last",
        "output_name": "HYG",
        "description": "HY Corporate Bond ETF — precio ajustado fin de mes",
    },
    "yf_GLD": {
        "column": "Adj Close",
        "resample_method": "last",
        "output_name": "GLD",
        "description": "Gold ETF — precio ajustado fin de mes",
    },
}

# ---------------------------------------------------------------------------
# Archivos FRED generados por data_loader.py
# Mapeo: nombre_archivo → { frecuencia original, método de resampling }
#
# "native_freq": frecuencia con la que FRED publica la serie.
#   - "monthly" → ya es mensual, solo se normaliza el índice.
#   - "daily"   → se resampled a mensual.
#
# "resample_method": solo aplica a series diarias.
#   - "last" → último valor del mes (tasas, spreads).
#   - "mean" → media mensual (alternativa, no usada aquí).
#
# Nota sobre USREC: se incluye con flag "lagged_publication" para recordar
# que esta serie tiene retraso de publicación y NO debe usarse como señal
# prospectiva en backtesting sin ajustar el lag.
# ---------------------------------------------------------------------------
FRED_FILES: dict[str, dict] = {
    "fred_CPIAUCSL": {
        "native_freq": "monthly",
        "resample_method": None,
        "output_name": "CPI",
        "description": "Consumer Price Index — nivel mensual",
    },
    "fred_UNRATE": {
        "native_freq": "monthly",
        "resample_method": None,
        "output_name": "UNRATE",
        "description": "Tasa de desempleo U-3 — mensual",
    },
    "fred_FEDFUNDS": {
        "native_freq": "monthly",
        "resample_method": None,
        "output_name": "FEDFUNDS",
        "description": "Fed Funds Rate — media mensual (ya publicada así)",
    },
    "fred_DFF": {
        "native_freq": "daily",
        "resample_method": "last",
        "output_name": "DFF",
        "description": "Fed Funds Rate diario — último valor del mes",
    },
    "fred_T10Y2Y": {
        "native_freq": "daily",
        "resample_method": "last",
        "output_name": "T10Y2Y",
        "description": "Spread curva 10Y-2Y — último valor del mes",
    },
    "fred_GS10": {
        "native_freq": "monthly",
        "resample_method": None,
        "output_name": "GS10",
        "description": "Treasury 10Y rate — mensual",
    },
    "fred_GS2": {
        "native_freq": "monthly",
        "resample_method": None,
        "output_name": "GS2",
        "description": "Treasury 2Y rate — mensual",
    },
    "fred_INDPRO": {
        "native_freq": "monthly",
        "resample_method": None,
        "output_name": "INDPRO",
        "description": "Producción industrial — índice mensual",
    },
    "fred_USREC": {
        "native_freq": "monthly",
        "resample_method": None,
        "output_name": "USREC",
        "description": "Recesión NBER — binaria mensual (RETRASO PUBLICACIÓN)",
        "lagged_publication": True,
    },
    "fred_T10YIE": {
        "native_freq": "daily",
        "resample_method": "last",
        "output_name": "T10YIE",
        "description": "Breakeven inflation 10Y — último valor del mes",
    },
    "fred_BAMLH0A0HYM2": {
        "native_freq": "daily",
        "resample_method": "last",
        "output_name": "HY_OAS",
        "description": "High Yield OAS spread — último valor del mes",
    },
}


# ══════════════════════════════════════════════════════════════════════════════
# 2. AUDITORÍA DE DATOS RAW
# ══════════════════════════════════════════════════════════════════════════════

def audit_raw_file(filepath: Path) -> dict:
    """
    Realiza una auditoría básica de un archivo CSV raw.

    Inspecciona el archivo sin modificarlo y devuelve un diccionario con
    metadatos: rango temporal, número de filas, NaNs por columna,
    duplicados en el índice y frecuencia inferida.

    Parámetros
    ----------
    filepath : Path
        Ruta al archivo CSV.

    Retorna
    -------
    dict : diccionario con resultados de la auditoría.
    """
    audit = {
        "file": filepath.name,
        "exists": filepath.exists(),
    }

    if not filepath.exists():
        logger.warning(f"  ⚠ Archivo no encontrado: {filepath}")
        audit["error"] = "FILE_NOT_FOUND"
        return audit

    try:
        df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    except Exception as e:
        logger.error(f"  ✗ Error leyendo {filepath.name}: {e}")
        audit["error"] = str(e)
        return audit

    # --- Metadatos básicos ---
    audit["rows"] = len(df)
    audit["columns"] = list(df.columns)
    audit["date_min"] = str(df.index.min().date()) if len(df) > 0 else None
    audit["date_max"] = str(df.index.max().date()) if len(df) > 0 else None

    # --- Duplicados en el índice temporal ---
    n_duplicates = df.index.duplicated().sum()
    audit["index_duplicates"] = int(n_duplicates)

    # --- NaNs por columna ---
    nan_counts = df.isna().sum().to_dict()
    audit["nans_per_column"] = nan_counts
    audit["total_nans"] = int(df.isna().sum().sum())

    # --- Frecuencia inferida ---
    # pd.infer_freq intenta detectar la frecuencia del índice temporal.
    # Devuelve None si no puede inferirla (gaps irregulares, datos mixtos).
    try:
        inferred_freq = pd.infer_freq(df.index)
    except (ValueError, TypeError):
        inferred_freq = None
    audit["inferred_freq"] = inferred_freq

    # --- Frecuencia estimada por mediana de diferencias ---
    # Más robusto que infer_freq para series con gaps (festivos, fines de semana).
    if len(df) > 1:
        median_delta = df.index.to_series().diff().median()
        audit["median_delta_days"] = median_delta.days
    else:
        audit["median_delta_days"] = None

    return audit


def run_full_audit(
    raw_dir: Path = RAW_DATA_DIR,
    market_files: dict = None,
    fred_files: dict = None,
) -> pd.DataFrame:
    """
    Ejecuta la auditoría completa sobre todos los archivos raw.

    Recorre todos los archivos definidos en los mapeos de mercado y FRED,
    genera un informe tabular y lo loguea.

    Parámetros
    ----------
    raw_dir : Path
        Directorio de datos raw.
    market_files : dict
        Mapeo de archivos de mercado. Por defecto MARKET_FILES.
    fred_files : dict
        Mapeo de archivos FRED. Por defecto FRED_FILES.

    Retorna
    -------
    pd.DataFrame : tabla de auditoría con una fila por archivo.
    """
    if market_files is None:
        market_files = MARKET_FILES
    if fred_files is None:
        fred_files = FRED_FILES

    logger.info("=" * 70)
    logger.info("AUDITORÍA DE DATOS RAW")
    logger.info("=" * 70)

    all_audits = []

    # Auditar archivos de mercado
    logger.info("--- Archivos de mercado (yfinance) ---")
    for filename, config in market_files.items():
        filepath = raw_dir / f"{filename}.csv"
        audit = audit_raw_file(filepath)
        audit["source"] = "yfinance"
        audit["output_name"] = config["output_name"]
        all_audits.append(audit)

        if "error" not in audit:
            logger.info(
                f"  {config['output_name']:>8s} | "
                f"{audit['rows']:>6d} filas | "
                f"{audit['date_min']} → {audit['date_max']} | "
                f"NaNs: {audit['total_nans']:>5d} | "
                f"Δ mediana: {audit['median_delta_days']}d | "
                f"freq: {audit['inferred_freq']}"
            )

    # Auditar archivos FRED
    logger.info("--- Archivos macroeconómicos (FRED) ---")
    for filename, config in fred_files.items():
        filepath = raw_dir / f"{filename}.csv"
        audit = audit_raw_file(filepath)
        audit["source"] = "FRED"
        audit["output_name"] = config["output_name"]
        audit["native_freq"] = config["native_freq"]
        all_audits.append(audit)

        if "error" not in audit:
            logger.info(
                f"  {config['output_name']:>8s} | "
                f"{audit['rows']:>6d} filas | "
                f"{audit['date_min']} → {audit['date_max']} | "
                f"NaNs: {audit['total_nans']:>5d} | "
                f"Δ mediana: {audit['median_delta_days']}d | "
                f"native: {config['native_freq']}"
            )

    # Construir tabla de auditoría
    audit_df = pd.DataFrame(all_audits)
    logger.info(f"\nAuditoría completada: {len(audit_df)} archivos inspeccionados.")

    return audit_df


# ══════════════════════════════════════════════════════════════════════════════
# 3. CARGA Y NORMALIZACIÓN DE ÍNDICE TEMPORAL
# ══════════════════════════════════════════════════════════════════════════════

def load_and_normalize_index(filepath: Path) -> pd.DataFrame:
    """
    Carga un CSV raw y normaliza su índice temporal.

    Operaciones realizadas:
    1. Parsear índice como DatetimeIndex.
    2. Eliminar timezone info si existe (→ timezone-naive).
       Justificación: los datos de mercado de yfinance a veces incluyen
       timezone (UTC o US/Eastern). Para alineación mensual no es relevante
       y complicaría los merges.
    3. Eliminar filas con índice duplicado (conservar la primera).
       Justificación: duplicados son errores de la fuente, no datos reales.
    4. Ordenar cronológicamente.

    NO se hace:
    - Forward-fill de valores.
    - Eliminación de NaNs en columnas de datos.
    - Ninguna transformación de valores.

    Parámetros
    ----------
    filepath : Path
        Ruta al archivo CSV.

    Retorna
    -------
    pd.DataFrame : DataFrame con índice temporal normalizado.
    """
    df = pd.read_csv(filepath, index_col=0, parse_dates=True)

    # --- Eliminar timezone info ---
    if df.index.tz is not None:
        logger.info(f"    Eliminando timezone ({df.index.tz}) del índice.")
        df.index = df.index.tz_localize(None)

    # --- Eliminar duplicados en índice ---
    n_dup = df.index.duplicated().sum()
    if n_dup > 0:
        logger.info(f"    Eliminando {n_dup} filas con fecha duplicada.")
        df = df[~df.index.duplicated(keep="first")]

    # --- Ordenar cronológicamente ---
    df = df.sort_index()

    # --- Asegurar que el índice tiene nombre ---
    df.index.name = "date"

    return df


# ══════════════════════════════════════════════════════════════════════════════
# 4. RESAMPLING A FRECUENCIA MENSUAL
# ══════════════════════════════════════════════════════════════════════════════

def resample_to_monthly(
    series: pd.Series,
    method: str = "last",
) -> pd.Series:
    """
    Convierte una serie temporal (diaria o irregular) a frecuencia mensual.

    El resampling se hace al mes calendario completo ("ME" = Month End),
    lo que asigna cada observación al mes al que pertenece.

    Métodos disponibles:
    - "last": último valor no-NaN del mes. Estándar para precios y tasas.
      Equivale a "¿cuál era el valor al cierre del periodo?"
    - "mean": media de todos los valores del mes. Apropiado para indicadores
      de régimen (VIX) donde interesa el nivel prevalente, no el snapshot.

    Prevención de look-ahead bias:
    - Solo se usan datos DENTRO del mes para calcular el valor mensual.
    - No se aplica ninguna ventana que cruce límites de mes.
    - Si un mes no tiene datos, el resultado es NaN (no se rellena).

    Parámetros
    ----------
    series : pd.Series
        Serie temporal con DatetimeIndex.
    method : str
        "last" o "mean".

    Retorna
    -------
    pd.Series : serie resampleada a frecuencia mensual (Month End).
    """
    resampler = series.resample("ME")

    if method == "last":
        result = resampler.last()
    elif method == "mean":
        result = resampler.mean()
    else:
        raise ValueError(f"Método de resampling no soportado: {method}")

    return result


def normalize_monthly_index(series: pd.Series) -> pd.Series:
    """
    Normaliza el índice de una serie ya mensual al último día del mes.

    Las series FRED mensuales publican sus datos con fecha del primer día
    del mes (e.g., 2024-01-01 para el dato de enero). Para alinear con
    las series resampleadas (que usan fin de mes), movemos el índice
    al último día del mes correspondiente.

    Esto NO cambia los valores ni introduce look-ahead bias: el dato
    de enero sigue siendo el dato de enero, solo cambia la etiqueta
    de 2024-01-01 a 2024-01-31.

    Parámetros
    ----------
    series : pd.Series
        Serie con frecuencia mensual (fechas típicamente al inicio del mes).

    Retorna
    -------
    pd.Series : misma serie con índice al último día de cada mes.
    """
    # to_period('M') agrupa por mes; to_timestamp('M') devuelve fin de mes.
    # Esto unifica tanto series con fecha al inicio como al final del mes.
    series.index = series.index.to_period("M").to_timestamp("M")
    return series


# ══════════════════════════════════════════════════════════════════════════════
# 5. PROCESAMIENTO DE DATOS DE MERCADO
# ══════════════════════════════════════════════════════════════════════════════

def process_market_data(
    raw_dir: Path = RAW_DATA_DIR,
    market_files: dict = None,
) -> pd.DataFrame:
    """
    Procesa todos los archivos de mercado (yfinance) a frecuencia mensual.

    Para cada ticker:
    1. Carga el archivo raw y normaliza el índice.
    2. Extrae la columna principal (Adj Close o Close según configuración).
    3. Resamplea de diario a mensual con el método configurado.
    4. Documenta pérdida de datos y cobertura.

    El DataFrame resultante tiene una columna por ticker y un índice
    mensual (fin de mes).

    Parámetros
    ----------
    raw_dir : Path
        Directorio de datos raw.
    market_files : dict
        Mapeo de configuración. Por defecto MARKET_FILES.

    Retorna
    -------
    pd.DataFrame : datos de mercado mensuales (columnas = tickers).
    """
    if market_files is None:
        market_files = MARKET_FILES

    logger.info("=" * 70)
    logger.info("PROCESAMIENTO DE DATOS DE MERCADO → MENSUAL")
    logger.info("=" * 70)

    monthly_series = {}

    for filename, config in market_files.items():
        filepath = raw_dir / f"{filename}.csv"
        output_name = config["output_name"]
        target_col = config["column"]
        method = config["resample_method"]

        logger.info(f"\nProcesando {output_name} ({filename}.csv)...")

        # --- Cargar y normalizar ---
        if not filepath.exists():
            logger.warning(f"  ⚠ Archivo no encontrado: {filepath}. Saltando.")
            continue

        df = load_and_normalize_index(filepath)

        # --- Verificar que la columna objetivo existe ---
        if target_col not in df.columns:
            # Intentar buscar la columna sin case-sensitivity
            col_match = [c for c in df.columns if c.lower() == target_col.lower()]
            if col_match:
                target_col = col_match[0]
                logger.info(f"    Columna encontrada como '{target_col}'")
            else:
                logger.warning(
                    f"  ⚠ Columna '{target_col}' no encontrada en {filename}. "
                    f"Columnas disponibles: {list(df.columns)}. Saltando."
                )
                continue

        daily_series = df[target_col].copy()
        n_daily = len(daily_series)
        n_daily_nans = daily_series.isna().sum()

        # --- Resamplear a mensual ---
        monthly = resample_to_monthly(daily_series, method=method)
        n_monthly = len(monthly)
        n_monthly_nans = monthly.isna().sum()

        logger.info(
            f"    Diario: {n_daily} obs ({n_daily_nans} NaNs) | "
            f"{daily_series.index.min().date()} → {daily_series.index.max().date()}"
        )
        logger.info(
            f"    Mensual ({method}): {n_monthly} meses ({n_monthly_nans} NaNs) | "
            f"{monthly.index.min().date()} → {monthly.index.max().date()}"
        )

        monthly.name = output_name
        monthly_series[output_name] = monthly

    # --- Construir DataFrame consolidado ---
    if not monthly_series:
        logger.error("No se procesó ningún archivo de mercado.")
        return pd.DataFrame()

    # Merge con outer join para no perder meses de series más largas (VIX)
    market_df = pd.DataFrame(monthly_series)
    market_df.index.name = "date"
    market_df = market_df.sort_index()

    logger.info(f"\n{'─' * 50}")
    logger.info(f"MARKET_MONTHLY: {market_df.shape[0]} meses × {market_df.shape[1]} series")
    logger.info(f"Rango: {market_df.index.min().date()} → {market_df.index.max().date()}")
    logger.info(f"NaNs totales por columna:")
    for col in market_df.columns:
        n_nan = market_df[col].isna().sum()
        n_valid = market_df[col].notna().sum()
        first_valid = market_df[col].first_valid_index()
        logger.info(
            f"    {col:>6s}: {n_valid:>4d} válidos, {n_nan:>4d} NaNs | "
            f"primer dato: {first_valid.date() if first_valid is not None else 'N/A'}"
        )

    return market_df


# ══════════════════════════════════════════════════════════════════════════════
# 6. PROCESAMIENTO DE DATOS MACROECONÓMICOS
# ══════════════════════════════════════════════════════════════════════════════

def process_macro_data(
    raw_dir: Path = RAW_DATA_DIR,
    fred_files: dict = None,
) -> pd.DataFrame:
    """
    Procesa todos los archivos FRED a frecuencia mensual uniforme.

    Para cada serie:
    1. Carga el archivo raw y normaliza el índice.
    2. Si es diaria → resamplea a mensual con el método configurado.
    3. Si ya es mensual → normaliza el índice al fin de mes.
    4. Documenta pérdida de datos y cobertura.

    Parámetros
    ----------
    raw_dir : Path
        Directorio de datos raw.
    fred_files : dict
        Mapeo de configuración. Por defecto FRED_FILES.

    Retorna
    -------
    pd.DataFrame : datos macroeconómicos mensuales (columnas = series FRED).
    """
    if fred_files is None:
        fred_files = FRED_FILES

    logger.info("=" * 70)
    logger.info("PROCESAMIENTO DE DATOS MACROECONÓMICOS → MENSUAL")
    logger.info("=" * 70)

    monthly_series = {}

    for filename, config in fred_files.items():
        filepath = raw_dir / f"{filename}.csv"
        output_name = config["output_name"]
        native_freq = config["native_freq"]
        method = config["resample_method"]

        logger.info(f"\nProcesando {output_name} ({filename}.csv)...")

        # --- Cargar y normalizar ---
        if not filepath.exists():
            logger.warning(f"  ⚠ Archivo no encontrado: {filepath}. Saltando.")
            continue

        df = load_and_normalize_index(filepath)

        # Las series FRED individuales tienen una sola columna de datos.
        # Tomamos la primera columna numérica (ignorando el índice).
        if df.shape[1] == 0:
            logger.warning(f"  ⚠ DataFrame vacío para {filename}. Saltando.")
            continue

        # Usar la primera (y típicamente única) columna
        raw_series = df.iloc[:, 0].copy()
        n_raw = len(raw_series)
        n_raw_nans = raw_series.isna().sum()

        logger.info(
            f"    Raw: {n_raw} obs ({n_raw_nans} NaNs) | "
            f"{raw_series.index.min().date()} → {raw_series.index.max().date()} | "
            f"freq nativa: {native_freq}"
        )

        # --- Procesar según frecuencia nativa ---
        if native_freq == "daily" and method is not None:
            # Serie diaria → resamplear a mensual
            monthly = resample_to_monthly(raw_series, method=method)
            logger.info(
                f"    Resampleado ({method}): {len(monthly)} meses "
                f"({monthly.isna().sum()} NaNs)"
            )
        elif native_freq == "monthly":
            # Serie ya mensual → solo normalizar índice al fin de mes
            monthly = normalize_monthly_index(raw_series)
            logger.info(
                f"    Índice normalizado a fin de mes: {len(monthly)} meses "
                f"({monthly.isna().sum()} NaNs)"
            )
        else:
            logger.warning(
                f"  ⚠ Configuración inesperada para {output_name}: "
                f"freq={native_freq}, method={method}. Saltando."
            )
            continue

        monthly.name = output_name
        monthly_series[output_name] = monthly

    # --- Construir DataFrame consolidado ---
    if not monthly_series:
        logger.error("No se procesó ningún archivo FRED.")
        return pd.DataFrame()

    macro_df = pd.DataFrame(monthly_series)
    macro_df.index.name = "date"
    macro_df = macro_df.sort_index()

    logger.info(f"\n{'─' * 50}")
    logger.info(f"MACRO_MONTHLY: {macro_df.shape[0]} meses × {macro_df.shape[1]} series")
    logger.info(f"Rango: {macro_df.index.min().date()} → {macro_df.index.max().date()}")
    logger.info(f"NaNs totales por columna:")
    for col in macro_df.columns:
        n_nan = macro_df[col].isna().sum()
        n_valid = macro_df[col].notna().sum()
        first_valid = macro_df[col].first_valid_index()
        logger.info(
            f"    {col:>10s}: {n_valid:>5d} válidos, {n_nan:>5d} NaNs | "
            f"primer dato: {first_valid.date() if first_valid is not None else 'N/A'}"
        )

    # --- Advertencia sobre USREC ---
    if "USREC" in macro_df.columns:
        logger.info(
            "\n    ⚠ NOTA: USREC (recesiones NBER) tiene retraso de publicación "
            "significativo. NO usar como señal prospectiva sin ajustar lag."
        )

    return macro_df


# ══════════════════════════════════════════════════════════════════════════════
# 7. ALINEACIÓN Y MERGE FINAL
# ══════════════════════════════════════════════════════════════════════════════

def combine_datasets(
    market_df: pd.DataFrame,
    macro_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Combina datos de mercado y macroeconómicos en un único DataFrame.

    Se usa un OUTER JOIN sobre el índice temporal para no perder
    observaciones de ninguna de las dos fuentes. Esto significa que
    habrá NaNs en series más cortas — lo cual es esperado y deseable
    en esta fase.

    NO se aplica forward-fill ni interpolación. Los NaNs quedan como
    indicadores explícitos de ausencia de datos.

    Prefijos en las columnas:
    - MKT_ → datos de mercado (yfinance)
    - MAC_ → datos macroeconómicos (FRED)

    Estos prefijos facilitan la selección de columnas en fases posteriores
    y mantienen la trazabilidad de la fuente.

    Parámetros
    ----------
    market_df : pd.DataFrame
        Datos de mercado mensuales.
    macro_df : pd.DataFrame
        Datos macroeconómicos mensuales.

    Retorna
    -------
    pd.DataFrame : dataset combinado con prefijos de fuente.
    """
    logger.info("=" * 70)
    logger.info("ALINEACIÓN Y MERGE FINAL")
    logger.info("=" * 70)

    # --- Añadir prefijos de fuente ---
    market_prefixed = market_df.add_prefix("MKT_")
    macro_prefixed = macro_df.add_prefix("MAC_")

    # --- Merge con outer join ---
    combined = market_prefixed.join(macro_prefixed, how="outer")
    combined.index.name = "date"
    combined = combined.sort_index()

    logger.info(f"Dataset combinado: {combined.shape[0]} meses × {combined.shape[1]} columnas")
    logger.info(f"Rango total: {combined.index.min().date()} → {combined.index.max().date()}")

    # --- Informe de cobertura cruzada ---
    logger.info("\nCobertura por columna:")
    for col in combined.columns:
        n_valid = combined[col].notna().sum()
        n_total = len(combined)
        pct = 100 * n_valid / n_total
        first = combined[col].first_valid_index()
        last = combined[col].last_valid_index()
        logger.info(
            f"    {col:<20s}: {n_valid:>4d}/{n_total} ({pct:5.1f}%) | "
            f"{first.date() if first else 'N/A'} → {last.date() if last else 'N/A'}"
        )

    return combined


# ══════════════════════════════════════════════════════════════════════════════
# 8. PERSISTENCIA — GUARDAR DATASETS PROCESADOS
# ══════════════════════════════════════════════════════════════════════════════

def save_processed_datasets(
    market_df: pd.DataFrame,
    macro_df: pd.DataFrame,
    combined_df: pd.DataFrame,
    audit_df: pd.DataFrame,
    output_dir: Path = PROCESSED_DATA_DIR,
) -> dict[str, Path]:
    """
    Guarda todos los datasets procesados en disco.

    Archivos generados:
    - market_monthly.csv   → solo datos de mercado (sin prefijo)
    - macro_monthly.csv    → solo datos macro (sin prefijo)
    - combined_monthly_raw.csv → ambos combinados (con prefijos MKT_/MAC_)
    - audit_report.csv     → informe de auditoría de la fase raw

    Parámetros
    ----------
    market_df : pd.DataFrame
    macro_df : pd.DataFrame
    combined_df : pd.DataFrame
    audit_df : pd.DataFrame
    output_dir : Path

    Retorna
    -------
    dict[str, Path] : mapeo nombre → ruta del archivo guardado.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_files = {}

    files_to_save = {
        "market_monthly": market_df,
        "macro_monthly": macro_df,
        "combined_monthly_raw": combined_df,
        "audit_report": audit_df,
    }

    logger.info("=" * 70)
    logger.info(f"GUARDANDO DATASETS PROCESADOS → {output_dir}")
    logger.info("=" * 70)

    for name, df in files_to_save.items():
        filepath = output_dir / f"{name}.csv"
        df.to_csv(filepath)
        saved_files[name] = filepath
        logger.info(
            f"  ✓ {filepath.name:<30s} | "
            f"{df.shape[0]:>5d} filas × {df.shape[1]:>3d} cols"
        )

    return saved_files


# ══════════════════════════════════════════════════════════════════════════════
# 9. FUNCIÓN PRINCIPAL — ORQUESTADOR DEL PASO 2
# ══════════════════════════════════════════════════════════════════════════════

def run_preprocessing(
    raw_dir: Path = RAW_DATA_DIR,
    output_dir: Path = PROCESSED_DATA_DIR,
) -> dict:
    """
    Ejecuta el pipeline completo de preprocesado (Paso 2).

    Secuencia:
    1. Auditoría de todos los archivos raw.
    2. Procesamiento de datos de mercado → mensual.
    3. Procesamiento de datos macro → mensual.
    4. Alineación y merge en dataset combinado.
    5. Persistencia de todos los datasets.

    Parámetros
    ----------
    raw_dir : Path
        Directorio con datos raw (salida del data_loader.py).
    output_dir : Path
        Directorio para datasets procesados.

    Retorna
    -------
    dict : con claves 'market', 'macro', 'combined', 'audit', 'files'.
    """
    logger.info("╔══════════════════════════════════════════════════════════════╗")
    logger.info("║   PREPROCESSING — Paso 2: Validación y alineación          ║")
    logger.info("║   Estrategia de asignación dinámica S&P 500                ║")
    logger.info("╚══════════════════════════════════════════════════════════════╝")
    logger.info(f"Entrada: {raw_dir}")
    logger.info(f"Salida:  {output_dir}")
    logger.info("")

    # --- Paso 2.1: Auditoría ---
    audit_df = run_full_audit(raw_dir=raw_dir)

    # --- Paso 2.2: Procesar mercado ---
    market_df = process_market_data(raw_dir=raw_dir)

    # --- Paso 2.3: Procesar macro ---
    macro_df = process_macro_data(raw_dir=raw_dir)

    # --- Paso 2.4: Combinar ---
    combined_df = combine_datasets(market_df, macro_df)

    # --- Paso 2.5: Guardar ---
    saved_files = save_processed_datasets(
        market_df=market_df,
        macro_df=macro_df,
        combined_df=combined_df,
        audit_df=audit_df,
        output_dir=output_dir,
    )

    # --- Resumen final ---
    logger.info("")
    logger.info("=" * 70)
    logger.info("RESUMEN DEL PASO 2")
    logger.info("=" * 70)
    logger.info(f"  Mercado:   {market_df.shape[0]} meses × {market_df.shape[1]} series")
    logger.info(f"  Macro:     {macro_df.shape[0]} meses × {macro_df.shape[1]} series")
    logger.info(f"  Combinado: {combined_df.shape[0]} meses × {combined_df.shape[1]} columnas")
    logger.info(f"  Archivos guardados en: {output_dir.resolve()}")
    logger.info("=" * 70)

    return {
        "market": market_df,
        "macro": macro_df,
        "combined": combined_df,
        "audit": audit_df,
        "files": saved_files,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 10. EJECUCIÓN DIRECTA (CLI)
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    """
    Ejecución desde línea de comandos.

    Uso:
        python preprocessing.py

    Requisitos previos:
        - Haber ejecutado data_loader.py (los archivos raw deben existir
          en data/raw/).
        - pip install pandas pyarrow
    """
    results = run_preprocessing()