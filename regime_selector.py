"""
================================================================================
REGIME SELECTOR — regime_selector.py
================================================================================
Proyecto: Análisis de contexto económico-financiero para asignación dinámica
          en el S&P 500.

Propósito: Módulo orquestador que permite seleccionar y ejecutar cualquiera
           de los modelos de régimen disponibles de forma cómoda y extensible.

Uso básico:
    from regime_selector import get_regime

    # Seleccionar modelo por nombre
    regime = get_regime(model="macro")
    regime = get_regime(model="financial")
    regime = get_regime(model="liquidity")

    # Con DataFrame de indicadores propio
    regime = get_regime(model="macro", indicators=my_indicators_df)

Cómo añadir un nuevo modelo:
    1. Crear un archivo regime_model_<nombre>.py con la función:
           def get_regime_series(indicators: pd.DataFrame) -> pd.Series
    2. Añadir una entrada en MODEL_REGISTRY (abajo) con:
           "<nombre>": {
               "module": "regime_model_<nombre>",
               "description": "Descripción del modelo",
           }
    3. El nuevo modelo estará disponible automáticamente como:
           get_regime(model="<nombre>")

Autor: Mauro Calvo Pérez y Jorge Fernández Beloso
Fecha: 2026-02
================================================================================
"""

import importlib
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# 1. CONFIGURACIÓN
# ══════════════════════════════════════════════════════════════════════════════

INDICATORS_DIR = Path("data/indicators")
REGIMES_DIR = Path("data/processed/regimes")
INDICATORS_FILE = "indicators_full.csv"

# ──────────────────────────────────────────────────────────────────────────────
# REGISTRO DE MODELOS DISPONIBLES
#
# Cada entrada mapea un nombre corto a su módulo Python.
# Para añadir un modelo nuevo, solo hay que añadir una línea aquí
# y crear el archivo correspondiente con la interfaz estándar:
#     def get_regime_series(indicators: pd.DataFrame) -> pd.Series
# ──────────────────────────────────────────────────────────────────────────────
MODEL_REGISTRY: dict[str, dict] = {
    "macro": {
        "module": "regime_model_macro",
        "description": (
            "Régimen macroeconómico — ciclo real (producción industrial, "
            "empleo, curva de tipos). Regímenes: expansion / neutral / contraction."
        ),
    },
    "financial": {
        "module": "regime_model_financial",
        "description": (
            "Régimen de condiciones financieras — volatilidad, crédito, "
            "estrés de mercado. Regímenes: risk_on / neutral / risk_off."
        ),
    },
    "liquidity": {
        "module": "regime_model_liquidity",
        "description": (
            "Régimen de liquidez/política monetaria — tipos de interés, "
            "tipo real, inflación. Regímenes: accommodative / neutral / restrictive."
        ),
    },
}


# ══════════════════════════════════════════════════════════════════════════════
# 2. CARGA DE INDICADORES
# ══════════════════════════════════════════════════════════════════════════════

def load_indicators(
    indicators_dir: Path = INDICATORS_DIR,
    filename: str = INDICATORS_FILE,
) -> pd.DataFrame:
    """
    Carga el dataset completo de indicadores.

    Lee indicators_full.csv generado por indicators.py (Paso 3).

    Parámetros
    ----------
    indicators_dir : Path
        Directorio donde está el archivo.
    filename : str
        Nombre del archivo.

    Retorna
    -------
    pd.DataFrame : indicadores con DatetimeIndex.
    """
    filepath = indicators_dir / filename
    if not filepath.exists():
        raise FileNotFoundError(
            f"No se encontró {filepath}. "
            f"¿Se ejecutó indicators.py (Paso 3)?"
        )

    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    logger.info(
        f"Indicadores cargados: {df.shape[0]} meses × {df.shape[1]} indicadores "
        f"desde {filepath}"
    )
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 3. FUNCIÓN PRINCIPAL: get_regime
# ══════════════════════════════════════════════════════════════════════════════

def get_regime(
    model: str,
    indicators: Optional[pd.DataFrame] = None,
    indicators_dir: Path = INDICATORS_DIR,
    save: bool = False,
    output_dir: Path = REGIMES_DIR,
) -> pd.Series:
    """
    Selecciona y ejecuta un modelo de régimen.

    Esta es la función principal del módulo. Carga los indicadores
    (si no se proporcionan), importa dinámicamente el modelo seleccionado,
    y devuelve la serie temporal de régimen.

    Parámetros
    ----------
    model : str
        Nombre del modelo a ejecutar. Opciones disponibles:
        - "macro"     → Régimen macroeconómico
        - "financial" → Régimen de condiciones financieras
        - "liquidity" → Régimen de liquidez/política monetaria

    indicators : pd.DataFrame, opcional
        DataFrame de indicadores. Si no se proporciona, se carga
        automáticamente desde indicators_full.csv.

    indicators_dir : Path
        Directorio de indicadores (solo si indicators=None).

    save : bool
        Si True, guarda la serie de régimen en data/processed/regimes/.

    output_dir : Path
        Directorio de salida para guardar (solo si save=True).

    Retorna
    -------
    pd.Series : serie temporal de régimen (int: -1, 0, 1).

    Raises
    ------
    ValueError : si el modelo no existe en el registro.
    ImportError : si el módulo del modelo no se puede importar.

    Ejemplo
    -------
    >>> from regime_selector import get_regime
    >>> regime = get_regime(model="macro")
    >>> print(regime.tail())
    """
    # --- Validar modelo ---
    model = model.lower().strip()

    if model not in MODEL_REGISTRY:
        available = list(MODEL_REGISTRY.keys())
        raise ValueError(
            f"Modelo '{model}' no encontrado. "
            f"Modelos disponibles: {available}. "
            f"Para añadir uno nuevo, actualiza MODEL_REGISTRY en regime_selector.py."
        )

    model_info = MODEL_REGISTRY[model]
    module_name = model_info["module"]

    logger.info("=" * 70)
    logger.info(f"REGIME SELECTOR — Modelo seleccionado: '{model}'")
    logger.info(f"  Módulo: {module_name}")
    logger.info(f"  {model_info['description']}")
    logger.info("=" * 70)

    # --- Cargar indicadores si no se proporcionan ---
    if indicators is None:
        indicators = load_indicators(indicators_dir)

    # --- Importar y ejecutar el modelo ---
    try:
        regime_module = importlib.import_module(module_name)
    except ImportError as e:
        raise ImportError(
            f"No se pudo importar el módulo '{module_name}'. "
            f"¿Existe el archivo {module_name}.py? Error: {e}"
        )

    # Verificar que el módulo tiene la interfaz estándar
    if not hasattr(regime_module, "get_regime_series"):
        raise AttributeError(
            f"El módulo '{module_name}' no implementa la función "
            f"'get_regime_series(indicators)'. Todos los modelos de régimen "
            f"deben implementar esta interfaz."
        )

    # Ejecutar el modelo
    regime_series = regime_module.get_regime_series(indicators)

    logger.info(f"\nRégimen '{model}' calculado: {regime_series.notna().sum()} meses válidos")

    # --- Guardar opcionalmente ---
    if save:
        _save_regime(regime_series, model, output_dir)

    return regime_series


# ══════════════════════════════════════════════════════════════════════════════
# 4. FUNCIONES AUXILIARES
# ══════════════════════════════════════════════════════════════════════════════

def _save_regime(
    regime: pd.Series,
    model_name: str,
    output_dir: Path = REGIMES_DIR,
) -> Path:
    """
    Guarda la serie de régimen en disco.

    Archivo generado: regime_{model_name}.csv

    Parámetros
    ----------
    regime : pd.Series
        Serie de régimen.
    model_name : str
        Nombre del modelo (para el nombre del archivo).
    output_dir : Path
        Directorio de salida.

    Retorna
    -------
    Path : ruta del archivo guardado.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    filepath = output_dir / f"regime_{model_name}.csv"

    regime_df = regime.to_frame(name=f"regime_{model_name}")
    regime_df.index.name = "date"
    regime_df.to_csv(filepath)

    logger.info(f"  ✓ Régimen guardado: {filepath}")
    return filepath


def list_models() -> dict[str, str]:
    """
    Lista todos los modelos de régimen disponibles.

    Retorna
    -------
    dict : {nombre: descripción} de cada modelo registrado.
    """
    return {name: info["description"] for name, info in MODEL_REGISTRY.items()}


def get_all_regimes(
    indicators: Optional[pd.DataFrame] = None,
    save: bool = False,
) -> pd.DataFrame:
    """
    Ejecuta TODOS los modelos de régimen y devuelve un DataFrame
    consolidado con una columna por modelo.

    Útil para comparación y diagnóstico, pero NO para combinar
    modelos en un ensemble (eso corresponde a fases posteriores).

    Parámetros
    ----------
    indicators : pd.DataFrame, opcional
        Si no se proporciona, se carga automáticamente.
    save : bool
        Si True, guarda cada régimen individual.

    Retorna
    -------
    pd.DataFrame : columnas = regime_macro, regime_financial, regime_liquidity.
    """
    logger.info("=" * 70)
    logger.info("EJECUTANDO TODOS LOS MODELOS DE RÉGIMEN")
    logger.info("=" * 70)

    if indicators is None:
        indicators = load_indicators()

    all_regimes = pd.DataFrame(index=indicators.index)

    for model_name in MODEL_REGISTRY:
        try:
            regime = get_regime(
                model=model_name,
                indicators=indicators,
                save=save,
            )
            all_regimes[f"regime_{model_name}"] = regime
        except Exception as e:
            logger.error(f"Error en modelo '{model_name}': {e}")
            continue

    all_regimes.index.name = "date"

    # --- Resumen de concordancia ---
    valid_rows = all_regimes.dropna()
    if len(valid_rows) > 0 and all_regimes.shape[1] > 1:
        logger.info(f"\nConcordancia entre modelos ({len(valid_rows)} meses con datos completos):")
        for col_a in all_regimes.columns:
            for col_b in all_regimes.columns:
                if col_a < col_b:
                    agreement = (valid_rows[col_a] == valid_rows[col_b]).mean()
                    logger.info(f"  {col_a} vs {col_b}: {agreement:.1%} concordancia")

    return all_regimes


# ══════════════════════════════════════════════════════════════════════════════
# 5. EJECUCIÓN DIRECTA
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    """
    Ejecución desde línea de comandos.

    Uso:
        # Ejecutar un modelo específico
        python regime_selector.py

    Requisitos previos:
        - Haber ejecutado indicators.py (Paso 3).
        - indicators_full.csv debe existir en data/indicators/.
    """
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Si se pasa un argumento, usarlo como modelo
    if len(sys.argv) > 1:
        model_name = sys.argv[1]
        regime = get_regime(model=model_name, save=True)
    else:
        # Por defecto, ejecutar todos
        all_regimes = get_all_regimes(save=True)

        # Guardar consolidado
        REGIMES_DIR.mkdir(parents=True, exist_ok=True)
        all_regimes.to_csv(REGIMES_DIR / "regimes_all.csv")
        logger.info(f"\n✓ Todos los regímenes guardados en {REGIMES_DIR}/")