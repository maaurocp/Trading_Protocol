"""
================================================================================
MODEL FACTORY — model_factory.py
================================================================================
Proyecto: Análisis de contexto económico-financiero para asignación dinámica
          en el S&P 500.

Propósito: Crear modelos de decisión táctica de forma controlada, validando
           que los indicadores solicitados existen en el dataset y que los
           parámetros son coherentes con el tipo de lógica elegido.

Uso:
    from model_factory import create_model

    model = create_model(
        name="macro_momentum_v1",
        indicators=["trend_momentum_6m", "cycle_indpro_yoy", "vol_vix_zscore_24m"],
        logic="zscore_composite",
        parameters={
            "directions": {
                "trend_momentum_6m": +1,
                "cycle_indpro_yoy": +1,
                "vol_vix_zscore_24m": -1,
            },
            "threshold_buy": 0.5,
            "threshold_sell": -0.5,
            "min_periods": 24,
        },
        description="Modelo táctico basado en momentum, ciclo y volatilidad",
    )

    model.save_model()

Autor: Mauro Calvo Pérez y Jorge Fernández Beloso
Fecha: 2026-02
================================================================================
"""

import logging
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from model_base import BaseModel, LOGIC_REGISTRY, MODELS_DIR, get_available_logics

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ══════════════════════════════════════════════════════════════════════════════

INDICATORS_DIR = Path("data/indicators")
INDICATORS_FILE = "indicators_full.csv"


# ══════════════════════════════════════════════════════════════════════════════
# VALIDACIÓN DE INDICADORES
# ══════════════════════════════════════════════════════════════════════════════

def _load_available_indicators(
    indicators_dir: Path = INDICATORS_DIR,
    filename: str = INDICATORS_FILE,
) -> list[str]:
    """
    Carga la lista de indicadores disponibles desde el archivo.

    Lee solo las cabeceras (no todo el dataset) para eficiencia.

    Retorna
    -------
    list[str] : nombres de columna disponibles.
    """
    filepath = indicators_dir / filename
    if not filepath.exists():
        logger.warning(
            f"Archivo de indicadores no encontrado: {filepath}. "
            f"La validación de indicadores se omitirá."
        )
        return []

    # Leer solo la primera fila para obtener columnas
    df_header = pd.read_csv(filepath, index_col=0, nrows=0)
    return list(df_header.columns)


def _validate_indicators(
    requested: list[str],
    available: list[str],
) -> tuple[list[str], list[str]]:
    """
    Valida que los indicadores solicitados existen.

    Retorna
    -------
    tuple : (válidos, no encontrados)
    """
    valid = [ind for ind in requested if ind in available]
    missing = [ind for ind in requested if ind not in available]
    return valid, missing


# ══════════════════════════════════════════════════════════════════════════════
# FUNCIÓN PRINCIPAL: create_model
# ══════════════════════════════════════════════════════════════════════════════

def create_model(
    name: str,
    indicators: list[str],
    logic: str,
    parameters: dict[str, Any],
    description: str = "",
    validate_indicators: bool = True,
    save: bool = False,
    models_dir: Path = MODELS_DIR,
) -> BaseModel:
    """
    Crea un nuevo modelo de decisión táctica.

    Esta es la función principal del factory. Valida los inputs,
    selecciona la clase de lógica correspondiente e instancia el modelo.

    Parámetros
    ----------
    name : str
        Nombre único del modelo (sin espacios).

    indicators : list[str]
        Lista de indicadores a utilizar. Deben existir en indicators_full.csv.
        Ejemplo: ["trend_momentum_6m", "vol_vix_zscore_24m"]

    logic : str
        Tipo de lógica de decisión. Opciones disponibles:
        - "zscore_composite"   → Composite z-score con dirección
        - "threshold_rules"    → Reglas con umbrales fijos por indicador
        - "weighted_composite" → Composite ponderado con pesos

    parameters : dict
        Parámetros de la lógica. Cada tipo tiene sus propios requisitos:

        zscore_composite:
            {
                "directions": {"ind1": +1, "ind2": -1, ...},
                "threshold_buy": 0.5,
                "threshold_sell": -0.5,
                "min_periods": 24  (opcional, default 24)
            }

        threshold_rules:
            {
                "thresholds": {
                    "ind1": {"bullish": 0.05, "bearish": -0.05},
                    ...
                }
            }

        weighted_composite:
            {
                "weights": {"ind1": 2.0, "ind2": 1.0, ...},
                "directions": {"ind1": +1, "ind2": -1, ...},
                "threshold_buy": 0.5,
                "threshold_sell": -0.5,
                "min_periods": 24  (opcional)
            }

    description : str
        Descripción legible del modelo.

    validate_indicators : bool
        Si True, verifica que los indicadores existen en indicators_full.csv.
        Útil desactivarlo en tests con datos sintéticos.

    save : bool
        Si True, guarda el modelo automáticamente tras crearlo.

    models_dir : Path
        Directorio donde guardar (si save=True).

    Retorna
    -------
    BaseModel : instancia del modelo creado (subtipo según lógica).

    Raises
    ------
    ValueError : si la lógica no existe o los indicadores no son válidos.
    """
    logger.info("=" * 70)
    logger.info(f"MODEL FACTORY — Creando modelo '{name}'")
    logger.info("=" * 70)

    # --- 1. Validar tipo de lógica ---
    logic = logic.lower().strip()
    if logic not in LOGIC_REGISTRY:
        raise ValueError(
            f"Tipo de lógica '{logic}' no reconocido. "
            f"Opciones disponibles: {get_available_logics()}"
        )

    logger.info(f"  Lógica: {logic}")
    logger.info(f"  Indicadores solicitados: {indicators}")

    # --- 2. Validar indicadores ---
    if validate_indicators:
        available = _load_available_indicators()
        if available:
            valid, missing = _validate_indicators(indicators, available)
            if missing:
                raise ValueError(
                    f"Indicadores no encontrados en indicators_full.csv: {missing}. "
                    f"Verifica los nombres. Indicadores disponibles incluyen: "
                    f"{available[:10]}..."
                )
            logger.info(f"  ✓ Todos los indicadores validados contra indicators_full.csv")
        else:
            logger.info(
                f"  ⚠ Validación de indicadores omitida "
                f"(indicators_full.csv no encontrado)"
            )

    # --- 3. Instanciar modelo ---
    ModelClass = LOGIC_REGISTRY[logic]

    try:
        model = ModelClass(
            name=name,
            indicators=indicators,
            parameters=parameters,
            description=description,
        )
    except (ValueError, KeyError) as e:
        raise ValueError(f"Error creando modelo '{name}': {e}")

    logger.info(f"  ✓ Modelo '{name}' creado exitosamente")
    logger.info(f"  Tipo: {model.__class__.__name__}")
    logger.info(f"  Repr: {model}")

    # --- 4. Guardar si solicitado ---
    if save:
        model.save_model(models_dir)

    return model


# ══════════════════════════════════════════════════════════════════════════════
# FUNCIONES AUXILIARES
# ══════════════════════════════════════════════════════════════════════════════

def list_available_indicators(
    indicators_dir: Path = INDICATORS_DIR,
) -> list[str]:
    """
    Lista todos los indicadores disponibles en indicators_full.csv.

    Útil para explorar qué indicadores se pueden usar al crear un modelo.
    """
    available = _load_available_indicators(indicators_dir)
    if not available:
        logger.warning("No se encontró indicators_full.csv.")
    return available


def list_saved_models(models_dir: Path = MODELS_DIR) -> list[str]:
    """
    Lista los nombres de modelos guardados en disco.

    Retorna
    -------
    list[str] : nombres de modelos (sin extensión .json).
    """
    if not models_dir.exists():
        return []
    return sorted(p.stem for p in models_dir.glob("*.json"))