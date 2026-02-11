"""
================================================================================
MODEL LOADER — model_loader.py
================================================================================
Proyecto: Análisis de contexto económico-financiero para asignación dinámica
          en el S&P 500.

Propósito: Cargar modelos de decisión táctica previamente guardados como JSON
           y devolver instancias listas para generar señales.

Uso:
    from model_loader import load_model, list_models

    # Ver modelos disponibles
    models = list_models()
    print(models)  # ['macro_momentum_v1', 'risk_composite_v2', ...]

    # Cargar un modelo
    model = load_model("macro_momentum_v1")

    # Generar señal
    signals = model.generate_signal(indicators_df)

Autor: Mauro Calvo Pérez y Jorge Fernández Beloso
Fecha: 2026-02
================================================================================
"""

import json
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from model_base import BaseModel, LOGIC_REGISTRY, MODELS_DIR

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# FUNCIÓN PRINCIPAL: load_model
# ══════════════════════════════════════════════════════════════════════════════

def load_model(
    name: str,
    models_dir: Path = MODELS_DIR,
) -> BaseModel:
    """
    Carga un modelo guardado desde su archivo JSON y devuelve una
    instancia lista para usar.

    El proceso:
    1. Lee el archivo {name}.json del directorio de modelos.
    2. Extrae los metadatos: indicadores, lógica, parámetros.
    3. Busca la clase correspondiente al logic_type en LOGIC_REGISTRY.
    4. Instancia el modelo con los parámetros guardados.

    Parámetros
    ----------
    name : str
        Nombre del modelo (sin extensión .json).
    models_dir : Path
        Directorio donde buscar el archivo.

    Retorna
    -------
    BaseModel : instancia del modelo (subtipo según lógica).

    Raises
    ------
    FileNotFoundError : si el archivo no existe.
    ValueError : si el tipo de lógica no está registrado.
    """
    filepath = models_dir / f"{name}.json"

    if not filepath.exists():
        available = list_models(models_dir)
        raise FileNotFoundError(
            f"Modelo '{name}' no encontrado en {models_dir}. "
            f"Modelos disponibles: {available}"
        )

    # --- Leer JSON ---
    with open(filepath, "r", encoding="utf-8") as f:
        model_dict = json.load(f)

    logger.info(f"Cargando modelo '{name}' desde {filepath}")

    # --- Validar campos requeridos ---
    required_fields = ["name", "indicators", "logic_type", "parameters"]
    for field in required_fields:
        if field not in model_dict:
            raise ValueError(
                f"Archivo de modelo corrupto: falta campo '{field}' en {filepath}"
            )

    # --- Buscar clase de lógica ---
    logic_type = model_dict["logic_type"]
    if logic_type not in LOGIC_REGISTRY:
        raise ValueError(
            f"Tipo de lógica '{logic_type}' no registrado. "
            f"Registrados: {list(LOGIC_REGISTRY.keys())}. "
            f"¿Se añadió una nueva lógica sin registrarla en model_base.py?"
        )

    ModelClass = LOGIC_REGISTRY[logic_type]

    # --- Instanciar modelo ---
    model = ModelClass(
        name=model_dict["name"],
        indicators=model_dict["indicators"],
        parameters=model_dict["parameters"],
        description=model_dict.get("description", ""),
    )

    # Restaurar fecha de creación original
    if "created_at" in model_dict:
        model.created_at = model_dict["created_at"]

    logger.info(
        f"  ✓ Modelo '{name}' cargado | "
        f"lógica={logic_type} | "
        f"{len(model.indicators)} indicadores | "
        f"creado={model.created_at}"
    )

    return model


# ══════════════════════════════════════════════════════════════════════════════
# FUNCIONES AUXILIARES
# ══════════════════════════════════════════════════════════════════════════════

def list_models(models_dir: Path = MODELS_DIR) -> list[str]:
    """
    Lista todos los modelos guardados en disco.

    Retorna
    -------
    list[str] : nombres de modelos disponibles (sin extensión).
    """
    if not models_dir.exists():
        return []
    return sorted(p.stem for p in models_dir.glob("*.json"))


def inspect_model(
    name: str,
    models_dir: Path = MODELS_DIR,
) -> dict:
    """
    Lee los metadatos de un modelo SIN instanciarlo.

    Útil para explorar modelos guardados sin cargar dependencias.

    Parámetros
    ----------
    name : str
        Nombre del modelo.
    models_dir : Path
        Directorio de modelos.

    Retorna
    -------
    dict : contenido completo del JSON del modelo.
    """
    filepath = models_dir / f"{name}.json"

    if not filepath.exists():
        raise FileNotFoundError(f"Modelo '{name}' no encontrado en {models_dir}")

    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def load_all_models(
    models_dir: Path = MODELS_DIR,
) -> dict[str, BaseModel]:
    """
    Carga todos los modelos guardados en disco.

    Retorna
    -------
    dict[str, BaseModel] : {nombre: instancia} de cada modelo.
    """
    names = list_models(models_dir)
    models = {}

    for name in names:
        try:
            models[name] = load_model(name, models_dir)
        except Exception as e:
            logger.error(f"Error cargando modelo '{name}': {e}")
            continue

    logger.info(f"Cargados {len(models)}/{len(names)} modelos desde {models_dir}")
    return models