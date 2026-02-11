"""
================================================================================
MODEL BASE — model_base.py
================================================================================
Proyecto: Análisis de contexto económico-financiero para asignación dinámica
          en el S&P 500.

Propósito: Definir la clase abstracta base para todos los modelos de decisión
           táctica. Establece la interfaz obligatoria, la validación de
           indicadores y la persistencia (serialización a JSON).

Filosofía de diseño:

    Cada modelo táctico es una combinación de tres elementos:
    1. INDICADORES: qué variables del contexto mira (subconjunto explícito).
    2. LÓGICA: cómo transforma esos indicadores en una señal (-1, 0, +1).
    3. PARÁMETROS: configuración numérica de la lógica (umbrales, ventanas).

    La clase BaseModel encapsula estos tres elementos y obliga a todas las
    implementaciones concretas a respetar la misma interfaz. Esto permite:
    - Crear modelos distintos que son intercambiables.
    - Serializar/deserializar modelos de forma uniforme.
    - Validar que los indicadores requeridos existen antes de ejecutar.

Señal de salida:
    -1 = reducir exposición (defensivo)
     0 = mantener (neutral)
    +1 = aumentar exposición (agresivo)

    El modelo opera como si controlara el 100% de su capital gestionado.
    El hecho de que esto represente solo el 30% táctico del total es
    responsabilidad de la capa de asignación (fases posteriores).

Separación de responsabilidades:
    - BaseModel NO conoce los modelos de régimen.
    - BaseModel NO conoce la asignación del 70/30.
    - BaseModel NO hace backtesting ni evalúa performance.
    - BaseModel SOLO produce una señal táctica basada en indicadores.

Autor: Mauro Calvo Pérez y Jorge Fernández Beloso
Fecha: 2026-02
================================================================================
"""

import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ══════════════════════════════════════════════════════════════════════════════

MODELS_DIR = Path("models")


# ══════════════════════════════════════════════════════════════════════════════
# CLASE ABSTRACTA BASE
# ══════════════════════════════════════════════════════════════════════════════

class BaseModel(ABC):
    """
    Clase abstracta base para modelos de decisión táctica.

    Todos los modelos de decisión deben heredar de esta clase e
    implementar el método `_compute_signal()`.

    Atributos
    ---------
    name : str
        Nombre único del modelo (usado como identificador y nombre de archivo).
    indicators : list[str]
        Lista explícita de indicadores que el modelo utiliza.
    logic_type : str
        Identificador del tipo de lógica (ej: "zscore_composite").
    parameters : dict
        Parámetros configurables de la lógica (umbrales, ventanas, etc.).
    description : str
        Descripción legible del modelo.
    created_at : str
        Fecha de creación (ISO format).
    """

    def __init__(
        self,
        name: str,
        indicators: list[str],
        logic_type: str,
        parameters: dict[str, Any],
        description: str = "",
    ):
        """
        Inicializa el modelo base.

        Parámetros
        ----------
        name : str
            Nombre único. Se usa como identificador y nombre de archivo.
            No debe contener espacios ni caracteres especiales.
        indicators : list[str]
            Lista de nombres de columna del indicators_full.csv que este
            modelo necesita. El modelo NO funcionará si alguno falta.
        logic_type : str
            Tipo de lógica de decisión (ej: "zscore_composite",
            "threshold_rules", "weighted_composite").
        parameters : dict
            Parámetros numéricos de la lógica. Cada tipo de lógica
            define qué parámetros espera.
        description : str
            Descripción opcional del modelo para documentación.
        """
        # --- Validación de nombre ---
        if not name or not isinstance(name, str):
            raise ValueError("El nombre del modelo debe ser un string no vacío.")
        if " " in name:
            raise ValueError(f"El nombre '{name}' no debe contener espacios. Usa guiones bajos.")

        # --- Validación de indicadores ---
        if not indicators or not isinstance(indicators, list):
            raise ValueError("Debe especificarse al menos un indicador como lista.")
        if len(indicators) == 0:
            raise ValueError("La lista de indicadores no puede estar vacía.")

        self.name = name
        self.indicators = list(indicators)  # Copia defensiva
        self.logic_type = logic_type
        self.parameters = dict(parameters)  # Copia defensiva
        self.description = description
        self.created_at = datetime.now().isoformat(timespec="seconds")

        logger.info(
            f"[BaseModel] Modelo '{self.name}' inicializado | "
            f"lógica={self.logic_type} | "
            f"{len(self.indicators)} indicadores | "
            f"params={self.parameters}"
        )

    # ──────────────────────────────────────────────────────────────────────
    # INTERFAZ PÚBLICA
    # ──────────────────────────────────────────────────────────────────────

    def generate_signal(self, indicators_df: pd.DataFrame) -> pd.Series:
        """
        Genera la señal táctica para cada fecha.

        Este método:
        1. Valida que los indicadores requeridos existen en el DataFrame.
        2. Extrae el subconjunto de indicadores que el modelo necesita.
        3. Delega el cálculo al método abstracto `_compute_signal()`.
        4. Valida que la señal de salida está en {-1, 0, 1}.

        Parámetros
        ----------
        indicators_df : pd.DataFrame
            DataFrame completo de indicadores (de indicators.py).

        Retorna
        -------
        pd.Series : señal táctica (-1, 0, +1) con DatetimeIndex.
        """
        logger.info(f"[{self.name}] Generando señal táctica...")

        # --- Validar indicadores disponibles ---
        available, missing = self._check_indicators(indicators_df)

        if missing:
            raise KeyError(
                f"[{self.name}] Indicadores requeridos no encontrados: {missing}. "
                f"Disponibles: {list(indicators_df.columns)[:10]}..."
            )

        # --- Extraer subconjunto ---
        subset = indicators_df[self.indicators].copy()
        logger.info(
            f"[{self.name}] Subconjunto: {subset.shape[0]} meses × "
            f"{subset.shape[1]} indicadores"
        )

        # --- Computar señal (implementada por subclases) ---
        signal = self._compute_signal(subset)

        # --- Validar output ---
        signal = self._validate_signal(signal, indicators_df.index)

        # --- Log resumen ---
        valid = signal.dropna()
        if len(valid) > 0:
            counts = valid.value_counts().sort_index()
            total = len(valid)
            logger.info(f"[{self.name}] Señal generada ({total} meses válidos):")
            labels = {-1: "reducir", 0: "mantener", 1: "aumentar"}
            for val, count in counts.items():
                pct = 100 * count / total
                logger.info(
                    f"[{self.name}]   {int(val):+d} ({labels.get(int(val), '?'):>9s}): "
                    f"{count:>4d} meses ({pct:5.1f}%)"
                )

        return signal

    def save_model(self, models_dir: Path = MODELS_DIR) -> Path:
        """
        Guarda el modelo como archivo JSON.

        El archivo contiene toda la información necesaria para recrear
        el modelo: nombre, indicadores, lógica, parámetros y metadatos.

        Parámetros
        ----------
        models_dir : Path
            Directorio donde guardar el archivo.

        Retorna
        -------
        Path : ruta del archivo guardado.
        """
        models_dir.mkdir(parents=True, exist_ok=True)
        filepath = models_dir / f"{self.name}.json"

        model_dict = self.to_dict()

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(model_dict, f, indent=2, ensure_ascii=False)

        logger.info(f"[{self.name}] Modelo guardado: {filepath}")
        return filepath

    def to_dict(self) -> dict:
        """
        Serializa el modelo a un diccionario.

        Este diccionario contiene toda la información necesaria para
        recrear el modelo desde model_loader.py.
        """
        return {
            "name": self.name,
            "indicators": self.indicators,
            "logic_type": self.logic_type,
            "parameters": self.parameters,
            "description": self.description,
            "created_at": self.created_at,
            "n_indicators": len(self.indicators),
        }

    # ──────────────────────────────────────────────────────────────────────
    # MÉTODO ABSTRACTO (implementar en subclases)
    # ──────────────────────────────────────────────────────────────────────

    @abstractmethod
    def _compute_signal(self, subset: pd.DataFrame) -> pd.Series:
        """
        Calcula la señal táctica a partir del subconjunto de indicadores.

        Este método debe ser implementado por cada tipo de lógica concreto.
        Recibe SOLO los indicadores declarados por el modelo (ya validados).

        Parámetros
        ----------
        subset : pd.DataFrame
            DataFrame con solo las columnas declaradas en self.indicators.

        Retorna
        -------
        pd.Series : señal en valores {-1, 0, +1} o float que será
                    discretizada por _validate_signal().
        """
        pass

    # ──────────────────────────────────────────────────────────────────────
    # MÉTODOS INTERNOS
    # ──────────────────────────────────────────────────────────────────────

    def _check_indicators(
        self, indicators_df: pd.DataFrame,
    ) -> tuple[list[str], list[str]]:
        """Verifica qué indicadores están disponibles y cuáles faltan."""
        available = [col for col in self.indicators if col in indicators_df.columns]
        missing = [col for col in self.indicators if col not in indicators_df.columns]
        return available, missing

    @staticmethod
    def _validate_signal(signal: pd.Series, expected_index: pd.Index) -> pd.Series:
        """
        Valida y normaliza la señal de salida.

        Asegura que:
        - El índice coincide con el esperado.
        - Los valores son -1, 0 o 1 (o NaN).
        """
        # Asegurar que es Series con el índice correcto
        if not isinstance(signal, pd.Series):
            signal = pd.Series(signal, index=expected_index)

        # Forzar valores a {-1, 0, 1}
        # Si _compute_signal devuelve floats continuos, discretizar aquí
        # sería incorrecto — cada subclase debe devolver ya discretizado.
        valid_values = {-1, 0, 1}
        unique_vals = set(signal.dropna().unique())
        if not unique_vals.issubset(valid_values):
            invalid = unique_vals - valid_values
            raise ValueError(
                f"Señal contiene valores inválidos: {invalid}. "
                f"Solo se permiten {{-1, 0, 1}} y NaN."
            )

        signal.name = "signal"
        return signal

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"name='{self.name}', "
            f"logic='{self.logic_type}', "
            f"indicators={len(self.indicators)}, "
            f"params={self.parameters})"
        )


# ══════════════════════════════════════════════════════════════════════════════
# IMPLEMENTACIONES CONCRETAS DE LÓGICA
# ══════════════════════════════════════════════════════════════════════════════
# Cada clase implementa un tipo de lógica de decisión distinto.
# Para añadir una nueva lógica, basta con crear una nueva clase que
# herede de BaseModel e implemente _compute_signal().


class ZScoreCompositeModel(BaseModel):
    """
    Modelo basado en composite z-score de indicadores.

    Lógica idéntica a los modelos de régimen (Paso 4), pero aplicada
    para generar señales tácticas en lugar de clasificar regímenes.

    Funcionamiento:
    1. Para cada indicador, calcula un z-score EXPANSIVO
       (usa solo datos pasados → sin look-ahead bias).
    2. Aplica la dirección configurada en parameters["directions"].
    3. Promedia los z-scores dirigidos en un composite score.
    4. Clasifica:
       - Composite > threshold_buy  → +1 (aumentar)
       - Composite < threshold_sell → -1 (reducir)
       - Resto                      →  0 (mantener)

    Parámetros esperados:
        parameters = {
            "directions": {"indicator_name": +1 o -1, ...},
            "threshold_buy": float (ej: 0.5),
            "threshold_sell": float (ej: -0.5),
            "min_periods": int (ej: 24),
        }
    """

    def __init__(self, name, indicators, parameters, description=""):
        super().__init__(
            name=name,
            indicators=indicators,
            logic_type="zscore_composite",
            parameters=parameters,
            description=description,
        )
        # Validar parámetros específicos
        required_params = ["directions", "threshold_buy", "threshold_sell"]
        for p in required_params:
            if p not in self.parameters:
                raise ValueError(
                    f"[{self.name}] Parámetro requerido faltante: '{p}'. "
                    f"ZScoreCompositeModel requiere: {required_params}"
                )

        # Validar que cada indicador tiene dirección
        directions = self.parameters["directions"]
        for ind in self.indicators:
            if ind not in directions:
                raise ValueError(
                    f"[{self.name}] Indicador '{ind}' no tiene dirección definida "
                    f"en parameters['directions']."
                )

    def _compute_signal(self, subset: pd.DataFrame) -> pd.Series:
        """Implementación del composite z-score."""
        directions = self.parameters["directions"]
        threshold_buy = self.parameters["threshold_buy"]
        threshold_sell = self.parameters["threshold_sell"]
        min_periods = self.parameters.get("min_periods", 24)

        directed_zscores = pd.DataFrame(index=subset.index)

        for col in subset.columns:
            series = subset[col]
            direction = directions[col]

            # Z-score expansivo (sin look-ahead)
            exp_mean = series.expanding(min_periods=min_periods).mean()
            exp_std = series.expanding(min_periods=min_periods).std(ddof=1)
            exp_std = exp_std.replace(0, np.nan)
            zscore = (series - exp_mean) / exp_std

            directed_zscores[col] = zscore * direction

        # Composite: media de z-scores dirigidos
        composite = directed_zscores.mean(axis=1)

        # Clasificar señal
        signal = pd.Series(0, index=subset.index, dtype=int)
        signal[composite > threshold_buy] = 1
        signal[composite < threshold_sell] = -1
        signal[composite.isna()] = np.nan

        return signal


class ThresholdRulesModel(BaseModel):
    """
    Modelo basado en reglas deterministas con umbrales fijos.

    Cada indicador tiene un umbral superior (bullish) e inferior (bearish).
    Se cuentan cuántos indicadores están en zona bullish vs bearish.
    La señal se determina por mayoría.

    Funcionamiento:
    1. Para cada indicador, evalúa si está por encima del umbral bullish,
       por debajo del bearish, o en zona neutral.
    2. Cuenta votos: bullish (+1) y bearish (-1).
    3. Si la mayoría es bullish → +1. Si bearish → -1. Resto → 0.
       "Mayoría" se define como > 50% de los indicadores con datos válidos.

    Parámetros esperados:
        parameters = {
            "thresholds": {
                "indicator_name": {"bullish": float, "bearish": float},
                ...
            }
        }

    Ejemplo:
        "thresholds": {
            "trend_momentum_6m": {"bullish": 0.05, "bearish": -0.05},
            "vol_vix_level": {"bullish": 15, "bearish": 25},
        }

    Nota: "bullish" y "bearish" se interpretan como:
    - Si el valor está POR ENCIMA de "bullish" → voto bullish.
    - Si el valor está POR DEBAJO de "bearish" → voto bearish.
    - Para indicadores donde alto = malo (ej: VIX), definir bullish < bearish.
    """

    def __init__(self, name, indicators, parameters, description=""):
        super().__init__(
            name=name,
            indicators=indicators,
            logic_type="threshold_rules",
            parameters=parameters,
            description=description,
        )
        if "thresholds" not in self.parameters:
            raise ValueError(
                f"[{self.name}] Parámetro requerido faltante: 'thresholds'."
            )
        for ind in self.indicators:
            if ind not in self.parameters["thresholds"]:
                raise ValueError(
                    f"[{self.name}] Indicador '{ind}' no tiene umbrales definidos."
                )

    def _compute_signal(self, subset: pd.DataFrame) -> pd.Series:
        """Implementación de reglas por umbrales."""
        thresholds = self.parameters["thresholds"]

        votes = pd.DataFrame(index=subset.index)

        for col in subset.columns:
            series = subset[col]
            th = thresholds[col]
            bull = th["bullish"]
            bear = th["bearish"]

            # Determinar si bullish > bearish (normal) o bullish < bearish (invertido)
            if bull > bear:
                # Normal: alto = bueno (ej: momentum)
                vote = pd.Series(0, index=subset.index, dtype=int)
                vote[series > bull] = 1
                vote[series < bear] = -1
            else:
                # Invertido: bajo = bueno (ej: VIX, donde bullish=15 < bearish=25)
                vote = pd.Series(0, index=subset.index, dtype=int)
                vote[series < bull] = 1   # Por debajo del umbral bajo = bueno
                vote[series > bear] = -1  # Por encima del umbral alto = malo

            vote[series.isna()] = np.nan
            votes[col] = vote

        # Contar votos válidos
        n_valid = votes.notna().sum(axis=1)
        n_bullish = (votes == 1).sum(axis=1)
        n_bearish = (votes == -1).sum(axis=1)

        # Señal por mayoría simple (>50% de indicadores con dato)
        signal = pd.Series(0, index=subset.index, dtype=int)
        signal[n_bullish > n_valid / 2] = 1
        signal[n_bearish > n_valid / 2] = -1
        signal[n_valid == 0] = np.nan

        return signal


class WeightedCompositeModel(BaseModel):
    """
    Modelo basado en composite ponderado de indicadores normalizados.

    Similar al ZScoreComposite pero con pesos distintos por indicador.
    Permite dar más importancia a ciertos indicadores sin optimizar
    (los pesos los define el investigador con criterio económico).

    Funcionamiento:
    1. Calcula z-score expansivo de cada indicador.
    2. Aplica dirección y peso.
    3. Calcula media ponderada.
    4. Clasifica con umbrales.

    Parámetros esperados:
        parameters = {
            "weights": {"indicator_name": float, ...},
            "directions": {"indicator_name": +1 o -1, ...},
            "threshold_buy": float,
            "threshold_sell": float,
            "min_periods": int,
        }

    Los pesos no necesitan sumar 1; se normalizan internamente.
    """

    def __init__(self, name, indicators, parameters, description=""):
        super().__init__(
            name=name,
            indicators=indicators,
            logic_type="weighted_composite",
            parameters=parameters,
            description=description,
        )
        required_params = ["weights", "directions", "threshold_buy", "threshold_sell"]
        for p in required_params:
            if p not in self.parameters:
                raise ValueError(
                    f"[{self.name}] Parámetro requerido faltante: '{p}'."
                )
        for ind in self.indicators:
            if ind not in self.parameters["weights"]:
                raise ValueError(f"[{self.name}] Indicador '{ind}' sin peso definido.")
            if ind not in self.parameters["directions"]:
                raise ValueError(f"[{self.name}] Indicador '{ind}' sin dirección definida.")

    def _compute_signal(self, subset: pd.DataFrame) -> pd.Series:
        """Implementación del composite ponderado."""
        weights = self.parameters["weights"]
        directions = self.parameters["directions"]
        threshold_buy = self.parameters["threshold_buy"]
        threshold_sell = self.parameters["threshold_sell"]
        min_periods = self.parameters.get("min_periods", 24)

        # Normalizar pesos para que sumen 1
        total_weight = sum(weights[col] for col in subset.columns)
        if total_weight == 0:
            raise ValueError(f"[{self.name}] Los pesos suman 0.")

        directed_zscores = pd.DataFrame(index=subset.index)
        norm_weights = {}

        for col in subset.columns:
            series = subset[col]
            direction = directions[col]
            w = weights[col] / total_weight
            norm_weights[col] = w

            exp_mean = series.expanding(min_periods=min_periods).mean()
            exp_std = series.expanding(min_periods=min_periods).std(ddof=1)
            exp_std = exp_std.replace(0, np.nan)
            zscore = (series - exp_mean) / exp_std

            directed_zscores[col] = zscore * direction * w

        # Composite ponderado (suma, no media, porque pesos ya normalizados)
        composite = directed_zscores.sum(axis=1)

        # Clasificar
        signal = pd.Series(0, index=subset.index, dtype=int)
        signal[composite > threshold_buy] = 1
        signal[composite < threshold_sell] = -1
        signal[composite.isna()] = np.nan

        return signal


# ══════════════════════════════════════════════════════════════════════════════
# REGISTRO DE TIPOS DE LÓGICA
# ══════════════════════════════════════════════════════════════════════════════
# Este diccionario mapea el string "logic_type" a la clase concreta.
# Para añadir una nueva lógica, crear la clase y añadirla aquí.

LOGIC_REGISTRY: dict[str, type] = {
    "zscore_composite": ZScoreCompositeModel,
    "threshold_rules": ThresholdRulesModel,
    "weighted_composite": WeightedCompositeModel,
}


def get_available_logics() -> list[str]:
    """Devuelve la lista de tipos de lógica disponibles."""
    return list(LOGIC_REGISTRY.keys())