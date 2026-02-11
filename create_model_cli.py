"""
================================================================================
CREATE MODEL CLI — create_model_cli.py
================================================================================
Proyecto: Análisis de contexto económico-financiero para asignación dinámica
          en el S&P 500.

Propósito: Interfaz interactiva de línea de comandos para crear modelos de
           decisión táctica sin editar código manualmente. Guía al usuario
           paso a paso por la configuración de indicadores, lógica, parámetros
           y metadatos del modelo.

Uso:
    python create_model_cli.py

Flujo:
    1. Muestra indicadores disponibles (de indicators_full.csv).
    2. Pide nombre, lógica, indicadores y régimen asociado.
    3. Pide parámetros específicos según la lógica elegida.
    4. Valida todo contra los módulos existentes.
    5. Crea el modelo via model_factory.create_model().
    6. Guarda en models/ con el campo "associated_regime" en el JSON.
    7. Muestra resumen.

Cómo ampliar para nuevos tipos de lógica:
    1. Añadir la nueva clase en model_base.py y registrarla en LOGIC_REGISTRY.
    2. Añadir una función _ask_params_<logic_name>() en este archivo.
    3. Registrarla en PARAM_COLLECTORS (al final de la sección de collectors).
    No hay que tocar nada más.

Autor: [Proyecto académico]
Fecha: 2026-02
================================================================================
"""

import json
import sys
import logging
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
# Imports de módulos del proyecto
# ──────────────────────────────────────────────────────────────────────────────
try:
    from model_factory import create_model, list_saved_models, list_available_indicators
    from model_base import get_available_logics, MODELS_DIR
except ImportError as e:
    print(f"\n✗ Error importando módulos del proyecto: {e}")
    print("  Asegúrate de ejecutar desde el directorio raíz del proyecto")
    print("  y de que model_base.py y model_factory.py estén presentes.")
    sys.exit(1)

# Configurar logging mínimo para que los módulos internos no inunden la consola
logging.basicConfig(level=logging.WARNING, format="%(message)s")

# ──────────────────────────────────────────────────────────────────────────────
# Regímenes disponibles (etiqueta organizativa, no afecta lógica)
# ──────────────────────────────────────────────────────────────────────────────
AVAILABLE_REGIMES = ["macro", "financial", "liquidity"]


# ══════════════════════════════════════════════════════════════════════════════
# 1. FUNCIONES DE PRESENTACIÓN
# ══════════════════════════════════════════════════════════════════════════════

def print_header():
    """Muestra la cabecera del programa."""
    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║   CREATE MODEL — Asistente de creación de modelos tácticos ║")
    print("║   Estrategia de asignación dinámica S&P 500                ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()


def show_available_indicators():
    """
    Carga y muestra los indicadores disponibles de indicators_full.csv,
    agrupados por categoría (prefijo).
    """
    indicators = list_available_indicators()

    if not indicators:
        print("  ⚠ No se encontró indicators_full.csv.")
        print("  Los indicadores no se validarán contra el archivo.")
        return []

    # Agrupar por prefijo (categoría)
    categories = {}
    for ind in indicators:
        prefix = ind.split("_")[0]
        categories.setdefault(prefix, []).append(ind)

    print("─" * 60)
    print("INDICADORES DISPONIBLES")
    print("─" * 60)

    category_names = {
        "trend": "Tendencia de mercado",
        "vol": "Volatilidad y riesgo",
        "val": "Valoración relativa",
        "cycle": "Ciclo económico",
        "mon": "Política monetaria",
        "credit": "Estrés financiero y crédito",
        "infl": "Inflación y expectativas",
        "breadth": "Amplitud cross-asset",
    }

    for prefix in sorted(categories.keys()):
        label = category_names.get(prefix, prefix)
        print(f"\n  [{label}]")
        for ind in sorted(categories[prefix]):
            print(f"    • {ind}")

    print(f"\n  Total: {len(indicators)} indicadores")
    print("─" * 60)
    return indicators


def show_saved_models():
    """Muestra los modelos ya guardados en disco."""
    saved = list_saved_models()
    if saved:
        print(f"\n  Modelos existentes ({len(saved)}):")
        for name in saved:
            print(f"    • {name}")
    else:
        print("\n  No hay modelos guardados todavía.")
    print()


def show_available_logics():
    """Muestra los tipos de lógica registrados."""
    logics = get_available_logics()
    print("\n  Tipos de lógica disponibles:")

    logic_descriptions = {
        "zscore_composite": "Composite z-score expansivo con dirección y umbrales",
        "threshold_rules": "Reglas deterministas con umbrales fijos por indicador",
        "weighted_composite": "Composite ponderado (z-score con pesos por indicador)",
    }

    for logic in logics:
        desc = logic_descriptions.get(logic, "")
        print(f"    • {logic:<25s} — {desc}")
    print()
    return logics


def show_available_regimes():
    """Muestra los regímenes disponibles para asociación."""
    print("\n  Regímenes disponibles (etiqueta organizativa):")
    regime_descriptions = {
        "macro": "Ciclo económico real (expansión/contracción)",
        "financial": "Condiciones financieras (risk-on/risk-off)",
        "liquidity": "Política monetaria (acomodaticio/restrictivo)",
    }
    for regime in AVAILABLE_REGIMES:
        desc = regime_descriptions.get(regime, "")
        print(f"    • {regime:<15s} — {desc}")
    print()


def print_model_summary(model_dict: dict, regime: str):
    """Muestra un resumen legible del modelo creado."""
    print()
    print("═" * 60)
    print("MODELO CREADO EXITOSAMENTE")
    print("═" * 60)
    print(f"  Nombre:     {model_dict['name']}")
    print(f"  Lógica:     {model_dict['logic_type']}")
    print(f"  Régimen:    {regime}")
    print(f"  Creado:     {model_dict['created_at']}")
    print(f"  Indicadores ({model_dict['n_indicators']}):")
    for ind in model_dict["indicators"]:
        print(f"    • {ind}")
    print(f"  Parámetros:")
    for key, val in model_dict["parameters"].items():
        if isinstance(val, dict):
            print(f"    {key}:")
            for k, v in val.items():
                print(f"      {k}: {v}")
        else:
            print(f"    {key}: {val}")
    if model_dict.get("description"):
        print(f"  Descripción: {model_dict['description']}")
    print(f"\n  ✓ Guardado en: models/{model_dict['name']}.json")
    print("═" * 60)


# ══════════════════════════════════════════════════════════════════════════════
# 2. FUNCIONES DE INPUT
# ══════════════════════════════════════════════════════════════════════════════

def ask_string(prompt: str, allow_empty: bool = False) -> str:
    """Pide un string al usuario. Repite hasta obtener input válido."""
    while True:
        value = input(prompt).strip()
        if value or allow_empty:
            return value
        print("  ✗ Este campo no puede estar vacío.")


def ask_float(prompt: str, default: float = None) -> float:
    """Pide un número decimal. Acepta valor por defecto."""
    while True:
        suffix = f" [{default}]" if default is not None else ""
        raw = input(f"{prompt}{suffix}: ").strip()

        if not raw and default is not None:
            return default

        try:
            return float(raw)
        except ValueError:
            print(f"  ✗ Valor inválido. Introduce un número (ej: 0.5, -0.3).")


def ask_int(prompt: str, default: int = None) -> int:
    """Pide un entero. Acepta valor por defecto."""
    while True:
        suffix = f" [{default}]" if default is not None else ""
        raw = input(f"{prompt}{suffix}: ").strip()

        if not raw and default is not None:
            return default

        try:
            return int(raw)
        except ValueError:
            print(f"  ✗ Valor inválido. Introduce un entero (ej: 24).")


def ask_direction(indicator_name: str) -> int:
    """
    Pide la dirección económica de un indicador.
    +1 = valor alto es favorable.
    -1 = valor alto es desfavorable.
    """
    while True:
        raw = input(f"    Dirección para '{indicator_name}' (+1 o -1): ").strip()
        if raw in ("+1", "1"):
            return 1
        elif raw in ("-1",):
            return -1
        else:
            print("  ✗ Introduce +1 (alto=bueno) o -1 (alto=malo).")


def ask_choice(prompt: str, options: list[str]) -> str:
    """Pide al usuario que elija de una lista de opciones."""
    while True:
        value = input(prompt).strip().lower()
        if value in options:
            return value
        print(f"  ✗ Opción inválida. Elige entre: {options}")


# ══════════════════════════════════════════════════════════════════════════════
# 3. INPUT BÁSICO DEL MODELO
# ══════════════════════════════════════════════════════════════════════════════

def ask_basic_inputs(
    available_indicators: list[str],
    available_logics: list[str],
    saved_models: list[str],
) -> dict:
    """
    Recoge los datos básicos del modelo: nombre, lógica, indicadores,
    régimen y descripción.

    Retorna
    -------
    dict con claves: name, logic, indicators, regime, description.
    """
    print("\n── DATOS BÁSICOS DEL MODELO ──\n")

    # --- Nombre ---
    while True:
        name = ask_string("  Nombre del modelo (sin espacios): ")
        if " " in name:
            print("  ✗ El nombre no debe contener espacios. Usa guiones bajos.")
            continue
        if name in saved_models:
            print(f"  ✗ Ya existe un modelo llamado '{name}'. Elige otro nombre.")
            continue
        break

    # --- Lógica ---
    show_available_logics()
    logic = ask_choice(
        f"  Tipo de lógica ({'/'.join(available_logics)}): ",
        available_logics,
    )

    # --- Indicadores ---
    print("\n  Introduce los indicadores separados por coma.")
    print("  (Copia los nombres exactos de la lista anterior)")
    while True:
        raw_indicators = ask_string("  Indicadores: ")
        indicators = [ind.strip() for ind in raw_indicators.split(",") if ind.strip()]

        if not indicators:
            print("  ✗ Debes seleccionar al menos un indicador.")
            continue

        # Validar contra disponibles (si hay lista)
        if available_indicators:
            invalid = [ind for ind in indicators if ind not in available_indicators]
            if invalid:
                print(f"  ✗ Indicadores no encontrados: {invalid}")
                print("  Verifica los nombres e intenta de nuevo.")
                continue

        print(f"  ✓ {len(indicators)} indicadores seleccionados.")
        break

    # --- Régimen asociado ---
    show_available_regimes()
    regime = ask_choice(
        f"  Régimen asociado ({'/'.join(AVAILABLE_REGIMES)}): ",
        AVAILABLE_REGIMES,
    )

    # --- Descripción ---
    description = ask_string(
        "  Descripción del modelo (opcional, Enter para omitir): ",
        allow_empty=True,
    )

    return {
        "name": name,
        "logic": logic,
        "indicators": indicators,
        "regime": regime,
        "description": description,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 4. PARÁMETROS ESPECÍFICOS POR TIPO DE LÓGICA
# ══════════════════════════════════════════════════════════════════════════════
# Cada función recoge los parámetros que su tipo de lógica necesita.
# Para añadir soporte a una nueva lógica, crear una función
# _ask_params_<nombre>() y registrarla en PARAM_COLLECTORS.

def _ask_params_zscore_composite(indicators: list[str]) -> dict:
    """
    Recoge parámetros para ZScoreCompositeModel.

    Requiere:
    - directions: dirección (+1/-1) por indicador.
    - threshold_buy: umbral para señal +1.
    - threshold_sell: umbral para señal -1.
    - min_periods: meses mínimos para estadísticas.
    """
    print("\n── PARÁMETROS: Z-SCORE COMPOSITE ──\n")

    # Umbrales
    print("  Umbrales del composite z-score:")
    threshold_buy = ask_float("    threshold_buy (señal +1 si composite >)", default=0.5)
    threshold_sell = ask_float("    threshold_sell (señal -1 si composite <)", default=-0.5)
    min_periods = ask_int("    min_periods (meses mínimos para z-score)", default=24)

    # Direcciones
    print("\n  Dirección económica de cada indicador:")
    print("    +1 = valor alto es favorable (ej: momentum, producción industrial)")
    print("    -1 = valor alto es desfavorable (ej: VIX, spreads de crédito)\n")

    directions = {}
    for ind in indicators:
        directions[ind] = ask_direction(ind)

    return {
        "directions": directions,
        "threshold_buy": threshold_buy,
        "threshold_sell": threshold_sell,
        "min_periods": min_periods,
    }


def _ask_params_threshold_rules(indicators: list[str]) -> dict:
    """
    Recoge parámetros para ThresholdRulesModel.

    Requiere umbrales bullish y bearish por indicador.

    Nota: para indicadores donde alto=malo (ej: VIX), el usuario
    debe definir bullish < bearish (ej: bullish=15, bearish=25).
    El modelo detecta automáticamente la inversión.
    """
    print("\n── PARÁMETROS: THRESHOLD RULES ──\n")
    print("  Para cada indicador, define los umbrales de decisión.")
    print("  • Si alto=bueno (ej: momentum):  bullish > bearish  (ej: 0.05, -0.05)")
    print("  • Si alto=malo  (ej: VIX):       bullish < bearish  (ej: 15, 25)\n")

    thresholds = {}
    for ind in indicators:
        print(f"\n  Indicador: {ind}")
        bullish = ask_float(f"    Umbral bullish", default=None)
        bearish = ask_float(f"    Umbral bearish", default=None)
        thresholds[ind] = {"bullish": bullish, "bearish": bearish}

    return {"thresholds": thresholds}


def _ask_params_weighted_composite(indicators: list[str]) -> dict:
    """
    Recoge parámetros para WeightedCompositeModel.

    Requiere:
    - weights: peso por indicador (se normalizan internamente).
    - directions: dirección (+1/-1) por indicador.
    - threshold_buy/sell: umbrales.
    - min_periods: meses mínimos.
    """
    print("\n── PARÁMETROS: WEIGHTED COMPOSITE ──\n")

    # Umbrales
    print("  Umbrales del composite ponderado:")
    threshold_buy = ask_float("    threshold_buy (señal +1 si composite >)", default=0.5)
    threshold_sell = ask_float("    threshold_sell (señal -1 si composite <)", default=-0.5)
    min_periods = ask_int("    min_periods (meses mínimos para z-score)", default=24)

    # Pesos
    print("\n  Peso de cada indicador (no necesitan sumar 1; se normalizan):")
    print("  Un peso mayor = más influencia en la señal final.\n")

    weights = {}
    for ind in indicators:
        weights[ind] = ask_float(f"    Peso para '{ind}'", default=1.0)

    # Direcciones
    print("\n  Dirección económica de cada indicador:")
    print("    +1 = valor alto es favorable")
    print("    -1 = valor alto es desfavorable\n")

    directions = {}
    for ind in indicators:
        directions[ind] = ask_direction(ind)

    return {
        "weights": weights,
        "directions": directions,
        "threshold_buy": threshold_buy,
        "threshold_sell": threshold_sell,
        "min_periods": min_periods,
    }


# ──────────────────────────────────────────────────────────────────────────────
# REGISTRO DE RECOLECTORES DE PARÁMETROS
#
# Mapea logic_type → función que recoge los parámetros.
# Para añadir soporte a una nueva lógica:
#   1. Crear _ask_params_<nombre>(indicators) → dict
#   2. Añadir aquí: "nombre": _ask_params_<nombre>
# ──────────────────────────────────────────────────────────────────────────────
PARAM_COLLECTORS = {
    "zscore_composite": _ask_params_zscore_composite,
    "threshold_rules": _ask_params_threshold_rules,
    "weighted_composite": _ask_params_weighted_composite,
}


def ask_parameters_for_logic(logic: str, indicators: list[str]) -> dict:
    """
    Despacha al recolector de parámetros correcto según el tipo de lógica.

    Raises
    ------
    ValueError : si no hay recolector registrado para la lógica.
    """
    if logic not in PARAM_COLLECTORS:
        raise ValueError(
            f"No hay recolector de parámetros para la lógica '{logic}'. "
            f"Registra uno en PARAM_COLLECTORS."
        )
    return PARAM_COLLECTORS[logic](indicators)


# ══════════════════════════════════════════════════════════════════════════════
# 5. PERSISTENCIA DEL RÉGIMEN ASOCIADO
# ══════════════════════════════════════════════════════════════════════════════

def _patch_json_with_regime(model_name: str, regime: str, models_dir: Path = MODELS_DIR):
    """
    Añade el campo 'associated_regime' al JSON del modelo guardado.

    Se hace post-save porque BaseModel.to_dict() no incluye este campo
    (es metadata organizativa, no lógica del modelo). Modificar el JSON
    después de guardarlo es más limpio que alterar la clase base.

    Parámetros
    ----------
    model_name : str
        Nombre del modelo (sin extensión).
    regime : str
        Régimen asociado (macro/financial/liquidity).
    models_dir : Path
        Directorio de modelos.
    """
    filepath = models_dir / f"{model_name}.json"

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    data["associated_regime"] = regime

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ══════════════════════════════════════════════════════════════════════════════
# 6. CONFIRMACIÓN
# ══════════════════════════════════════════════════════════════════════════════

def ask_confirmation(basics: dict, parameters: dict) -> bool:
    """
    Muestra un resumen antes de crear y pide confirmación.

    Retorna True si el usuario confirma, False si cancela.
    """
    print("\n" + "─" * 60)
    print("RESUMEN ANTES DE CREAR")
    print("─" * 60)
    print(f"  Nombre:       {basics['name']}")
    print(f"  Lógica:       {basics['logic']}")
    print(f"  Régimen:      {basics['regime']}")
    print(f"  Indicadores:  {basics['indicators']}")
    print(f"  Descripción:  {basics['description'] or '(ninguna)'}")
    print(f"  Parámetros:")
    for key, val in parameters.items():
        if isinstance(val, dict):
            print(f"    {key}:")
            for k, v in val.items():
                print(f"      {k}: {v}")
        else:
            print(f"    {key}: {val}")
    print("─" * 60)

    response = input("\n  ¿Crear este modelo? (s/n): ").strip().lower()
    return response in ("s", "si", "sí", "y", "yes")


# ══════════════════════════════════════════════════════════════════════════════
# 7. FUNCIÓN PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

def main():
    """
    Flujo principal del asistente de creación de modelos.

    1. Muestra cabecera e información.
    2. Recoge datos básicos (nombre, lógica, indicadores, régimen).
    3. Recoge parámetros según la lógica elegida.
    4. Pide confirmación.
    5. Crea el modelo via model_factory.
    6. Guarda en disco con metadata de régimen.
    7. Muestra resumen.
    """
    print_header()

    # --- Mostrar contexto ---
    available_indicators = show_available_indicators()
    available_logics = get_available_logics()
    saved_models = list_saved_models()
    show_saved_models()

    # --- Paso 1: Datos básicos ---
    try:
        basics = ask_basic_inputs(available_indicators, available_logics, saved_models)
    except (KeyboardInterrupt, EOFError):
        print("\n\n  Cancelado por el usuario.")
        return

    # --- Paso 2: Parámetros específicos de la lógica ---
    try:
        parameters = ask_parameters_for_logic(basics["logic"], basics["indicators"])
    except (KeyboardInterrupt, EOFError):
        print("\n\n  Cancelado por el usuario.")
        return

    # --- Paso 3: Confirmación ---
    try:
        if not ask_confirmation(basics, parameters):
            print("\n  Creación cancelada.")
            return
    except (KeyboardInterrupt, EOFError):
        print("\n\n  Cancelado por el usuario.")
        return

    # --- Paso 4: Crear modelo ---
    print("\n  Creando modelo...")

    try:
        # Desactivar validación de indicadores si no hay archivo
        # (la validación ya se hizo en ask_basic_inputs)
        model = create_model(
            name=basics["name"],
            indicators=basics["indicators"],
            logic=basics["logic"],
            parameters=parameters,
            description=basics["description"],
            validate_indicators=bool(available_indicators),
            save=True,
        )
    except (ValueError, KeyError) as e:
        print(f"\n  ✗ Error creando el modelo: {e}")
        return

    # --- Paso 5: Añadir régimen al JSON ---
    try:
        _patch_json_with_regime(basics["name"], basics["regime"])
    except Exception as e:
        print(f"\n  ⚠ Modelo creado pero no se pudo añadir el régimen al JSON: {e}")

    # --- Paso 6: Mostrar resumen ---
    model_dict = model.to_dict()
    print_model_summary(model_dict, basics["regime"])


# ══════════════════════════════════════════════════════════════════════════════
# 8. EJECUCIÓN DIRECTA
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    main()