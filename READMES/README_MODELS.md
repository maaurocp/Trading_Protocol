# Model Framework — Paso 5: Laboratorio de Modelos Tácticos

## Explicación conceptual

El Paso 5 construye la infraestructura para crear, almacenar y reutilizar modelos de decisión táctica sobre el S&P 500. Es un **laboratorio de modelos**: permite definir combinaciones arbitrarias de indicadores + lógica + parámetros, guardarlas como archivos JSON y recargarlas en fases posteriores (backtesting, ejecución).

### Separación de responsabilidades

```
┌─────────────────────────────────────────────────────────────┐
│                    ESTE MÓDULO                              │
│                                                             │
│  Indicadores → [Modelo Táctico] → Señal (-1, 0, +1)       │
│                                                             │
│  El modelo opera como si controlara el 100% de su capital.  │
│  Solo produce señal táctica.                                │
└─────────────────────────────────────────────────────────────┘
         ↑                                    ↓
     indicators.py                    fases posteriores
     (Paso 3)                         (asignación 70/30,
                                       backtesting)

NO conoce: regímenes, pesos, portfolio, backtesting.
```

### Arquitectura

```
model_factory.py  ── crea ──→  BaseModel (instancia)
                                   │
                                   ├── .generate_signal(df) → pd.Series
                                   ├── .save_model()        → JSON
                                   └── .to_dict()           → dict

model_loader.py   ── carga ──→  BaseModel (instancia)
                                   │
                                   └── misma interfaz

model_base.py     ── define ──→  BaseModel (abstracta)
                                   ├── ZScoreCompositeModel
                                   ├── ThresholdRulesModel
                                   └── WeightedCompositeModel
```

---

## Tipos de lógica disponibles

### 1. `zscore_composite` — Composite Z-Score con dirección

La misma metodología que los modelos de régimen (Paso 4), pero para señales tácticas. Z-score expansivo (sin look-ahead) con dirección económica y umbrales.

```python
model = create_model(
    name="macro_momentum_v1",
    indicators=["trend_momentum_6m", "cycle_indpro_yoy", "vol_vix_zscore_24m"],
    logic="zscore_composite",
    parameters={
        "directions": {
            "trend_momentum_6m": +1,      # Momentum alto = bueno
            "cycle_indpro_yoy": +1,       # Crecimiento industrial = bueno
            "vol_vix_zscore_24m": -1,     # VIX alto = malo
        },
        "threshold_buy": 0.5,
        "threshold_sell": -0.5,
        "min_periods": 24,
    },
)
```

### 2. `threshold_rules` — Reglas deterministas con umbrales

Cada indicador vota bullish/bearish según umbrales fijos. La señal se decide por mayoría simple.

```python
model = create_model(
    name="simple_rules_v1",
    indicators=["trend_momentum_6m", "vol_vix_level", "mon_yield_curve_level"],
    logic="threshold_rules",
    parameters={
        "thresholds": {
            "trend_momentum_6m": {"bullish": 0.05, "bearish": -0.05},
            "vol_vix_level": {"bullish": 15, "bearish": 25},         # Invertido: bajo=bueno
            "mon_yield_curve_level": {"bullish": 1.0, "bearish": -0.5},
        },
    },
)
```

Para indicadores donde alto = malo (VIX), definir `bullish < bearish`. El modelo detecta automáticamente la inversión.

### 3. `weighted_composite` — Composite ponderado

Como zscore_composite pero con pesos distintos por indicador. Permite al investigador expresar un criterio sobre qué indicadores son más relevantes, sin optimización.

```python
model = create_model(
    name="weighted_macro_v1",
    indicators=["trend_momentum_6m", "cycle_indpro_yoy", "credit_hy_oas_zscore_24m"],
    logic="weighted_composite",
    parameters={
        "weights": {
            "trend_momentum_6m": 2.0,            # Doble peso
            "cycle_indpro_yoy": 1.0,
            "credit_hy_oas_zscore_24m": 1.5,
        },
        "directions": {
            "trend_momentum_6m": +1,
            "cycle_indpro_yoy": +1,
            "credit_hy_oas_zscore_24m": -1,
        },
        "threshold_buy": 0.5,
        "threshold_sell": -0.5,
    },
)
```

---

## Flujo completo de uso

### Crear un modelo

```python
from model_factory import create_model

model = create_model(
    name="mi_modelo_v1",
    indicators=["trend_momentum_6m", "vol_vix_zscore_24m"],
    logic="zscore_composite",
    parameters={
        "directions": {"trend_momentum_6m": +1, "vol_vix_zscore_24m": -1},
        "threshold_buy": 0.5,
        "threshold_sell": -0.5,
    },
    description="Mi primer modelo táctico",
    save=True,  # Guarda automáticamente en models/
)
```

### Generar señal

```python
import pandas as pd
indicators_df = pd.read_csv("data/indicators/indicators_full.csv", index_col=0, parse_dates=True)

signal = model.generate_signal(indicators_df)
# signal es pd.Series con valores {-1, 0, +1}
```

### Guardar y cargar

```python
# Guardar
model.save_model()  # → models/mi_modelo_v1.json

# Cargar más tarde
from model_loader import load_model
model = load_model("mi_modelo_v1")
signal = model.generate_signal(indicators_df)
```

### Explorar modelos guardados

```python
from model_loader import list_models, inspect_model, load_all_models

# Listar nombres
names = list_models()  # ['mi_modelo_v1', 'macro_momentum_v1', ...]

# Ver metadatos sin instanciar
meta = inspect_model("mi_modelo_v1")
print(meta["indicators"])
print(meta["parameters"])

# Cargar todos a la vez
all_models = load_all_models()
for name, model in all_models.items():
    signal = model.generate_signal(indicators_df)
```

---

## Estructura de almacenamiento

```
models/
├── macro_momentum_v1.json
├── risk_composite_v2.json
└── simple_rules_v1.json
```

Cada archivo JSON contiene:

```json
{
  "name": "macro_momentum_v1",
  "indicators": ["trend_momentum_6m", "cycle_indpro_yoy", "vol_vix_zscore_24m"],
  "logic_type": "zscore_composite",
  "parameters": {
    "directions": {"trend_momentum_6m": 1, "cycle_indpro_yoy": 1, "vol_vix_zscore_24m": -1},
    "threshold_buy": 0.5,
    "threshold_sell": -0.5,
    "min_periods": 24
  },
  "description": "Modelo táctico basado en momentum, ciclo y volatilidad",
  "created_at": "2026-02-11T10:30:00",
  "n_indicators": 3
}
```

---

## Cómo añadir un nuevo tipo de lógica

Para añadir una lógica (por ejemplo, "moving_average_crossover"):

**1. Crear la clase en `model_base.py`:**

```python
class MovingAverageCrossoverModel(BaseModel):
    def __init__(self, name, indicators, parameters, description=""):
        super().__init__(
            name=name,
            indicators=indicators,
            logic_type="ma_crossover",  # Identificador
            parameters=parameters,
            description=description,
        )

    def _compute_signal(self, subset: pd.DataFrame) -> pd.Series:
        # Implementar lógica
        # Debe devolver pd.Series con valores en {-1, 0, 1}
        ...
```

**2. Registrar en `LOGIC_REGISTRY`** (al final de `model_base.py`):

```python
LOGIC_REGISTRY = {
    "zscore_composite": ZScoreCompositeModel,
    "threshold_rules": ThresholdRulesModel,
    "weighted_composite": WeightedCompositeModel,
    "ma_crossover": MovingAverageCrossoverModel,  # ← Nuevo
}
```

**3. Usar inmediatamente:**

```python
model = create_model(name="test_ma", indicators=[...], logic="ma_crossover", ...)
```

No hay que modificar `model_factory.py` ni `model_loader.py`.

---

## Limitaciones

1. **Sin optimización**: los parámetros (umbrales, pesos, direcciones) los define el investigador. No hay búsqueda automática de valores óptimos.

2. **Sin backtesting**: los modelos generan señales pero no se evalúa su rentabilidad. Eso corresponde a fases posteriores.

3. **Sin combinación con régimen**: los modelos tácticos y los modelos de régimen operan en capas separadas. La combinación (ej: "usar señal táctica solo si el régimen es favorable") es responsabilidad de la capa de asignación.

4. **JSON como formato**: simple y legible, pero no soporta tipos complejos (funciones, objetos). Si se necesitara almacenar lógica más compleja, habría que migrar a pickle o dill (con las implicaciones de seguridad correspondientes).

5. **Validación estática**: la factory valida que los indicadores existan en el CSV, pero no verifica que tengan suficientes datos no-NaN para producir señales útiles.