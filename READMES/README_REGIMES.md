# Regime Classification — Paso 4: Modelos de Clasificación de Régimen

## Explicación conceptual

El Paso 4 transforma el universo de 56 indicadores del Paso 3 en **clasificaciones de régimen**: etiquetas discretas que describen el tipo de entorno económico-financiero actual. Cada modelo responde a una pregunta diferente:

| Modelo | Pregunta | Regímenes |
|---|---|---|
| `macro` | ¿En qué fase del ciclo económico estamos? | expansion / neutral / contraction |
| `financial` | ¿Las condiciones financieras favorecen o penalizan a los activos de riesgo? | risk_on / neutral / risk_off |
| `liquidity` | ¿La política monetaria está estimulando o restringiendo? | accommodative / neutral / restrictive |

Los tres modelos son **independientes** y capturan dimensiones distintas del contexto. Un mismo mes puede ser "contracción macro" pero "acomodaticio monetario" (la Fed bajando tipos para combatir la recesión). Esta ortogonalidad es intencional.

---

## Metodología: Composite Expanding Z-Score

Los tres modelos comparten la misma metodología para evitar arbitrariedad:

### Paso 1 — Z-score expansivo por indicador

Para cada indicador `X`, en cada fecha `t`:

```
z_t = (X_t - mean(X_1..X_t)) / std(X_1..X_t)
```

Usa **solo datos pasados** (expanding window). Los primeros 24 meses son NaN por estabilidad estadística. No hay look-ahead bias.

### Paso 2 — Aplicar dirección económica

Cada indicador tiene una dirección predefinida con justificación económica:
- **+1**: valor alto = condiciones favorables (ej: producción industrial creciendo)
- **-1**: valor alto = condiciones adversas (ej: desempleo subiendo)

Se multiplica el z-score por la dirección → un z-score positivo SIEMPRE significa "condiciones favorables" para ese modelo.

### Paso 3 — Composite score

Se promedian los z-scores dirigidos de todos los indicadores del modelo. La media (no suma) hace que el composite sea comparable incluso si algún indicador no está disponible.

### Paso 4 — Clasificación con umbrales fijos

```
Composite > +0.5σ  →  Régimen positivo  ( 1)
Composite < -0.5σ  →  Régimen negativo  (-1)
Resto               →  Neutral           ( 0)
```

El umbral de **±0.5σ** (medio sigma) es una convención estándar en investigación cuantitativa. No se ha optimizado mirando resultados históricos; es un punto de corte simétrico que separa condiciones moderadamente por encima/debajo de la media.

### ¿Por qué este enfoque?

- **Sin optimización**: los umbrales son fijos y simétricos, no ajustados a datos.
- **Adaptativo**: el z-score expansivo adapta automáticamente lo que "alto" o "bajo" significa a medida que se acumula historia.
- **Robusto**: al promediar múltiples indicadores, ninguno domina el resultado.
- **Interpretable**: cada componente tiene significado económico claro.
- **Sin look-ahead**: verificado formalmente — añadir datos futuros no cambia regímenes pasados.

---

## Indicadores por modelo

### Modelo Macro

| Indicador | Dirección | Rol |
|---|---|---|
| `cycle_indpro_yoy` | + | Crecimiento de la actividad industrial |
| `cycle_indpro_accel` | + | Aceleración/desaceleración del crecimiento |
| `cycle_unemployment_yoy_diff` | - | Deterioro del mercado laboral (YoY) |
| `cycle_unemployment_3m_diff` | - | Deterioro laboral reciente (3m) |
| `mon_yield_curve_level` | + | Curva de tipos (invertida = riesgo recesión) |

### Modelo Financial

| Indicador | Dirección | Rol |
|---|---|---|
| `vol_vix_zscore_24m` | - | Volatilidad implícita en niveles extremos |
| `vol_implied_vs_realized_6m` | - | Prima de riesgo de varianza |
| `credit_hy_oas_zscore_24m` | - | Estrés en crédito high yield |
| `credit_hy_oas_3m_change` | - | Dirección reciente del estrés crediticio |
| `credit_riskon_riskoff_mom_6m` | + | Rotación entre riesgo y refugio |
| `trend_drawdown` | + | Profundidad de la caída desde máximos |

### Modelo Liquidity

| Indicador | Dirección | Rol |
|---|---|---|
| `mon_real_rate` | - | Tipo real (positivo = restrictivo) |
| `mon_fedfunds_diff_12m` | - | Dirección de la política de la Fed |
| `mon_yield_curve_level` | + | Pendiente de la curva (empinada = sana) |
| `mon_yield_curve_diff_6m` | + | Cambio reciente en la pendiente |
| `infl_cpi_accel_6m` | - | Aceleración de inflación (fuerza restricción) |
| `infl_breakeven_3m_change` | - | Expectativas de inflación al alza |

---

## Uso

### Uso básico

```python
from regime_selector import get_regime

# Seleccionar un modelo
regime = get_regime(model="macro")
regime = get_regime(model="financial")
regime = get_regime(model="liquidity")

# El resultado es una pd.Series: -1, 0, 1
print(regime.tail(12))
```

### Guardar en disco

```python
regime = get_regime(model="macro", save=True)
# → Guarda en data/processed/regimes/regime_macro.csv
```

### Ejecutar todos los modelos

```python
from regime_selector import get_all_regimes

all_regimes = get_all_regimes(save=True)
# → DataFrame con columnas: regime_macro, regime_financial, regime_liquidity
```

### Usar con indicadores propios

```python
import pandas as pd
from regime_selector import get_regime

my_indicators = pd.read_csv("my_custom_indicators.csv", index_col=0, parse_dates=True)
regime = get_regime(model="financial", indicators=my_indicators)
```

### Desde línea de comandos

```bash
# Ejecutar todos los modelos
python regime_selector.py

# Ejecutar uno específico
python regime_selector.py macro
```

---

## Cómo añadir un nuevo modelo de régimen

El sistema es extensible por diseño. Para añadir un modelo (por ejemplo, "sentiment"):

**1. Crear el archivo `regime_model_sentiment.py`** con esta interfaz mínima:

```python
def get_regime_series(indicators: pd.DataFrame) -> pd.Series:
    """Debe devolver pd.Series con valores en {-1, 0, 1}."""
    # Tu lógica aquí
    return regime_series
```

**2. Registrar en `regime_selector.py`**, añadiendo una entrada al diccionario:

```python
MODEL_REGISTRY = {
    # ... modelos existentes ...
    "sentiment": {
        "module": "regime_model_sentiment",
        "description": "Régimen de sentimiento de mercado — ..."
    },
}
```

**3. Usar inmediatamente:**

```python
regime = get_regime(model="sentiment")
```

No hay que modificar nada más. La importación dinámica se encarga del resto.

---

## Archivos de salida

```
data/processed/regimes/
├── regime_macro.csv        # Serie temporal: date, regime_macro
├── regime_financial.csv    # Serie temporal: date, regime_financial
├── regime_liquidity.csv    # Serie temporal: date, regime_liquidity
└── regimes_all.csv         # Consolidado de los tres modelos
```

---

## Limitaciones

1. **Umbrales fijos ±0.5σ**: razonables pero arbitrarios. No hay garantía de que produzcan la separación óptima de regímenes. Esto es deliberado — optimizar umbrales introduciría overfitting.

2. **Pesos iguales**: todos los indicadores dentro de un modelo pesan lo mismo. Un esquema de pesos podría mejorar la señal pero requeriría criterios de optimización (fuera del alcance de esta fase).

3. **Sin persistencia**: el régimen puede cambiar de un mes a otro sin inercia. En la práctica, los regímenes económicos son persistentes. Un filtro de persistencia (ej: "cambiar solo si el nuevo régimen se mantiene 2 meses") sería una mejora natural para fases posteriores.

4. **Sensibilidad al inicio**: el z-score expansivo es menos estable en los primeros años (poca historia). Los primeros 24 meses se descartan, pero los meses 25-48 siguen teniendo más incertidumbre estadística que los posteriores.

5. **Sin combinación de modelos**: los tres modelos son independientes. No se define aquí cómo combinarlos (ensemble, votación, ponderación). Eso corresponde a fases posteriores.
