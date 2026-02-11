# Indicators — Paso 3: Universo de Indicadores de Contexto

## Explicación conceptual

Este módulo construye un universo amplio de **56 indicadores** distribuidos en **8 categorías** que describen el contexto económico-financiero en el que opera el S&P 500. Los indicadores miden el *estado del entorno*, no intentan predecir precios directamente.

La filosofía es generar un laboratorio estable: muchas variantes razonables (distintas ventanas, transformaciones) para que la **selección** se haga en fases posteriores con criterios explícitos y reproducibles. Ningún indicador se ha descartado por "funcionar peor" — esa evaluación no corresponde a esta fase.

### ¿Por qué un universo amplio?

Un sistema de asignación dinámica puede adoptar distintas filosofías (momentum, macro-regime, risk-parity, etc.) y cada una requiere distintos subconjuntos de indicadores. Tener un universo amplio pre-calculado permite iterar rápidamente sobre múltiples versiones del sistema sin rehacer el pipeline de datos.

---

## Inventario completo de indicadores

### Categoría 1: Tendencia de mercado (8 indicadores)

Mide la dirección y fuerza del movimiento del S&P 500 en distintos horizontes. El momentum tiene persistencia documentada en la literatura académica (Jegadeesh & Titman, 1993).

| Indicador | Descripción | Ventana |
|---|---|---|
| `trend_momentum_1m` | Retorno acumulado SPY 1 mes | 1m |
| `trend_momentum_3m` | Retorno acumulado SPY 3 meses | 3m |
| `trend_momentum_6m` | Retorno acumulado SPY 6 meses | 6m |
| `trend_momentum_12m` | Retorno acumulado SPY 12 meses | 12m |
| `trend_price_vs_ma_6m` | Ratio SPY / media móvil 6m | 6m |
| `trend_price_vs_ma_12m` | Ratio SPY / media móvil 12m | 12m |
| `trend_drawdown` | Drawdown desde máximo histórico | Acum. |
| `trend_momentum_accel` | Cambio MoM del momentum 6m | 6m+1m |

### Categoría 2: Volatilidad y riesgo (7 indicadores)

Mide la incertidumbre del mercado, tanto implícita (VIX, forward-looking) como realizada (backward-looking). El spread entre ambas es un proxy del variance risk premium (Bollerslev et al., 2009).

| Indicador | Descripción | Ventana |
|---|---|---|
| `vol_vix_level` | Nivel del VIX (media mensual) | Spot |
| `vol_vix_mom_change` | Cambio absoluto mensual del VIX | 1m |
| `vol_vix_zscore_24m` | Z-score 24m del VIX | 24m |
| `vol_realized_3m` | Vol. realizada (std retornos log) | 3m |
| `vol_realized_6m` | Vol. realizada (std retornos log) | 6m |
| `vol_realized_12m` | Vol. realizada (std retornos log) | 12m |
| `vol_implied_vs_realized_6m` | VIX - vol. realizada anualizada | 6m |

### Categoría 3: Valoración relativa (6 indicadores)

Mide el posicionamiento relativo del mercado de renta variable frente a otras clases de activos. **Limitación importante**: no disponemos de P/E, CAPE ni earnings yield; estos son indicadores de valor relativo cross-asset, no de valoración fundamental.

| Indicador | Descripción | Ventana |
|---|---|---|
| `val_equity_bond_ratio` | Ratio SPY/TLT | Spot |
| `val_equity_bond_zscore_24m` | Z-score 24m del ratio SPY/TLT | 24m |
| `val_equity_gold_ratio` | Ratio SPY/GLD | Spot |
| `val_equity_gold_momentum_12m` | Momentum 12m del ratio SPY/GLD | 12m |
| `val_real_yield_10y` | GS10 - Breakeven Inflation 10Y | Spot |
| `val_bond_yield_vs_spy_ret` | GS10 - retorno 12m SPY | 12m |

### Categoría 4: Ciclo económico (8 indicadores)

Mide el estado y la dirección de la actividad económica real. La producción industrial y el empleo son los pilares del NBER para datar recesiones.

| Indicador | Descripción | Lag natural |
|---|---|---|
| `cycle_indpro_yoy` | Crecimiento YoY producción industrial | ~1-2 meses |
| `cycle_indpro_mom_3m` | Momentum 3m de INDPRO | ~1-2 meses |
| `cycle_indpro_mom_6m` | Momentum 6m de INDPRO | ~1-2 meses |
| `cycle_indpro_accel` | Aceleración (cambio en YoY vs 3m atrás) | ~1-2 meses |
| `cycle_unemployment_level` | Tasa de desempleo U-3 | ~1 mes |
| `cycle_unemployment_yoy_diff` | Cambio YoY desempleo (pp) | ~1 mes |
| `cycle_unemployment_3m_diff` | Cambio 3m desempleo (pp) | ~1 mes |
| `cycle_nber_recession` | Recesión NBER (0/1) ⚠ SOLO VALIDACIÓN | 6-18 meses |

### Categoría 5: Política monetaria (8 indicadores)

Mide el estado de la política de la Fed y la curva de tipos. La inversión de curva es uno de los predictores de recesión mejor documentados (Estrella & Mishkin, 1996).

| Indicador | Descripción | Ventana |
|---|---|---|
| `mon_fedfunds_level` | Nivel de Fed Funds Rate | Spot |
| `mon_fedfunds_diff_6m` | Cambio Fed Funds 6m (pp) | 6m |
| `mon_fedfunds_diff_12m` | Cambio Fed Funds 12m (pp) | 12m |
| `mon_real_rate` | Tipo real: FEDFUNDS - CPI YoY | ~1-2 meses |
| `mon_yield_curve_level` | Spread 10Y-2Y | Spot |
| `mon_yield_curve_diff_3m` | Cambio curva 10Y-2Y 3m | 3m |
| `mon_yield_curve_diff_6m` | Cambio curva 10Y-2Y 6m | 6m |
| `mon_gs10_level` | Rendimiento Treasury 10Y | Spot |

### Categoría 6: Estrés financiero y crédito (7 indicadores)

Mide las condiciones del mercado de crédito corporativo. Los spreads de HY son especialmente sensibles al deterioro del ciclo.

| Indicador | Descripción | Ventana |
|---|---|---|
| `credit_hy_oas_level` | Spread HY option-adjusted | Spot |
| `credit_hy_oas_3m_change` | Cambio HY OAS 3m | 3m |
| `credit_hy_oas_zscore_24m` | Z-score 24m del HY OAS | 24m |
| `credit_hy_ig_ratio` | Ratio HYG/LQD (calidad crédito) | Spot |
| `credit_hy_ig_momentum_6m` | Momentum 6m ratio HYG/LQD | 6m |
| `credit_riskon_riskoff_ratio` | Ratio HYG/TLT (risk-on/off) | Spot |
| `credit_riskon_riskoff_mom_6m` | Momentum 6m ratio HYG/TLT | 6m |

### Categoría 7: Inflación y expectativas (7 indicadores)

Mide el nivel, dirección y expectativas de inflación. La dirección (aceleración/desaceleración) importa tanto como el nivel, porque determina la reacción de la Fed.

| Indicador | Descripción | Lag natural |
|---|---|---|
| `infl_cpi_yoy` | Inflación CPI year-over-year | ~2 semanas |
| `infl_cpi_mom` | Inflación CPI mes a mes | ~2 semanas |
| `infl_cpi_accel_6m` | Aceleración: CPI YoY actual vs 6m atrás | ~2 semanas |
| `infl_cpi_trend_6m` | Media móvil 6m del CPI MoM | ~2 semanas |
| `infl_breakeven_10y` | Breakeven inflation 10Y (mercado) | 0 |
| `infl_breakeven_3m_change` | Cambio breakeven 3m | 3m |
| `infl_surprise` | CPI YoY - Breakeven (sorpresa inflac.) | ~2 semanas |

### Categoría 8: Amplitud cross-asset (5 indicadores)

Mide la participación y coordinación entre clases de activos. **Limitación**: no es amplitud intra-equity (advance/decline, breadth del S&P 500), sino cross-asset entre 6-7 ETFs.

| Indicador | Descripción | Ventana |
|---|---|---|
| `breadth_positive_assets_1m` | Nº activos con retorno mensual > 0 | 1m |
| `breadth_positive_mom6m_fraction` | Fracción activos con mom 6m > 0 | 6m |
| `breadth_return_dispersion_1m` | Dispersión (std) de retornos entre activos | 1m |
| `breadth_avg_corr_12m` | Correlación media rolling 12m entre activos | 12m |
| `breadth_avg_return_3m` | Retorno medio 3m cross-asset | 3m |

---

## Cómo este universo permite múltiples versiones futuras

El diseño permite que fases posteriores seleccionen distintos subconjuntos sin rehacer cálculos:

- **Versión "macro pura"**: usar solo categorías 4, 5, 7 (ciclo + política monetaria + inflación).
- **Versión "market-based"**: usar solo categorías 1, 2, 6 (tendencia + volatilidad + crédito).
- **Versión "risk regime"**: combinar VIX z-score + HY OAS + curva de tipos + drawdown.
- **Versión "all-in"**: usar las 8 categorías con selección estadística.
- **Sensibilidad a ventanas**: comparar momentum 3m vs 6m vs 12m sin recalcular.

El archivo `indicators_metadata.csv` facilita el filtrado programático por categoría, fuente o limitaciones.

---

## Ejecución

```bash
# Prerrequisitos: haber ejecutado Pasos 1 y 2
python indicators.py
```

```python
from indicators import run_indicators

# Pipeline completo
results = run_indicators()
indicators = results["indicators"]   # DataFrame: 56 columnas × N meses
metadata = results["metadata"]       # DataFrame: 56 filas con metadatos

# Solo una categoría
from indicators import build_trend_indicators, load_processed_data
market, macro = load_processed_data()
trend = build_trend_indicators(market)
```

---

## Limitaciones del universo

1. **Sin valoración fundamental**: no disponemos de P/E, CAPE, earnings yield. Los indicadores de "valoración" son estrictamente de valor relativo cross-asset.
2. **Sin amplitud intra-equity**: solo tenemos SPY, no los 500 componentes. Los indicadores de breadth son cross-asset (6-7 ETFs), no amplitud del mercado de renta variable.
3. **Cobertura desigual**: HYG desde 2007, GLD desde 2004, T10YIE desde 2003. Los indicadores que dependen de estas series tendrán NaN antes de esas fechas.
4. **Revisiones retroactivas**: CPI, INDPRO y otros datos macro son valores "finales" revisados de FRED, no los publicados en tiempo real. Esto introduce un sesgo residual en backtesting.
5. **USREC no es señal**: la serie de recesiones NBER se incluye exclusivamente para validación ex-post, nunca como input de un modelo prospectivo.
6. **Granularidad mensual**: al trabajar con datos mensuales se pierde información intra-mes. Indicadores como VIX z-score o volatilidad realizada serían más precisos con datos diarios.
