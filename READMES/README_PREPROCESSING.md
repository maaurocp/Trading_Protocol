# Preprocessing — Paso 2: Validación, Limpieza y Alineación Temporal

## Explicación conceptual del proceso

El Paso 2 transforma datos raw heterogéneos en datasets mensuales alineados y auditados. El principio rector es **conservar la fidelidad de los datos**: no se inventan valores, no se usa información futura, y toda pérdida de datos se documenta explícitamente.

### Flujo del pipeline

```
data/raw/                          data/processed/
├── yf_SPY.csv ──┐                 ├── market_monthly.csv
├── yf_VIX.csv   │  ┌──────────┐  ├── macro_monthly.csv
├── yf_TLT.csv   ├─►│ AUDITORÍA│  ├── combined_monthly_raw.csv
├── yf_TIP.csv   │  └────┬─────┘  └── audit_report.csv
├── yf_LQD.csv   │       │
├── yf_HYG.csv   │       ▼
├── yf_GLD.csv ──┘  ┌──────────┐
                     │NORMALIZAR│
├── fred_CPI...──┐   │  ÍNDICE  │
├── fred_UNR...  │  └────┬─────┘
├── fred_FED...  │       │
├── fred_DFF...  ├──►    ▼
├── fred_T10...  │  ┌──────────┐
├── fred_GS1...  │  │RESAMPLE  │
├── fred_GS2...  │  │→ MENSUAL │
├── fred_IND...  │  └────┬─────┘
├── fred_USR...  │       │
├── fred_T10...  │       ▼
├── fred_BAM...──┘  ┌──────────┐
                    │ ALINEAR  │
                    │ & MERGE  │
                    └────┬─────┘
                         │
                         ▼
                    ┌──────────┐
                    │ GUARDAR  │
                    └──────────┘
```

### Decisiones de resampling

| Tipo de serie | Método | Justificación |
|---|---|---|
| Precios de activos (SPY, TLT, TIP, LQD, HYG, GLD) | Último valor del mes | Estándar financiero: el precio de cierre de fin de mes representa el valor de liquidación real del periodo. La media mensual suavizaría artificialmente la volatilidad. |
| VIX (volatilidad implícita) | Media mensual | El VIX es un indicador de régimen, no un activo. La media captura el nivel de estrés prevalente durante todo el mes, no un snapshot arbitrario del último día. |
| Tasas/spreads FRED diarios (DFF, T10Y2Y, T10YIE, HY_OAS) | Último valor del mes | Consistente con usar la información más actualizada al cierre del periodo. |
| Series FRED mensuales (CPI, UNRATE, FEDFUNDS, GS10, GS2, INDPRO, USREC) | Sin resampling | Ya son mensuales. Solo se normaliza el índice temporal al fin de mes para alinear con el resto. |

### Prevención de look-ahead bias

1. **Sin forward-fill entre meses**: si un dato no existe para un mes, queda como NaN.
2. **Resampling cerrado por mes**: solo se usan datos dentro del propio mes calendario.
3. **Sin ventanas centradas**: ninguna operación usa datos futuros.
4. **USREC marcado**: la serie de recesiones NBER se señaliza explícitamente como serie con retraso de publicación. El NBER declara recesiones con meses o años de retraso respecto a cuando realmente comenzaron.
5. **Revisiones de datos macro**: los valores de series como CPI o INDPRO son los datos "finales" de FRED, no los publicados originalmente. Esto introduce un sesgo residual inevitable a menos que se usen las series "vintage" de FRED (fuera del alcance de este proyecto).

---

## Ejecución

```bash
# Prerrequisitos: haber ejecutado data_loader.py previamente
python preprocessing.py
```

**Uso programático:**

```python
from preprocessing import run_preprocessing, process_market_data, process_macro_data

# Pipeline completo
results = run_preprocessing()
market = results["market"]
macro = results["macro"]
combined = results["combined"]

# Solo mercado o solo macro
market_df = process_market_data()
macro_df = process_macro_data()
```

---

## Datasets de salida

### `market_monthly.csv`

Datos de mercado resampleados a frecuencia mensual (fin de mes).

| Columna | Fuente | Método | Descripción |
|---|---|---|---|
| SPY | yfinance | last | S&P 500 ETF — Adj Close fin de mes |
| VIX | yfinance | mean | Volatilidad implícita — media mensual |
| TLT | yfinance | last | Treasury 20Y+ ETF — Adj Close fin de mes |
| TIP | yfinance | last | TIPS ETF — Adj Close fin de mes |
| LQD | yfinance | last | IG Corporate Bond ETF — Adj Close fin de mes |
| HYG | yfinance | last | HY Corporate Bond ETF — Adj Close fin de mes |
| GLD | yfinance | last | Gold ETF — Adj Close fin de mes |

### `macro_monthly.csv`

Datos macroeconómicos de FRED normalizados a frecuencia mensual (fin de mes).

| Columna | Serie FRED | Freq. original | Método | Descripción |
|---|---|---|---|---|
| CPI | CPIAUCSL | Mensual | — | Índice de precios al consumo |
| UNRATE | UNRATE | Mensual | — | Tasa de desempleo U-3 |
| FEDFUNDS | FEDFUNDS | Mensual | — | Fed Funds Rate (media mensual publicada) |
| DFF | DFF | Diaria | last | Fed Funds Rate — último valor del mes |
| T10Y2Y | T10Y2Y | Diaria | last | Spread curva 10Y-2Y |
| GS10 | GS10 | Mensual | — | Rendimiento Treasury 10Y |
| GS2 | GS2 | Mensual | — | Rendimiento Treasury 2Y |
| INDPRO | INDPRO | Mensual | — | Producción industrial (índice) |
| USREC | USREC | Mensual | — | Recesión NBER (0/1) ⚠ retraso |
| T10YIE | T10YIE | Diaria | last | Breakeven inflation 10Y |
| HY_OAS | BAMLH0A0HYM2 | Diaria | last | Spread HY option-adjusted |

### `combined_monthly_raw.csv`

Merge de mercado + macro con prefijos de fuente:
- `MKT_` → columnas de mercado
- `MAC_` → columnas macroeconómicas

Outer join: todas las fechas de ambos datasets, NaNs donde no hay cobertura.

### `audit_report.csv`

Informe de auditoría de los archivos raw: filas, rango temporal, NaNs, duplicados, frecuencia inferida.

---

## Limitaciones conocidas

1. **Cobertura desigual**: VIX desde 1990, SPY desde 1993, HYG desde 2007. El dataset combinado tiene NaNs estructurales en los años donde no existían ciertos ETFs.

2. **Revisiones retroactivas de datos macro**: los valores de FRED son datos "finales" revisados, no los publicados originalmente en cada fecha. Para un backtesting estricto se necesitarían las series "vintage" de FRED.

3. **USREC no es señal en tiempo real**: las recesiones se declaran con meses/años de retraso. Solo válido para validación ex-post.

4. **yfinance no es fuente oficial**: los Adj Close pueden recalcularse retroactivamente por Yahoo.

5. **Sin forward-fill**: series mensuales como CPI tienen un dato por mes. Al alinear con series que pueden tener su último dato a finales de mes, pueden existir desfases de días que el outer join resuelve correctamente (ambas series usan fin de mes como índice).

6. **Pérdida de granularidad**: al resamplear de diario a mensual se pierde información intra-mes (extremos, volatilidad intraperiodo). Esto es aceptable para el horizonte de la estrategia (mensual/semanal) pero debe tenerse en cuenta.
