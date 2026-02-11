# Data Loader — Módulo de Adquisición de Datos

## Proyecto
Análisis de contexto económico-financiero para una estrategia de asignación dinámica en el S&P 500 (70% buy-and-hold + 30% táctico).

---

## Diseño del módulo

El módulo `data_loader.py` sigue un diseño de **pipeline de extracción pura**: descarga datos crudos de dos fuentes públicas y los almacena sin transformación. Esto garantiza la reproducibilidad y separa la adquisición del procesamiento.

**Principios de diseño:**

1. **Datos raw**: no se aplica forward-fill, interpolación, ni eliminación de NaNs.
2. **Separación de fuentes**: funciones independientes para yfinance y FRED, ejecutables por separado o juntas.
3. **Configurabilidad**: tickers, series, fechas y rutas son parámetros modificables sin tocar la lógica.
4. **Archivos individuales + consolidado**: cada serie/ticker se guarda por separado (para trazabilidad) y FRED genera además un archivo consolidado (para conveniencia).
5. **Logging exhaustivo**: cada paso queda registrado con timestamps para auditoría.

---

## Instalación y ejecución

```bash
# 1. Instalar dependencias
pip install -r requirements.txt

# 2. Configurar API key de FRED (gratuita)
#    Registro: https://fred.stlouisfed.org/docs/api/api_key.html
export FRED_API_KEY="tu_api_key_aqui"

# 3. Ejecutar
python data_loader.py
```

**Uso programático (import):**

```python
from data_loader import download_all, download_yfinance_data, download_fred_data

# Descarga completa
data = download_all(fred_api_key="TU_KEY")

# Solo mercado
yf_data = download_yfinance_data()

# Solo macro
fred_data = download_fred_data(fred_api_key="TU_KEY")

# Con parámetros personalizados
yf_data = download_yfinance_data(
    tickers={"SPY": "S&P 500", "QQQ": "Nasdaq 100"},
    start_date="2000-01-01",
    end_date="2024-12-31",
)
```

---

## Datasets descargados

### Datos de mercado (yfinance) — frecuencia diaria

| Ticker | Nombre | Finalidad en el proyecto | Disponible desde |
|--------|--------|--------------------------|------------------|
| `SPY` | SPDR S&P 500 ETF | Activo principal de la estrategia (benchmark y target) | 1993-01 |
| `^VIX` | CBOE Volatility Index | Indicador de régimen de volatilidad (risk-on/off) | 1990-01 |
| `TLT` | iShares 20+ Year Treasury ETF | Proxy de tipos de interés largos y flight-to-quality | 2002-07 |
| `TIP` | iShares TIPS Bond ETF | Expectativas de inflación real del mercado | 2003-12 |
| `LQD` | iShares IG Corporate Bond ETF | Spread de crédito investment grade | 2002-07 |
| `HYG` | iShares HY Corporate Bond ETF | Estrés en crédito high yield | 2007-04 |
| `GLD` | SPDR Gold Shares | Activo refugio y cobertura inflacionaria | 2004-11 |

**Campos descargados por ticker:** Open, High, Low, Close, Adj Close, Volume, Dividends, Stock Splits.

### Datos macroeconómicos (FRED)

| Código | Nombre | Frecuencia | Finalidad en el proyecto |
|--------|--------|------------|--------------------------|
| `CPIAUCSL` | Consumer Price Index | Mensual | Nivel de inflación general |
| `UNRATE` | Unemployment Rate | Mensual | Indicador retrasado del ciclo económico |
| `FEDFUNDS` | Fed Funds Rate (mensual) | Mensual | Política monetaria — tipo oficial |
| `DFF` | Fed Funds Rate (diario) | Diaria | Política monetaria — granularidad diaria |
| `T10Y2Y` | Spread curva 10Y-2Y | Diaria | Señal de inversión de curva / recesión |
| `GS10` | Treasury 10Y Rate | Mensual | Tipo de interés de referencia largo plazo |
| `GS2` | Treasury 2Y Rate | Mensual | Expectativas de política monetaria |
| `INDPRO` | Industrial Production | Mensual | Indicador coincidente de actividad económica |
| `USREC` | NBER Recession Indicator | Mensual | Validación histórica (NO señal en tiempo real) |
| `T10YIE` | Breakeven Inflation 10Y | Diaria | Expectativas de inflación del mercado |
| `BAMLH0A0HYM2` | HY OAS Spread | Diaria | Estrés en mercado de crédito |

---

## Archivos generados en `data/raw/`

```
data/raw/
├── yf_SPY.csv            # Precios diarios SPY
├── yf_VIX.csv            # Precios diarios VIX
├── yf_TLT.csv            # Precios diarios TLT
├── yf_TIP.csv            # Precios diarios TIP
├── yf_LQD.csv            # Precios diarios LQD
├── yf_HYG.csv            # Precios diarios HYG
├── yf_GLD.csv            # Precios diarios GLD
├── fred_CPIAUCSL.csv     # CPI mensual
├── fred_UNRATE.csv       # Desempleo mensual
├── fred_FEDFUNDS.csv     # Fed Funds mensual
├── fred_DFF.csv          # Fed Funds diario
├── fred_T10Y2Y.csv       # Spread curva diario
├── fred_GS10.csv         # Treasury 10Y mensual
├── fred_GS2.csv          # Treasury 2Y mensual
├── fred_INDPRO.csv       # Producción industrial mensual
├── fred_USREC.csv        # Recesiones NBER mensual
├── fred_T10YIE.csv       # Breakeven inflation diario
├── fred_BAMLH0A0HYM2.csv # HY spread diario
└── fred_consolidated.csv # Todas las series FRED en un archivo
```

---

## Limitaciones conocidas

### yfinance
- **No es fuente oficial**: depende de scraping de Yahoo Finance; puede fallar o cambiar sin aviso.
- **Profundidad histórica variable**: SPY desde 1993, HYG solo desde 2007. No todos los tickers cubren el periodo completo.
- **Adj Close puede recalcularse**: Yahoo recalcula precios ajustados retroactivamente.
- **No hay datos intradiarios históricos** en periodos largos.

### FRED
- **Revisiones retroactivas**: los datos macroeconómicos se revisan. El valor disponible hoy para una fecha pasada puede diferir del publicado originalmente (look-ahead bias).
- **USREC tiene retraso**: el NBER declara recesiones con meses o años de retraso. Esta serie sirve solo para validación ex-post, no como señal.
- **Frecuencias mixtas**: series diarias y mensuales coexisten. La alineación temporal se hará en fases posteriores.
- **Requiere API key**: gratuita pero necesaria.

---

## Notas para fases posteriores

Este módulo **solo descarga y almacena datos**. Las siguientes tareas quedan para módulos posteriores:

- Alineación temporal de frecuencias (resampleo diario → semanal/mensual).
- Forward-fill o interpolación de series mensuales.
- Cálculo de retornos, volatilidad realizada, z-scores, etc.
- Construcción de features para el modelo de asignación.
- Tratamiento de look-ahead bias en datos macroeconómicos.
