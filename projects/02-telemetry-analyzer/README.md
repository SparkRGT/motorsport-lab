# F1 Telemetry Analyzer

Analizador de telemetría de EA SPORTS F1 25. Lee un CSV ya grabado, valida cada vuelta y compara dos vueltas sobre la distancia que comparten. El resultado es un objeto de Python y, si se pide, un informe JSON con gráficas PNG.

Este proyecto es el Proyecto 02 de Motorsport Lab. No hay interfaz de consola, API web ni base de datos.

## Problema

Una sesión de F1 25 produce miles de muestras de velocidad, acelerador, freno y distancia. Comparar dos vueltas por el reloj de la sesión no dice en qué tramo apareció la diferencia. Este analizador alinea las vueltas por `lap_distance` y mide ahí el tiempo, las zonas de frenada y aceleración, y los segmentos de curva derivados de esa telemetría.

## Objetivos y alcance

El proyecto ya permite:

- cargar, validar y normalizar el CSV;
- segmentar las vueltas por `lap_number`;
- calcular métricas básicas y compararlas;
- alinear dos vueltas por distancia y calcular el delta temporal;
- detectar zonas de frenado, zonas de aceleración y segmentos de curva;
- localizar tramos donde el delta temporal acumulado cambia;
- medir indicadores de frenada, curva y aceleración sobre las ventanas de la vuelta de referencia;
- exportar un informe de sesión en JSON y cuatro gráficas PNG.

No clasifica errores de pilotaje, no recomienda una trazada y no identifica curvas oficiales del circuito. Esas interpretaciones quedan para el Proyecto 03.

## Tecnologías

- Python 3.12
- pandas
- NumPy
- matplotlib
- pytest

`pyproject.toml` también declara `pyarrow`. El flujo implementado lee CSV y no escribe Parquet.

## Arquitectura

```text
EA SPORTS F1 25
       │
       ▼
Proyecto 01 — F1 Telemetry Collector
       │
       ▼
CSV de telemetría
       │
       ▼
load_csv → validate_dataset → normalize_dataset → segment_laps
       │
       ├── calculate_basic_metrics → analyze_session
       └── analyze_lap_performance
              ├── align_laps_by_distance
              ├── calculate_time_delta
              └── detect_performance_segments
                     │
                     ├── analyze_event_performance
                     ├── calculate_driver_performance_metrics
                     └── generate_session_report
```

Los módulos públicos están en `src/telemetry_analyzer/`:

| Módulo | Responsabilidad |
| --- | --- |
| `ingestion` | `load_csv` |
| `processing` | `validate_dataset`, `normalize_dataset`, `segment_laps` |
| `metrics` | `calculate_basic_metrics`, `compare_laps` |
| `analysis` | sesión, alineación, delta temporal, zonas, curvas, eventos e indicadores |
| `visualization` | gráficas de velocidad, acelerador y freno, y `save_figure` |
| `reporting` | `generate_session_report` |

## Estructura

```text
02-telemetry-analyzer/
├── README.md
├── pyproject.toml
├── fixtures/
│   ├── telemetry_sample.csv
│   └── telemetry_laps_sample.csv
├── src/telemetry_analyzer/
└── tests/
```

`fixtures/` contiene los CSV sintéticos que usan las pruebas. `output/` es el directorio documentado para informes generados y está ignorado por Git. `data/real/` es solo para capturas locales y tampoco se versiona.

## Requisitos

- Windows con PowerShell
- Python 3.12

Los ejemplos de este documento se ejecutan desde `projects/02-telemetry-analyzer`, con el entorno virtual activado.

## Instalación

Desde la raíz del repositorio:

```powershell
Set-Location projects\02-telemetry-analyzer
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Si PowerShell bloquea la activación, habilítala solo para esta sesión y vuelve a ejecutar el comando de activación:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

## Pruebas

Desde `projects/02-telemetry-analyzer`, con el entorno activado:

```powershell
python -m pytest -v
```

Las pruebas que leen el CSV del Proyecto 01 se omiten si ese archivo no está en el disco. No copian la captura al repositorio.

## Analizar un CSV

Desde `projects/02-telemetry-analyzer`:

```python
from pathlib import Path

from telemetry_analyzer.analysis import analyze_session
from telemetry_analyzer.ingestion.csv_loader import load_csv
from telemetry_analyzer.metrics import calculate_basic_metrics
from telemetry_analyzer.processing import normalize_dataset, segment_laps, validate_dataset

csv_path = Path("fixtures/telemetry_laps_sample.csv")

dataset = load_csv(csv_path)
validate_dataset(dataset)
laps = segment_laps(normalize_dataset(dataset))

for lap in laps.values():
    print(lap.lap_number, calculate_basic_metrics(lap).duration)

summary = analyze_session(csv_path)
print(summary.lap_numbers, summary.fastest_lap, summary.slowest_lap)
```

`fixtures/telemetry_laps_sample.csv` tiene las vueltas 1, 2 y 3. `analyze_session` elige la vuelta más rápida y la más lenta solo entre las vueltas cuya `lap_distance` máxima llega al menos al 90 % de la vuelta más larga de ese archivo. Si hay menos de dos vueltas que cumplan esa condición, `fastest_lap` y `slowest_lap` quedan vacíos. Una vuelta corta sigue apareciendo en `lap_numbers`.

## Informe JSON y gráficas

Desde `projects/02-telemetry-analyzer`:

```python
from pathlib import Path

from telemetry_analyzer.reporting import generate_session_report

csv_path = Path("fixtures/telemetry_laps_sample.csv")

summary = generate_session_report(csv_path)

report = generate_session_report(
    csv_path,
    reference_lap=1,
    compared_lap=2,
    output_dir=Path("output/sample"),
)

print(report.comparison_note)
print(report.plots)
```

Sin números de vuelta, `comparison` es `None` y `comparison_note` indica que no se pidió comparación. Con las dos vueltas, el informe incluye la comparación de métricas, el delta temporal por distancia, los segmentos de rendimiento, los eventos y los indicadores de frenada, curva y aceleración.

Hay que pasar las dos vueltas o ninguna. Una sola vuelta, un número que no existe en el archivo, un CSV ausente o un dataset inválido producen un error. Si las vueltas no comparten distancia, el resumen de sesión se conserva, `comparison` queda vacío y `comparison_note` lo explica.

Con `output_dir` el informe queda así:

```text
output/sample/
├── session_report.json
└── plots/
    ├── speed_comparison.png
    ├── speed_delta.png
    ├── throttle_comparison.png
    └── brake_comparison.png
```

Las gráficas solo se crean cuando la comparación existe. En el JSON, `plots` usa rutas relativas a `output_dir`, por ejemplo `plots/speed_comparison.png`. El JSON no incluye la tabla alineada ni la ruta absoluta del equipo. `output/` no se versiona.

## Comparar dos vueltas

`reference_lap` es la base. `compared_lap` es la vuelta que se resta. La convención de todo el proyecto es:

```text
delta = compared - reference
```

Un delta positivo significa que la vuelta comparada tiene un valor mayor: más tiempo de sesión, más velocidad o más tiempo acumulado en ese tramo, según el campo. Un delta negativo significa que la vuelta comparada tiene un valor menor. El signo no se traduce a "mejor" o "peor".

## Duración de vuelta y delta temporal

Son dos medidas distintas.

La duración de una vuelta es `LapMetrics.duration`: el `session_time` de la última muestra menos el de la primera. `LapComparison.time_delta` resta esas dos duraciones. `SessionSummary.fastest_lap_time` y `slowest_lap_time` también usan esa duración.

`total_time_delta` se calcula sobre la distancia compartida. En cada intervalo, el tiempo sale de la distancia y de la velocidad en km/h. Dos vueltas con la misma velocidad en el mismo tramo pueden tener `total_time_delta` cercano a cero aunque sus duraciones de `session_time` sean distintas, porque el reloj de la sesión y la velocidad registrada no se fuerzan a coincidir.

## Columnas del CSV

El archivo debe traer, en este orden, `session_time`, `frame`, `lap_number`, `lap_distance`, `speed`, `throttle`, `brake`, `steering`, `gear`, `rpm` y `drs`. La velocidad está en km/h, la distancia en metros y el acelerador y el freno entre 0 y 1. `drs` es 0 o 1. `gear` admite `-1` para marcha atrás.

## Relación con el Proyecto 01

El Proyecto 01, `projects/01-telemetry-collector`, graba la telemetría UDP de F1 25. Empareja el paquete 6 (Car Telemetry) con el paquete 2 (Lap Data) y añade una fila por fotograma a `projects/01-telemetry-collector/data/telemetry/telemetry_session.csv`.

Este analizador lee ese archivo donde está. No lo copia a este proyecto. Desde `projects/02-telemetry-analyzer`:

```python
from pathlib import Path

from telemetry_analyzer.reporting import generate_session_report

csv_path = Path("../01-telemetry-collector/data/telemetry/telemetry_session.csv")
summary = generate_session_report(csv_path)
print(summary.session.lap_numbers)

report = generate_session_report(
    csv_path,
    reference_lap=1,
    compared_lap=2,
    output_dir=Path("output/collector"),
)
print(report.comparison_note)
```

Las vueltas 1 y 2 tienen que aparecer en `summary.session.lap_numbers`. Si no están, la llamada lanza un error. Si están y no comparten distancia, `comparison` queda vacío y `comparison_note` lo explica. Las pruebas automáticas omiten este CSV cuando no está en el disco.

La telemetría de ese archivo proviene de la simulación de EA SPORTS F1 25. No es telemetría de un vehículo real de Fórmula 1.

## Limitaciones

- Las curvas son segmentos de telemetría: empiezan en una zona de frenado de la vuelta de referencia y terminan en la aceleración posterior. No son curvas oficiales ni una clasificación de la trazada.
- Las ventanas de frenada, curva y aceleración salen de la vuelta de referencia. La otra vuelta se mide sobre esas mismas distancias.
- Un evento fuera de la distancia común se omite.
- Frenada, curva y aceleración pueden solaparse. Se conservan las tres porque describen ventanas distintas.
- Los indicadores describen la telemetría de la ventana. No explican por qué el piloto ganó o perdió tiempo.
- La vuelta más rápida y la más lenta dependen de la sesión analizada, no de una longitud fija de circuito.
- No hay sectores oficiales, GPS, análisis de varios pilotos ni diagnóstico automático.

## Siguiente paso

El Proyecto 03 interpretará el rendimiento del piloto a partir de estos resultados. Este proyecto se detiene en la medición.
