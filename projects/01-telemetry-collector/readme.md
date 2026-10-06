# F1 25 Telemetry Collector

Collector de telemetría UDP para EA SPORTS F1 25.

## Descripción

Este proyecto recibe y procesa los paquetes de telemetría enviados por
EA SPORTS F1 25 mediante el protocolo UDP.

El objetivo inicial es construir desde cero un sistema capaz de:

- recibir paquetes UDP;
- identificar el tipo de paquete;
- interpretar su encabezado binario;
- decodificar estructuras de telemetría;
- convertir los datos a estructuras manejables;
- registrar posteriormente la información en formatos como JSON y CSV.

Este proyecto utiliza exclusivamente la telemetría proporcionada por el
videojuego F1 25. No representa telemetría obtenida directamente de un
monoplaza real de Fórmula 1.

---

## Arquitectura

F1 25 envía la telemetría por UDP. El collector conserva el Packet 6
(Car Telemetry) y el Packet 2 (Lap Data) del mismo frame y escribe un
CSV compatible con el telemetry analyzer.

```text
F1 25
  │
  │ UDP
  ▼
UDP Receiver
  │
  ▼
Packet Header
  ├── Packet ID 6 — Car Telemetry
  └── Packet ID 2 — Lap Data
          │
          ▼
Sincronización
session_uid + overall_frame_identifier + player_car_index
          │
          ▼
Fila completa
          │
          ▼
CSV incremental
          │
          ▼
Telemetry Analyzer
```

El JSON de la misma sesión se escribe al cerrar el collector con Ctrl+C.

---

## Entorno

### Hardware

- PlayStation ejecutando EA SPORTS F1 25
- PC Windows 11 ejecutando el collector
- Ambos dispositivos conectados a la misma red local

### Software

- Windows 11
- Python 3.12.3
- VS Code
- Git

---

## Configuración de F1 25

Configuración utilizada durante las pruebas:

| Parámetro | Valor |
|---|---|
| UDP Telemetry | On |
| Broadcast | Off |
| IP | 192.168.1.14 |
| Port | 20777 |
| Send Rate | 20 |
| Format | 2025 |

La dirección IP corresponde al PC que recibe la telemetría.

La dirección IP puede cambiar dependiendo de la configuración de red del
equipo. Si cambia, debe actualizarse también la configuración UDP dentro
de F1 25.

---

## Protocolo

F1 25 utiliza UDP para transmitir los datos de telemetría.

Los paquetes contienen un encabezado común que permite identificar,
entre otros datos, el formato del paquete, la versión del juego, la
versión del paquete y el identificador del tipo de paquete.

El proyecto utilizará el `Packet ID` para determinar cómo interpretar
el contenido específico de cada paquete.

Los datos binarios utilizan Little Endian.

---

## Dataset para el analyzer

El CSV de una captura real queda en:

`data/telemetry/telemetry_session.csv`

Se crea al iniciar el collector. El encabezado se escribe una vez y cada
fila completa se agrega en ese momento. Una fila solo existe cuando el
Packet 6 y el Packet 2 comparten la misma clave:

- `session_uid`
- `overall_frame_identifier`
- `player_car_index`

`session_uid` se usa solo para esa asociación. No es una columna del CSV.
`session_time` tampoco es clave: es el tiempo que trae el Packet 6.

UDP puede entregar un frame más nuevo antes que el anterior. El collector
no escribe ese frame hasta que el anterior llega, o hasta que han pasado
8 frames más nuevos. A 20 Hz esa ventana dura 0.4 s. El caso real de F1 25
reordenaba como máximo 2 frames, por ejemplo `1285` antes de `1284`.
Si el frame anterior no aparece dentro de la ventana, se conserva el salto
y no se inventa la muestra. El buffer de 64 frames sigue siendo solo para
parejas incompletas: un Packet 2 o un Packet 6 que todavía no tiene su par.

Si `m_lapDistance` es negativo, esa muestra no se escribe. No se convierte
a cero. `lap_number` sale de `m_currentLapNum`. `frame` es
`overall_frame_identifier`, no `frame_identifier`.

| Columna | Origen | Significado |
|---|---|---|
| session_time | Packet 6, header | Segundos de la sesión |
| frame | header `overall_frame_identifier` | Frame global del paquete |
| lap_number | Packet 2, `m_currentLapNum` | Número de vuelta del juego |
| lap_distance | Packet 2, `m_lapDistance` | Distancia de la vuelta, en metros |
| speed | Packet 6 | Velocidad, en km/h |
| throttle | Packet 6 | Acelerador, de 0 a 1 |
| brake | Packet 6 | Freno, de 0 a 1 |
| steering | Packet 6 | Dirección, de -1 a 1 |
| gear | Packet 6 | -1 reversa, 0 neutro, 1 a 8 marchas |
| rpm | Packet 6, `engine_rpm` | Revoluciones del motor |
| drs | Packet 6 | 0 desactivado, 1 activado |

### Captura manual

Desde `projects/01-telemetry-collector`, con el entorno del proyecto:

```text
.\.venv\Scripts\python.exe -m telemetry_collector.main
```

En F1 25 la telemetría UDP debe apuntar a la IP de este PC, puerto 20777,
formato 2025. Conduce al menos una vuelta y detén el collector con Ctrl+C.

### Uso en el analyzer

El archivo no sustituye a los CSV sintéticos de
`projects/02-telemetry-analyzer/data/sample/`. Esos siguen siendo fixtures
de prueba.

Copia `telemetry_session.csv` a
`projects/02-telemetry-analyzer/data/real/` si quieres conservarlo aparte
del collector. Desde el analyzer:

```python
from telemetry_analyzer.ingestion.csv_loader import load_csv
from telemetry_analyzer.processing.validation import validate_dataset
from telemetry_analyzer.processing.normalization import normalize_dataset
from telemetry_analyzer.processing.laps import segment_laps
from telemetry_analyzer.metrics.basic import calculate_basic_metrics

dataset = load_csv("data/real/telemetry_session.csv")
validate_dataset(dataset)
normalized = normalize_dataset(dataset)
laps = segment_laps(normalized)

for lap in laps.values():
    print(lap.lap_number, calculate_basic_metrics(lap))
```

Una sesión nueva de F1 25 dentro de la misma ejecución del collector
reinicia `session_time`. El analyzer exige que esa columna no retroceda,
así que conviene una ejecución del collector por sesión.

---

## Primer paquete capturado

Durante la fase inicial se realizó una prueba real con F1 25.

El PC recibió correctamente un paquete UDP enviado por el juego.

Tamaño del paquete:

45 bytes

El paquete fue almacenado temporalmente como:

`first_packet.bin`

Los primeros bytes capturados fueron:

```text
e9 07 19 01 19 01 03 66 55 b4 71 f7 66 29 ef 0c
12 81 42 f1 04 00 00 f1 04 00 00 13 ff 42 55 54 4e
04 00 00 00 00 00 00 00 00 00 00 00 00 00