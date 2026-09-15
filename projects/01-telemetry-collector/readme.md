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

La arquitectura inicial del proyecto es:

F1 25
   │
   │ UDP
   ▼
UDP Receiver
   │
   ▼
Packet Parser
   │
   ▼
Normalized Telemetry
   ├── JSON
   └── CSV

La arquitectura será ampliada progresivamente durante las siguientes fases.

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