# D405 Gussbauteil-Scanner – Professionelle Qualitätskontrolle

Vollständiges Python-Projekt für professionelle 3D-Scans mit dem Intel RealSense D405.
Optimiert für Gussbauteile mit Oberflächenvorbereitung, maximaler Tiefenqualität,
Punktwolken-Registrierung und CAD-Abstandsanalyse.

## Features

- Optimierte D405-Kameraparameter für Nahbereich (15–25 cm)
- Automatische Multi-Frame-Mittelung (Temporal Filtering)
- Drehteller-gesteuerter 360°-Scan-Workflow
- ICP-Registrierung mehrerer Einzelscans
- SOR-Outlier-Entfernung und Poisson-Mesh-Rekonstruktion
- CAD-Referenzvergleich (Cloud-to-Mesh) mit Heatmap
- HTML-QK-Report mit allen Abweichungen
- Vollständige Logging- und Konfigurations-Pipeline

## Projektstruktur

```
d405_scanner/
├── main.py                  # Haupteinstieg – kompletter Scan-Workflow
├── config/
│   └── settings.py          # Alle Parameter zentral konfigurierbar
├── core/
│   ├── camera.py            # D405-Kamera-Initialisierung & Streaming
│   └── capture.py           # Frame-Aufnahme, Mittelung, Filterung
├── processing/
│   ├── pointcloud.py        # Punktwolken-Verarbeitung & Registrierung
│   └── mesh.py              # Mesh-Rekonstruktion (Poisson)
├── analysis/
│   └── quality.py           # CAD-Vergleich, Abstandsanalyse, Heatmap
├── utils/
│   ├── logger.py            # Strukturiertes Logging
│   ├── visualizer.py        # Live-Vorschau & 3D-Visualisierung
│   └── exporter.py          # STL/PLY/OBJ Export
├── output/                  # Scan-Ergebnisse (automatisch befüllt)
└── tests/
    └── test_pipeline.py     # Unit-Tests ohne echte Kamera (Mocks)
```

## Installation

```bash
pip install pyrealsense2 open3d numpy scipy matplotlib pillow jinja2
```

## Schnellstart

```bash
# Einzelnen Scan aufnehmen
python main.py --mode single

# 360°-Drehteller-Scan (8 Positionen)
python main.py --mode turntable --positions 8

# Scan gegen CAD-Referenz prüfen
python main.py --mode compare --reference mein_bauteil.stl

# Vollständiger Workflow (Scan + Analyse + Report)
python main.py --mode full --reference mein_bauteil.stl --positions 12
```

## Realistische Genauigkeit

| Setup               | Typische Genauigkeit |
|---------------------|----------------------|
| Freihand            | ±1.5–3.0 mm          |
| Stativ, kein Spray  | ±0.8–1.5 mm          |
| Stativ + AESUB      | ±0.4–0.8 mm          |
| Stativ + AESUB + ICP| ±0.3–0.5 mm          |

Für Toleranzen < 0.2 mm: dedizierter Strukturlichtscanner empfohlen.
