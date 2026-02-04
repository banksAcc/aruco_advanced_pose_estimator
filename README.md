# BLE-Aruco Stylus Tracker: 6DOF Pose Estimation System

![Version](https://img.shields.io/badge/version-1.0.0-blue)
![License](https://img.shields.io/badge/license-All%20Rights%20Reserved-red)
![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux-lightgrey)

Sistema integrato per la stima della posa a 6 gradi di libertà (6DOF) di uno stilo basato su **icosaedro troncato**, con attivazione remota via **BLE** e pipeline di visione industriale **Basler**.
<p align="center">
<img src="src\debug\frame_000011_20260204_223833_overlay.png" alt="L'output del sistema" height="400">
</p>

---

## 📌 Panoramica del Sistema

Il progetto gestisce l'intero ciclo di vita del dato, dalla pressione di un pulsante fisico su un **ESP32** alla generazione di file di test (JSON/CSV) con la posizione millimetrica della punta della penna.



### Workflow Operativo
1.  **Connessione**: All'avvio, il PC (`main.py`) stabilisce un link BLE con l'ESP32 basandosi sulle impostazioni di `config.yaml`.
2.  **Trigger**: La pressione di un pulsante sull'ESP32 invia un comando al PC.
3.  **Acquisizione**: Il PC attiva la camera Basler, catturando una sequenza di frame gestiti in multi-threading.
4.  **Processamento**: Rilevamento ArUco, fusione delle pose e proiezione sulla punta (Tip).
5.  **Output**: Salvataggio di overlay grafici per debug, log CSV per analisi e matrici JSON per i test.

---

## 🏗️ Architettura del Software

### 1. Vision & Pose Estimation (`src/vision/api.py`)
La funzione core `estimate_truncated_ico_from_image` implementa la pipeline di visione:
* **Detection & Refinement**: Individuazione marker con raffinazione corner sub-pixel (`cornerSubPix`) per la massima precisione angolare.
* **Filtraggio Area**: Scarto automatico di marker troppo piccoli o distanti.
* **PnP Solving**: Risoluzione del problema *Perspective-n-Point* per ottenere la trasformazione $T_{cam \to marker}$.
* **Proiezione**: Trasformazione della posa di ogni marker nel centro dell'oggetto:
    $$T_{cam \to body} = T_{cam \to marker} \times T_{marker \to body}$$

### 2. Fusione Dati Robusta (`core/geometry/geometry.py`)
Il modulo `average_poses` fonde le molteplici ipotesi di posa in un unico dato stabile:
* **Angle Filter**: Scarta i marker visti con angoli troppo acuti dove l'errore sull'asse $Z$ aumenta esponenzialmente.
* **Weighted Mean**: Calcola la media pesata basata sull'area dei marker (esponente di peso personalizzabile in config).
* **Outlier Rejection**: Filtra i marker che si discostano eccessivamente dalla media preliminare.

### 3. Modellazione Geometrica (`snippet/gen_ico_transform.py`)
Script fondamentale per la generazione della **Mappa Geometrica Digitale**:
* Genera matematicamente i vertici dell'icosaedro troncato (12 pentagoni, 20 esagoni) usando la sezione aurea $\phi$.
* Applica offset manuali per compensare eventuali imprecisioni nel montaggio fisico dei marker.
* Esporta `transforms_final.json`, contenente le matrici $T_{body \to face}$ necessarie per la localizzazione.

---

## 📂 Struttura delle Cartelle

| Directory | Descrizione |
| :--- | :--- |
| `calibration/` | Script di calibrazione camera e generazione pattern ArUco (SVG). |
| `config/` | `config.yaml`: Il cervello del sistema (soglie, IP, parametri geometrici). |
| `src/core/` | Logica matematica, trasformazioni robotiche e motori di visione. |
| `src/services/` | Gestione hardware: client BLE e driver per camera Basler. |
| `src/snippet/` | Tool di utilità: generatore solido 3D e test rapidi via webcam. |

---

## 🛠️ Configurazione e Debug

Il sistema include un'interfaccia di debug avanzata tramite `core/vision/viz.py`:
* **Disegno Contorni**: Identifica i marker rilevati e previene il problema del "flipping".
* **Overlay punta**: Visualizza in tempo reale la stima della punta traslata lungo l'asse $Z$ locale.
* **Validazione**: Separa visivamente i dati validi dai disturbi ambientali.

> **Nota**: Se lo stilo viene riassemblato, è necessario rieseguire `gen_ico_transform.py` per aggiornare il modello matematico nel file `transforms_final.json`.

## License
All rights reserved.

This software and all associated files are the exclusive property of Angelo Milella - COMAU.
Unauthorized copying, modification, distribution, or use of this software, via any medium, is strictly prohibited.

For inquiries about licensing, please contact: <angelo_milella_dev@yahoo.com>.

---