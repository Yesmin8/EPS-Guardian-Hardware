# EPS Guardian — Hardware Demonstration  
**Hybrid AI-Based EPS Anomaly Surveillance for CubeSats**


![IEEE Challenge](https://img.shields.io/badge/IEEE-AESS%20%26%20IES%202025-blue)
![Status](https://img.shields.io/badge/Status-Hardware%20Validated-success)


##  Project Overview

**EPS Guardian** is a hybrid hardware–software system designed to monitor and analyze anomalies in a CubeSat **Electrical Power System (EPS)**.  
This repository focuses on the **hardware demonstration**, validating that the proposed AI-based supervision architecture can operate under real embedded constraints.

The system follows a **Pure Observer philosophy**:

- Detects, analyzes, and reports anomalies  
- **Never applies autonomous corrective actions on hardware**

This design choice guarantees **scientific reproducibility**, **full traceability**, and a **safe separation between detection and action**, which is essential for critical space systems.



##  System Philosophy

> **Detect fast. Decide safely. Act deliberately.**


##  System Architecture

### MCU (ESP32)

- Real-time monitoring  
- Deterministic safety rules  
- Lightweight Autoencoder using **TensorFlow Lite Micro**  
- Immediate alert transmission  

### OBC (PC Simulation)

- Temporal validation over **30-sample windows**  
- LSTM Autoencoder for long-term pattern analysis  
- Decision fusion and **advisory actions only**

 **No automatic recovery is applied on hardware.**  
All actions remain suggestions or simulated commands due to hardware limitations.


## Repository Structure
```bash
EPS-Guardian-Hardware/
│
├── eps_guardian_mcu/
│ ├── eps_guardian_mcu.ino # ESP32 firmware
│ ├── eps_guardian_ai_model.h # Quantized AI model (TFLite Micro)
│ ├── mcu_output.txt # Real MCU execution logs
│
├── obc/
│ ├── obc_main.py # OBC main loop
│ ├── obc_ai_inference.py # LSTM inference
│ ├── obc_decision_engine.py # Decision logic
│ ├── obc_state_machine.py # OBC FSM
│ ├── obc_action_manager.py # Simulated actions
│ ├── obc_output.txt # OBC execution logs
│ └── models/
│ ├── ai_model_lstm_autoencoder.h5
│ ├── ai_sequence_scaler.pkl
│
├── wiring/
│ └── eps_guardian_breadboard.png # Hardware wiring diagram
│
└── README.md
```
## Physically Emulated Virtual Sensors (Computed Onboard the MCU)

In addition to physical sensors, **EPS Guardian** computes a set of
**physically emulated virtual sensors directly on the MCU**.
These variables are derived from simplified yet physically consistent
electrical models and are used **exclusively for anomaly detection and
diagnosis**, not for control.

All virtual sensors are **deterministic**, **fixed**, and **non-adaptive**,
ensuring reproducibility and traceability.

### Virtual Sensor Definitions

| Variable | Description |
|--------|-------------|
| `vbatt` | Fixed 2S Li-ion battery electrical model (virtual) |
| `ibatt` | Battery current derived from energy balance (virtual) |
| `SOC` | Coulomb counting with capacity constraint (virtual) |
| `voc` | Open-circuit voltage estimated from an SOC lookup table (virtual) |
| `psolar` | Solar power estimation (*p = v · i*) (virtual) |
| `pbatt`, `pbus` | Battery and EPS bus power computation (*p = v · i*) (virtual) |
| `vbus`, `ibus` | Simplified EPS bus regulation model outputs (virtual) |

### Design Rationale

The use of physically emulated virtual sensors provides several advantages:

- Enables **rich state observability** without increasing hardware complexity  
- Allows early detection of **energy imbalance and degradation patterns**  
- Reduces dependency on additional physical sensors  
- Preserves a **pure observer architecture** with no feedback control  

These virtual variables are treated exactly like physical measurements
within the anomaly detection pipeline and are logged and transmitted
to the OBC for temporal validation.

**Important:**  
Virtual sensors are **never used for real-time control or actuation**.
They serve only for **monitoring, diagnosis, and scientific analysis**.

---

## Hardware Components

| Component        | Role |
|------------------|------|
| ESP32            | MCU (edge intelligence) |
| INA219           | Voltage & current sensing |
| DS18B20          | Battery temperature monitoring |
| Potentiometers  | Solar and load simulation |
| LEDs (G / Y / R) | System state visualization |
| Breadboard       | Test bench integration |

## Hardware Wiring

The hardware setup used for the EPS Guardian demonstration includes:

- ESP32 as the central MCU  
- INA219 connected via **I²C**  
- DS18B20 on a **OneWire** bus  
- Potentiometers simulating solar input and load  
- LEDs indicating system state (**NORMAL / WARNING / CRITICAL**)

Wiring diagram available in:  
`wiring/eps_guardian_breadboard.png`




## MCU Behavior (ESP32)

### Key Characteristics

- Fixed physical battery model (**R_int = 80 mΩ**)  
- No online calibration  
- No model adaptation  
- Fully deterministic behavior  


### Detection Mechanisms

#### Deterministic Rules

- **R1**: Over-voltage / under-voltage (extreme values)  
- **R2**: Overcurrent / bus overload  
- **R3**: Undervoltage under discharge (low SOC)  

#### AI Autoencoder

- Monitors reconstruction error (**MSE**)  
- Detects subtle deviations and progressive anomalies  


### Outputs

- CSV logs generated every **2 seconds**  
- JSON packets sent to the OBC upon alerts  
- LED indication:
  - 🟢 **NORMAL**
  - 🟡 **WARNING**
  - 🔴 **CRITICAL**

## OBC Behavior (PC Simulation)

The OBC acts as a higher-level intelligence layer and **never reacts to isolated samples**.

### OBC Logic

- Receives alerts from the MCU  
- Accumulates **30 consecutive samples**  
- Performs temporal analysis using an **LSTM Autoencoder**  
- Confirms or rejects the anomaly  
- Generates **advisory actions only**

### Example Final Decision

Final decision: CRITICAL_RULE_R2
Fault type: BUS_OVERLOAD
Confidence: 90%

This decision may differ from the initial MCU trigger  
(e.g., R3 detected locally and R2 confirmed globally), demonstrating **multi-level diagnosis consistency**.

##  WARNING Messages (Key Design Choice)

EPS Guardian explicitly introduces a **WARNING** state:

- Indicates suspicious or early-stage behavior  
- Does not require immediate intervention  
- Designed for progressive anomalies  
- Reduces false critical escalations  

WARNING states are evaluated temporally and may:

- **Disappear** → downgraded  
- **Persist** → escalated to CRITICAL by the OBC  

##  Experimental Results (Hardware)

- Stable ESP32 execution with TFLite Micro  
- Average inference latency: **< 100 ms**  
- RAM usage: **~40 KB**  
- Reliable serial communication between MCU and OBC  
- Successful detection of a progressive undervoltage scenario  
- Correct escalation after temporal validation  

The files `mcu_output.txt` and `obc_output.txt` provide **full traceability** of all decisions.

## Safety & Reproducibility

- Fixed models  
- No hidden adaptation  
- Complete logs  
- Explainable decisions  
- Clear separation between MCU and OBC responsibilities  

This design is **defensible, auditable, and space-compatible**.

##  Project Status

- Software simulation validated  
- Hardware-in-the-loop demonstration completed  

**Future work**: integration with a real EPS control board  

**Technology Readiness Level:** TRL 6–7

## Final Note

**EPS Guardian** demonstrates that hybrid AI-based supervision can be deployed on constrained embedded hardware while preserving **safety**, **transparency**, and **scientific rigor**.

## Citation

If you use this work in academic research, please cite:

EPS Guardian – Hybrid AI-Based EPS Anomaly Surveillance for CubeSats  
IEEE AESS & IES Challenge 2025