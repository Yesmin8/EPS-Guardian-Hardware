"""
Configuration du système OBC
Tous les paramètres, seuils et constantes
"""

import os

# ============================================================================
# CONFIGURATION GÉNÉRALE
# ============================================================================

# Niveaux de log
LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR

# Communication série
SERIAL_PORT = "COM6"
SERIAL_BAUDRATE = 115200
SERIAL_TIMEOUT = 2.0

# ============================================================================
# CONFIGURATION IA
# ============================================================================

# Chemins des modèles
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "ai_model_lstm_autoencoder.h5")
SCALER_PATH = os.path.join(MODEL_DIR, "ai_sequence_scaler.pkl")

# Paramètres IA
SEQUENCE_LENGTH = 30
FEATURE_COUNT = 7  # v_batt, i_batt, t_batt, v_bus, i_bus, v_solar, i_solar

# Seuils de décision (MSE)
NORMAL_THRESHOLD = 0.6091226281233376
WARNING_THRESHOLD = 0.8377888951216337
CRITICAL_THRESHOLD = 1.0664551621199299

# ============================================================================
# CONFIGURATION EPS - SEUILS PHYSIQUES
# ============================================================================

# Batterie
BATTERY_CONFIG = {
    "VOLTAGE_NOMINAL": 12.0,      # V
    "VOLTAGE_MIN": 6.5,          # V - décharge profonde
    "VOLTAGE_MAX": 16.0,         # V - surtension
    "CURRENT_CHARGE_MAX": -2.0,  # A - charge max (négatif)
    "CURRENT_DISCHARGE_MAX": 2.0,# A - décharge max
    "TEMP_MIN": -10.0,           # °C - froid extrême
    "TEMP_MAX": 45.0,            # °C - surchauffe
    "TEMP_CRITICAL": 50.0,       # °C - danger
}

# Bus de puissance
BUS_CONFIG = {
    "VOLTAGE_NOMINAL": 12.0,
    "VOLTAGE_TOLERANCE": 0.5,    # V
    "CURRENT_MAX": 3.0,          # A
}

# Panneaux solaires
SOLAR_CONFIG = {
    "VOLTAGE_OPEN_CIRCUIT": 21.0, # V - tension à vide
    "VOLTAGE_MIN_OPERATING": 1.0, # V - minimum pour fonctionnement
    "CURRENT_SHORT_CIRCUIT": 3.0, # A - courant de court-circuit
    "CURRENT_MIN": 0.05,          # A - courant minimum détectable
    "POWER_MIN": 0.1,             # W - puissance minimum
}

# Capteurs (plages plausibles)
SENSOR_CONFIG = {
    "VOLTAGE_MIN": 0.1,          # V
    "VOLTAGE_MAX": 20.0,         # V
    "CURRENT_MIN": -1.0,         # A (peut être négatif pour charge)
    "CURRENT_MAX": 10.0,         # A
    "TEMP_MIN": -50.0,           # °C
    "TEMP_MAX": 100.0,           # °C
}

# ============================================================================
# CONFIGURATION DÉCISION
# ============================================================================

# Seuils de confiance
CONFIDENCE_THRESHOLDS = {
    "HIGH": 0.8,
    "MEDIUM": 0.6,
    "LOW": 0.4,
}

# Poids pour la décision hybride
DECISION_WEIGHTS = {
    "MCU": 0.3,     # Poids du MCU (temps réel)
    "OBC": 0.3,     # Poids de l'OBC (temporel)
    "DIAG": 0.4,    # Poids du diagnostic (expert)
}

# ============================================================================
# CONFIGURATION ACTIONS
# ============================================================================

# Délais d'exécution (secondes)
ACTION_TIMEOUTS = {
    "SAFE_MODE": 300,        # 5 minutes
    "POWER_SAVING": 600,     # 10 minutes
    "ISOLATE_LOAD": 60,      # 1 minute
    "MONITOR_CLOSELY": 900,  # 15 minutes
}

# Priorités d'action (1-10, 10 = plus haute)
ACTION_PRIORITIES = {
    "EMERGENCY_SHUTDOWN": 10,
    "ENTER_SAFE_MODE": 9,
    "ISOLATE_LOAD": 8,
    "SHUTDOWN_NON_CRITICAL": 7,
    "POWER_SAVING_MODE": 6,
    "SWITCH_TO_BATTERY": 5,
    "MONITOR_CLOSELY": 4,
    "IGNORE_SENSOR": 3,
    "MONITOR": 2,
    "NO_ACTION": 1,
}

# ============================================================================
# CONFIGURATION MACHINE À ÉTATS
# ============================================================================

# Temps minimum dans chaque état (secondes)
MIN_STATE_TIMES = {
    "BOOT": 10,
    "NOMINAL": 30,
    "DEGRADED": 60,
    "SAFE_MODE": 300,
    "EMERGENCY": 10,
    "RECOVERY": 30,
    "SHUTDOWN": 5,
}

# Conditions de transition
STATE_TRANSITION_CONDITIONS = {
    "NOMINAL_TO_DEGRADED": ["CONFIRMED_WARNING", "OBC_DETECTED_WARNING"],
    "DEGRADED_TO_SAFE": ["CONFIRMED_CRITICAL", "OBC_DETECTED_CRITICAL"],
    "SAFE_TO_EMERGENCY": ["BATTERY_OVERHEAT", "BATTERY_DEEP_DISCHARGE"],
    "EMERGENCY_TO_SAFE": ["NORMAL", "RECOVERY_START"],
    "SAFE_TO_RECOVERY": ["NORMAL", "HEALTHY_FOR_2MIN"],
    "RECOVERY_TO_NOMINAL": ["HEALTHY_FOR_5MIN"],
}

# ============================================================================
# CONFIGURATION MACHINE À ÉTATS OBC
# ============================================================================

# États OBC
OBC_STATES = {
    "BOOT": "Démarrage système",
    "IDLE": "En attente d'alerte",
    "ACCUMULATING": "Accumulation d'alertes",
    "ANALYZING": "Analyse de l'épisode",
    "DECIDING": "Prise de décision",
    "RECOVERY_MONITORING": "Surveillance récupération",
    "ESCALATION": "Mode urgence",
}

# Paramètres d'accumulation
ACCUMULATION_CONFIG = {
    "BUFFER_SIZE": 30,  # Nombre d'alertes pour un épisode
    "MIN_EPISODE_DURATION": 10,  # Durée minimale d'un épisode (secondes)
    "MAX_EPISODE_DURATION": 600,  # Durée maximale d'un épisode (secondes)
}

# Paramètres de récupération
RECOVERY_CONFIG = {
    "STABLE_THRESHOLD": 5,  # Nombre de messages NORMAL pour stabilité
    "MONITOR_TIMEOUT": 300,  # Timeout monitoring (secondes)
    "STABLE_TIMEOUT": 60,  # Durée de stabilité requise (secondes)
}

# Paramètres d'escalade
ESCALATION_CONFIG = {
    "MAX_RETRIES": 3,  # Nombre de tentatives avant escalade
    "ESCALATION_DELAY": 30,  # Délai avant escalade (secondes)
    "EMERGENCY_ACTIONS": ["FULL_SYSTEM_RESET", "EMERGENCY_SHUTDOWN"],
}

# ============================================================================
# CONFIGURATION JOURNALISATION
# ============================================================================

LOG_CONFIG = {
    "MAX_ENTRIES": 1000,
    "AUTO_SAVE_INTERVAL": 10,  # Sauvegarde toutes les N entrées
    "LOG_FILE": "obc_system_log.json",
    "BACKUP_COUNT": 3,
}

# ============================================================================
# CONSTANTES DE CALCUL
# ============================================================================

# Coefficients pour indicateurs dérivés
DERIVED_COEFFICIENTS = {
    "BATTERY_CAPACITY": 10.0,  # Ah
    "SOLAR_EFFICIENCY": 0.18,  # 18%
    "CHARGE_EFFICIENCY": 0.85, # 85%
    "DISCHARGE_EFFICIENCY": 0.90, # 90%
}

# ============================================================================
# AJOUTER LES VARIABLES MANQUANTES POUR COMPATIBILITÉ
# ============================================================================

# Variables nécessaires pour obc_fault_diagnosis.py
FEATURE_NAMES = [
    "v_batt",    # Tension batterie (V)
    "i_batt",    # Courant batterie (A)
    "t_batt",    # Température batterie (°C)
    "v_solar",   # Tension panneaux solaires (V)
    "i_solar",   # Courant panneaux solaires (A)
    "v_bus",     # Tension bus principal (V)
    "i_load"     # Courant charge (A)
]

# Variables nécessaires pour compatibilité (basées sur votre ancien code)
PHYSICAL_LIMITS = {
    "v_batt": {"min": 6.0, "max": 8.4, "nominal": 7.4},
    "i_batt": {"min": -5.0, "max": 5.0, "nominal": 0.0},
    "t_batt": {"min": -20, "max": 60, "nominal": 25},
    "v_solar": {"min": 0, "max": 20, "nominal": 16},
    "i_solar": {"min": 0, "max": 3, "nominal": 1.5},
    "v_bus": {"min": 6.0, "max": 8.4, "nominal": 7.4},
    "i_load": {"min": 0, "max": 2, "nominal": 0.5}
}

# ============================================================================
# VÉRIFICATIONS
# ============================================================================

def validate_config():
    """Valide la configuration"""
    errors = []
    
    # Vérifier les chemins de fichiers
    if not os.path.exists(MODEL_PATH):
        errors.append(f"Modèle IA introuvable: {MODEL_PATH}")
    
    if not os.path.exists(SCALER_PATH):
        errors.append(f"Scaler IA introuvable: {SCALER_PATH}")
    
    # Vérifier les seuils
    if not (NORMAL_THRESHOLD < WARNING_THRESHOLD < CRITICAL_THRESHOLD):
        errors.append("Les seuils MSE doivent être croissants")
    
    if BATTERY_CONFIG["VOLTAGE_MIN"] >= BATTERY_CONFIG["VOLTAGE_MAX"]:
        errors.append("VOLTAGE_MIN doit être < VOLTAGE_MAX")
    
    if BATTERY_CONFIG["TEMP_MIN"] >= BATTERY_CONFIG["TEMP_MAX"]:
        errors.append("TEMP_MIN doit être < TEMP_MAX")
    
    # Vérifier les poids de décision
    total_weight = sum(DECISION_WEIGHTS.values())
    if abs(total_weight - 1.0) > 0.01:
        errors.append(f"Les poids de décision doivent sommer à 1.0 (actuel: {total_weight})")
    
    return errors

CONFIG_ERRORS = validate_config()
if CONFIG_ERRORS:
    print("ERREURS DE CONFIGURATION:")
    for error in CONFIG_ERRORS:
        print(f"  - {error}")
# ============================================================================
# CONFIGURATION COMPATIBILITÉ OBC PROFESSIONNEL
# ============================================================================

OBC_CONFIG = {
    "serial_port": SERIAL_PORT,
    "baudrate": SERIAL_BAUDRATE,
    "timeout": SERIAL_TIMEOUT,
    "log_level": LOG_LEVEL,
    "max_buffer_size": 1000,
    "decision_lock_time": 300,  # 5 minutes
    "recovery_timeout": RECOVERY_CONFIG.get("MONITOR_TIMEOUT", 300),
    "recovery_stable_threshold": RECOVERY_CONFIG.get("STABLE_THRESHOLD", 5),
    "enable_rl": True,
    "rl_advisor_mode": True,
    "rl_min_confidence": 0.7,
    "auto_reconnect": True,
    "buffer_only_alerts": False,
}

# Variables nécessaires pour l'architecture modulaire
SYSTEM_STATES = {
    "BOOT": "Démarrage",
    "WAITING_MCU": "Attente MCU",
    "IDLE": "En attente",
    "ACCUMULATING": "Accumulation données",
    "ANALYZING": "Analyse en cours",
    "DECIDING": "Prise de décision",
    "RECOVERY_MONITORING": "Surveillance récupération",
    "ESCALATION": "Mode urgence"
}

# Constantes de décision
DECISION_CONFIDENCE_THRESHOLD = 0.6
MIN_SEQUENCE_FOR_ANALYSIS = 20
MAX_EPISODE_DURATION = 60  # secondes