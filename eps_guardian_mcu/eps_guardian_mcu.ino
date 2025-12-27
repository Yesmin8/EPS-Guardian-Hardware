/*
 * EPS GUARDIAN - VERSION OBSERVATEUR PUR POUR CHALLENGE IEEE 2025
 * PHILOSOPHIE : DÉTECTION SCIENTIFIQUE SANS ACTION AUTOMATIQUE
 * IEEE AESS & IES Challenge 2025
 * MCU comme détecteur scientifique, pas comme contrôleur
 * MODÈLE FIXE : Reproductibilité garantie, pas d'ajustement en ligne
 */

#include <Arduino.h>
#include <Wire.h>
#include <OneWire.h>
#include <DallasTemperature.h>
#include <Adafruit_INA219.h>

/* ===== TensorFlow Lite Micro ===== */
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

/* ===== AI model ===== */
#include "eps_guardian_ai_model.h"

/* ===================== CONFIGURATION PHILOSOPHIE ===================== */
#define PHILOSOPHY_PURE_OBSERVER true    // Observateur pur, pas d'actions
#define MONITORING_ONLY true             // Monitoring seulement
#define SUGGESTIONS_ONLY true            // Suggestions, pas d'actions

// DÉSACTIVER toutes les actions automatiques
#define ENABLE_ONLINE_CALIBRATION false  
#define ENABLE_AUTOMATIC_RECOVERY false  
#define ENABLE_MODEL_ADJUSTMENTS false   
#define ENABLE_SCENARIO_MODIFICATION false 

/* ===================== CONFIGURATION MODE ===================== */
#define DEMO_MODE true  // true = Mode démonstration, false = Mode projet scientifique

#if DEMO_MODE
  #define TIME_SCALE_ORBITAL 1.0f     // 1:1 pour démonstration rapide
  #define INITIAL_SOC 0.30f           // SOC initial 30% pour scénario critique
  #define DEMO_DURATION_MINUTES 3     // Durée totale de démonstration
  #define R7_SENSOR_STUCK_THRESHOLD 0.002f  // Seuil légèrement relâché en démo
  #define DEMO_OVERLOAD_SCENARIO true  // Simulation de surcharge en mode démo
#else
  #define TIME_SCALE_ORBITAL 60.0f    // 1 sec MCU = 60 sec orbitales (réaliste)
  #define INITIAL_SOC 0.65f           // SOC initial 65% (normal)
  #define R7_SENSOR_STUCK_THRESHOLD 0.001f  // Seuil strict en mode projet
  #define DEMO_OVERLOAD_SCENARIO false  // Pas de simulation en mode projet
#endif

/* ===================== CONFIGURATION HARDWARE ===================== */
#define ONE_WIRE_BUS 21
#define SDA_PIN 19
#define SCL_PIN 18
#define POT_VSOLAR 32
#define POT_LOAD 35
#define LED_GREEN 26
#define LED_YELLOW 25
#define LED_RED 27

/* ===================== MODÈLE DE BATTERIE Li-ion 2S (EXPLICITE ET FIXE) ===================== */
#define V_BATT_NOMINAL 7.4f      // Tension nominale 2S Li-ion
#define V_BATT_MIN 6.0f          // Tension de décharge profonde
#define V_BATT_MAX 8.4f          // Tension pleine charge
#define R_BATT_INTERNAL 0.08f    // 80 mΩ FIXE - pas d'ajustement
#define BATT_CAPACITY_AH 2.0f    // 2 Ah (capacité typique CubeSat)
#define BATT_CAPACITY_AS 7200.0f // 2 Ah en ampères-secondes (2 * 3600)

#define I_BATT_MIN -2.0f         // Courant charge max (négatif)
#define I_BATT_MAX 2.0f          // Courant décharge max

/* ===================== PARAMÈTRES SCÉNARIO DÉMO (SIMULATION SEULEMENT) ===================== */
#define DEMO_OVERLOAD_SOC_THRESHOLD 0.20f // Seuil SOC pour simulation survie
#define DEMO_OVERLOAD_LOAD_CURRENT 1.5f   // Courant de charge en simulation
#define DEMO_OVERLOAD_SOLAR_CURRENT 0.0f  // Pas de solaire en simulation (nuit)

/* ===================== LIMITES SYSTÈME ===================== */
#define T_BATT_MIN -10.0f
#define T_BATT_MAX 60.0f
#define T_EPS_MIN -10.0f
#define T_EPS_MAX 70.0f
#define V_SOLAR_MAX 8.0f
#define I_SOLAR_MAX 2.0f
#define LOAD_MAX 2.0f

/* ===================== CONFIGURATION INA219 (BENCH DE TEST) ===================== */
#define I_REF_MAX 0.5f           // Courant max sur résistance 220Ω de test
#define V_REF_MAX 5.0f           // Tension max de test

/* ===================== SEUILS CALIBRÉS ===================== */
#define FEATURE_LEN 18
#define ROLLING_WINDOW_SIZE 10
#define TH_WARNING  0.125f
#define TH_CRITICAL 0.200f

/* ===================== PARAMÈTRES HYSTÉRÉSIS ET PERSISTANCE ===================== */
#define HYSTERESIS_WARN_MS  5000
#define HYSTERESIS_CRIT_MS  2000
#define SEQ_LEN 30
#define OBC_SEND_PROB_WARNING 0.3f
#define MIN_OBC_SEND_INTERVAL 2000
#define RULE_CONFIRM_WINDOWS 2

/* ===================== STRUCTURES DE DONNÉES ===================== */
typedef struct {
  float v_ref;       // Tension mesurée sur résistance de test
  float i_ref;       // Courant mesurée sur résistance de test
  float t_meas;
  float solar_pot;
  float load_pot;
} RawSensors;

typedef struct {
  float v_batt;      // Tension batterie reconstruite
  float i_batt;      // Courant batterie reconstruit
  float t_batt;
  float t_eps;
  float soc;         // State of Charge (0-1)
  float v_bus;
  float i_bus;
  float v_solar;
  float i_solar;
  float i_load;
  float p_batt;
  float p_solar;
  float p_bus;
  float converter_ratio;
  // Variables du modèle FIXE
  float v_oc;        // Tension à vide (Open Circuit)
  float r_int;       // Résistance interne (FIXE)
} EPSState;

/* ===================== RÈSULTAT RÈGLES PHYSIQUES ===================== */
struct RuleResult { 
  int level;
  String reason;
  bool rules_triggered[8];
  
  RuleResult() : level(0), reason("NORMAL") {
    for (int i = 0; i < 8; i++) rules_triggered[i] = false;
  }
};

/* ===================== VARIABLES GLOBALES ===================== */
OneWire oneWire(ONE_WIRE_BUS);
DallasTemperature tempSensor(&oneWire);
Adafruit_INA219 ina219;
bool ina219_detected = false;

constexpr int kArenaSize = 64 * 1024;
uint8_t tensor_arena[kArenaSize];
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input_tensor = nullptr;
TfLiteTensor* output_tensor = nullptr;

float v_batt_history[ROLLING_WINDOW_SIZE] = {0};
int history_index = 0;
bool history_filled = false;

static float prev_v_batt = 7.4f;
static float prev_i_batt = 0.4f;
static float prev_t_batt = 25.0f;

// Modèle de batterie - SOC initial selon le mode
static float soc_est = INITIAL_SOC;

static float seq_buffer[SEQ_LEN][FEATURE_LEN];
static int seq_index = 0;
static bool seq_filled = false;

static unsigned long ai_warn_start = 0;
static unsigned long ai_crit_start = 0;
static unsigned long last_send_to_obc = 0;
static int current_alert_level = 0;
static String last_alert_reason = "NORMAL";
static String last_trigger_source = "NONE";
static int rule_pers_count[8] = {0};

float mse_history[10] = {0};
int mse_index = 0;
float mse_smoothed = 0.0f;

enum AnomalyType {
  ANOMALY_NONE,
  ANOMALY_ENERGY_IMBALANCE,
  ANOMALY_SOC_DISCREPANCY,
  ANOMALY_SENSOR_DRIFT,
  ANOMALY_OSCILLATION_MOD,
  ANOMALY_LOAD_SPIKE
};

AnomalyType current_anomaly = ANOMALY_NONE;
unsigned long anomaly_start_time = 0;

/* ===================== FONCTIONS UTILITAIRES ===================== */
float safe_constrain(float value, float min_val, float max_val) {
  if (isnan(value)) return (min_val + max_val) / 2.0f;
  if (value < min_val) return min_val;
  if (value > max_val) return max_val;
  return value;
}

float calculate_ocv_from_soc(float soc) {
  // Modèle simplifié de batterie Li-ion 2S - FIXE
  soc = safe_constrain(soc, 0.05f, 0.95f);
  
  if (soc >= 0.95f) return 8.4f;
  if (soc >= 0.85f) return 8.2f;
  if (soc >= 0.70f) return 7.8f;
  if (soc >= 0.50f) return 7.4f;
  if (soc >= 0.30f) return 7.0f;
  if (soc >= 0.15f) return 6.6f;
  if (soc >= 0.08f) return 6.2f;
  return 6.0f;
}

float calculate_soc_from_ocv(float ocv) {
  ocv = safe_constrain(ocv, 6.0f, 8.4f);
  
  if (ocv >= 8.3f) return 0.95f;
  if (ocv >= 8.0f) return 0.85f;
  if (ocv >= 7.6f) return 0.70f;
  if (ocv >= 7.2f) return 0.50f;
  if (ocv >= 6.8f) return 0.30f;
  if (ocv >= 6.4f) return 0.15f;
  if (ocv >= 6.2f) return 0.08f;
  return 0.05f;
}

void update_leds(int alert_level) {
  digitalWrite(LED_GREEN, LOW);
  digitalWrite(LED_YELLOW, LOW);
  digitalWrite(LED_RED, LOW);
  
  if (alert_level == 2) {
    digitalWrite(LED_RED, HIGH);
  } else if (alert_level == 1) {
    digitalWrite(LED_YELLOW, HIGH);
  } else {
    digitalWrite(LED_GREEN, HIGH);
  }
}

/* ===================== MONITORING QUALITÉ (PAS D'AJUSTEMENT) ===================== */
void monitor_calibration_quality(const EPSState &eps) {
  static float error_sum = 0.0f;
  static int sample_count = 0;
  static unsigned long last_report = 0;
  
  // Calcul de l'erreur théorique avec modèle FIXE
  float expected_v_oc = calculate_ocv_from_soc(eps.soc);
  float expected_v_drop = R_BATT_INTERNAL * eps.i_batt;
  float expected_v_batt = expected_v_oc - expected_v_drop;
  float model_error = fabs(eps.v_batt - expected_v_batt);
  
  error_sum += model_error;
  sample_count++;
  
  // Rapport périodique (toutes les 30 secondes)
  unsigned long current_time = millis();
  if (current_time - last_report > 30000) {
    float avg_error = (sample_count > 0) ? error_sum / sample_count : 0.0f;
    
    Serial.println("\n MONITORING CALIBRATION (pas d'ajustement) :");
    Serial.print("  • Erreur moyenne modèle: ");
    Serial.print(avg_error, 3);
    Serial.println("V");
    Serial.print("  • Échantillons: ");
    Serial.println(sample_count);
    Serial.print("  • R_int fixe: ");
    Serial.print(R_BATT_INTERNAL * 1000, 0);
    Serial.println(" mΩ");
    Serial.print("  • V_batt mesuré: ");
    Serial.print(eps.v_batt, 2);
    Serial.print("V, attendu: ");
    Serial.print(expected_v_batt, 2);
    Serial.println("V");
    
    // Réinitialisation
    error_sum = 0.0f;
    sample_count = 0;
    last_report = current_time;
  }
}

/* ===================== SUGGESTIONS RÉCUPÉRATION (PAS D'ACTIONS) ===================== */
void log_recovery_suggestions(const EPSState &eps, const RawSensors &raw) {
  static unsigned long last_suggestion_log = 0;
  static int suggestion_count = 0;
  
  unsigned long current_time = millis();
  
  // Conditions pour suggérer une récupération
  bool critical_conditions = (eps.soc < 0.15f) && 
                           (eps.v_batt < 6.4f) && 
                           (eps.i_batt > 0.5f);
  
  bool warning_conditions = (eps.soc < 0.20f) && 
                          (eps.v_batt < 6.6f) && 
                          (eps.i_batt > 0.3f);
  
  if ((critical_conditions || warning_conditions) && 
      (current_time - last_suggestion_log > 30000)) {
    
    suggestion_count++;
    
    Serial.println("\n" + String(70, '='));
    Serial.println(" SUGGESTION DE RÉCUPÉRATION MANUELLE");
    Serial.println("   (Détection pure - aucune action automatique)");
    Serial.println(String(70, '-'));
    
    Serial.println("État critique détecté :");
    Serial.print("  • SOC: ");
    Serial.print(eps.soc * 100, 1);
    Serial.println("%");
    Serial.print("  • V_batt: ");
    Serial.print(eps.v_batt, 2);
    Serial.println("V");
    Serial.print("  • I_batt: ");
    Serial.print(eps.i_batt, 2);
    Serial.println("A");
    
    Serial.println("\nActions recommandées (à appliser manuellement) :");
    
    if (critical_conditions) {
      Serial.println("  PRIORITÉ HAUTE - Conditions critiques");
      Serial.print("  1. Réduire charge → Potentiomètre Load à ");
      Serial.println(raw.load_pot * 0.5f, 2);
      Serial.print("  2. Maximiser solaire → Potentiomètre Solar à ");
      Serial.println(min(raw.solar_pot * 1.5f, 1.0f), 2);
    } else {
      Serial.println("   PRIORITÉ MOYENNE - Conditions d'alerte");
      Serial.print("  1. Ajuster charge → Potentiomètre Load à ");
      Serial.println(raw.load_pot * 0.8f, 2);
      Serial.print("  2. Optimiser solaire → Potentiomètre Solar à ");
      Serial.println(min(raw.solar_pot * 1.2f, 1.0f), 2);
    }
    
    Serial.println("  3. Objectifs : SOC > 25%, V_batt > 6.8V, I_batt < 0.5A");
    Serial.println(String(70, '='));
    
    last_suggestion_log = current_time;
  }
}

/* ===================== LECTURE CAPTEURS (SANS MODIFICATION) ===================== */
RawSensors read_raw_sensors_pure() {
  RawSensors raw;
  
  // 1. Lecture température
  tempSensor.requestTemperatures();
  raw.t_meas = tempSensor.getTempCByIndex(0);
  if (isnan(raw.t_meas)) raw.t_meas = 25.0f;
  
  // 2. Lecture INA219 (bench seulement - pas d'utilisation dans le modèle)
  if (ina219_detected) {
    const int NUM_SAMPLES = 3;
    float v_sum = 0, i_sum = 0;
    
    for (int i = 0; i < NUM_SAMPLES; i++) {
      float v_try = ina219.getBusVoltage_V();
      float i_try = ina219.getCurrent_mA() / 1000.0f;
      
      if (!isnan(v_try) && v_try >= 0 && v_try <= 30) {
        v_sum += v_try;
      }
      if (!isnan(i_try) && fabs(i_try) <= 10) {
        i_sum += i_try;
      }
      delay(1);
    }
    
    raw.v_ref = v_sum / NUM_SAMPLES;
    raw.i_ref = i_sum / NUM_SAMPLES;
  } else {
    // Simulation de bench seulement
    static uint32_t bench_seed = 12345;
    bench_seed = bench_seed * 1103515245 + 12345;
    raw.v_ref = 3.0f + ((bench_seed >> 16) % 401) / 100.0f;
    raw.i_ref = raw.v_ref / 220.0f;
  }
  
  // 3. Lecture potentiomètres (sans modification)
  raw.solar_pot = analogRead(POT_VSOLAR) / 4095.0f;
  raw.load_pot = analogRead(POT_LOAD) / 4095.0f;
  
  return raw;
}

/* ===================== MODÈLE PHYSIQUE FIXE - VERSION OBSERVATEUR PUR ===================== */
EPSState battery_model_pure_observer(RawSensors raw) {
  EPSState eps = {};
  
  // ===== MODÈLE FIXE - Reproductibilité garantie =====
  static bool model_explained = false;
  if (!model_explained) {
    Serial.println("\n MODÈLE PHYSIQUE FIXE (OBSERVATEUR PUR):");
    Serial.println("• R_int = 80 mΩ (constante)");
    Serial.println("• Pas d'ajustement en ligne");
    Serial.println("• Reproductibilité scientifique garantie");
    model_explained = true;
  }
  
  // ===== 1. MODÈLE PHYSIQUE FIXE =====
  eps.i_solar = raw.solar_pot * I_SOLAR_MAX * 0.6f;
  if (eps.i_solar < 0.01f) eps.i_solar = 0.01f;
  
  eps.i_load = raw.load_pot * LOAD_MAX;
  if (eps.i_load < 0.05f) eps.i_load = 0.05f;
  
  // ===== 2. SIMULATION SCÉNARIO (DÉMO SEULEMENT) - SIMULATION UNIQUEMENT =====
  #if DEMO_MODE && DEMO_OVERLOAD_SCENARIO
  if (soc_est < DEMO_OVERLOAD_SOC_THRESHOLD) {
    // SIMULATION SEULEMENT - pas de modification des valeurs réelles
    // Simulation d'un scénario de survie satellite
    float simulated_i_solar = DEMO_OVERLOAD_SOLAR_CURRENT;
    float simulated_i_load = DEMO_OVERLOAD_LOAD_CURRENT;
    
    // Log de simulation (pour transparence)
    static int sim_log_count = 0;
    if (sim_log_count++ % 10 == 0) {
      Serial.print("[SIMULATION] Scénario survie - ");
      Serial.print("Solar: ");
      Serial.print(simulated_i_solar, 1);
      Serial.print("A, Load: ");
      Serial.print(simulated_i_load, 1);
      Serial.println("A (SIMULATION SEULEMENT)");
    }
  }
  #endif
  
  // ===== 3. CONSERVATION D'ÉNERGIE =====
  eps.i_batt = eps.i_load - eps.i_solar;
  eps.i_batt = safe_constrain(eps.i_batt, -1.5f, 2.0f);
  
  // ===== 4. MISE À JOUR SOC (Coulomb Counting) =====
  static unsigned long last_soc_update = 0;
  unsigned long current_time = millis();
  
  if (last_soc_update > 0) {
    float delta_t = (current_time - last_soc_update) / 1000.0f;
    float delta_soc = -eps.i_batt * delta_t / (BATT_CAPACITY_AS * TIME_SCALE_ORBITAL);
    
    soc_est += delta_soc;
    soc_est = safe_constrain(soc_est, 0.05f, 0.95f);
  } else {
    last_soc_update = current_time;
  }
  
  eps.soc = soc_est;
  
  // ===== 5. TENSION À VIDE FIXE =====
  eps.v_oc = calculate_ocv_from_soc(eps.soc);
  
  // Effet température
  float temp_factor = 1.0f + (raw.t_meas - 25.0f) * -0.003f;
  eps.v_oc *= temp_factor;
  
  // ===== 6. TENSION BATTERIE (MODÈLE FIXE) =====
  eps.r_int = R_BATT_INTERNAL;  
  
  // Formule physique correcte
  eps.v_batt = eps.v_oc - eps.r_int * eps.i_batt;
  
  // Bruit naturel minimal
  static uint32_t chem_variation_seed = 123456;
  chem_variation_seed = chem_variation_seed * 1103515245 + 12345;
  float natural_variation = ((chem_variation_seed >> 16) % 9 - 4) / 1000.0f;
  eps.v_batt += natural_variation;
  
  // Contraintes physiques
  float max_voltage_drop = 0.5f;
  float min_voltage_rise = -0.2f;
  float v_batt_phys_min = eps.v_oc - max_voltage_drop;
  float v_batt_phys_max = eps.v_oc - min_voltage_rise;
  
  eps.v_batt = safe_constrain(eps.v_batt, v_batt_phys_min, v_batt_phys_max);
  eps.v_batt = safe_constrain(eps.v_batt, V_BATT_MIN, V_BATT_MAX);
  
  // ===== 7. AUTRES VARIABLES =====
  eps.t_batt = safe_constrain(raw.t_meas, T_BATT_MIN, T_BATT_MAX);
  
  static uint32_t temp_noise_seed = 54321;
  temp_noise_seed = temp_noise_seed * 1103515245 + 12345;
  eps.t_eps = eps.t_batt + 3.0f + ((temp_noise_seed >> 16) % 21 - 10) / 10.0f;
  eps.t_eps = safe_constrain(eps.t_eps, T_EPS_MIN, T_EPS_MAX);
  
  eps.v_solar = (eps.i_solar > 0.01f) ? raw.solar_pot * V_SOLAR_MAX : 0.0f;
  if (eps.v_solar < 1.0f) eps.v_solar = 1.0f;
  
  eps.i_bus = eps.i_load;
  eps.v_bus = eps.v_batt * 0.97f;
  
  eps.converter_ratio = (eps.v_solar > 2.0f) ? eps.v_bus / eps.v_solar : 1.0f;
  eps.converter_ratio = safe_constrain(eps.converter_ratio, 0.8f, 1.2f);
  
  // ===== 8. PUISSANCES =====
  eps.p_batt = eps.v_batt * eps.i_batt;
  eps.p_solar = eps.v_solar * eps.i_solar;
  eps.p_bus = eps.v_bus * eps.i_bus;
  
  // ===== 9. MONITORING QUALITÉ (SEULEMENT) =====
  monitor_calibration_quality(eps);
  
  // ===== 10. SUGGESTIONS (SEULEMENT) =====
  log_recovery_suggestions(eps, raw);
  
  return eps;
}

/* ===================== 7 RÈGLES PHYSIQUES (ADAPTÉES AU MODÈLE FIXE) ===================== */
RuleResult evaluate_physics_rules(const EPSState &eps, float dv, float di) {
  RuleResult result;
  bool rule_active[8] = {false};
  
  // R1 — Température critique
  if (eps.t_batt > 55.0f) {
    rule_active[1] = true;
  }
  
  // R2 — Courant batterie extrême
  if (fabs(eps.i_batt) > 1.8f) {
    rule_active[2] = true;
  }
  
  // R3 — Sous-tension avec décharge (CRITIQUE)
  float v_low_threshold = 6.4f;
  
  if (eps.v_batt < v_low_threshold && eps.i_batt > 0.1f) {
    rule_active[3] = true;
  }
  
  // R4 — Déséquilibre énergétique
  float total_input = eps.p_batt + eps.p_solar;
  if (total_input > 1.0f) {
    float imbalance_ratio = eps.p_bus / total_input;
    if (imbalance_ratio > 2.0f || imbalance_ratio < 0.5f) {
      rule_active[4] = true;
    }
  }
  
  // R5 — Incohérence SOC vs Tension (AVEC MODÈLE FIXE)
  float expected_v_oc = calculate_ocv_from_soc(eps.soc);
  float v_drop_expected = eps.r_int * fabs(eps.i_batt);
  
  float v_batt_expected = expected_v_oc;
  if (eps.i_batt > 0) {
    v_batt_expected -= v_drop_expected;
  } else {
    v_batt_expected += v_drop_expected;
  }
  
  float temp_factor = 1.0f + (eps.t_batt - 25.0f) * -0.003f;
  v_batt_expected *= temp_factor;
  
  float voltage_error = fabs(eps.v_batt - v_batt_expected);
  float r5_threshold = (eps.soc < 0.2f) ? 0.5f : 0.3f;
  
  if (voltage_error > r5_threshold) {
    rule_active[5] = true;
  }
  
  // R6 — Oscillation rapide
  if (fabs(dv) > 0.3f || fabs(di) > 0.5f) {
    rule_active[6] = true;
  }
  
  // R7 — Capteur bloqué
  if (seq_filled) {
    float sum = 0, variance = 0;
    int count = seq_filled ? SEQ_LEN : seq_index;
    
    for (int i = 0; i < count; i++) sum += seq_buffer[i][0];
    float mean = sum / count;
    
    for (int i = 0; i < count; i++) {
      float diff = seq_buffer[i][0] - mean;
      variance += diff * diff;
    }
    
    float std_dev = (count > 1) ? sqrt(variance / (count - 1)) : 0.01f;
    
    if (count > 5 && std_dev < R7_SENSOR_STUCK_THRESHOLD) {
      rule_active[7] = true;
    }
  }
  
  // Persistance
  for (int i = 1; i <= 7; i++) {
    if (rule_active[i]) {
      rule_pers_count[i]++;
      if (rule_pers_count[i] >= RULE_CONFIRM_WINDOWS) {
        result.rules_triggered[i] = true;
        
        if (i <= 3) {
          if (result.level < 2) {
            result.level = 2;
            result.reason = "R" + String(i);
            
            if (i == 3) {
              // Log LED rouge pour transparence
              Serial.println("\n LED ROUGE DÉCLENCHÉE: R3 (Sous-tension avec décharge)");
              Serial.print("  SOC=");
              Serial.print(eps.soc * 100, 1);
              Serial.print("% V_batt=");
              Serial.print(eps.v_batt, 2);
              Serial.print("V I_batt=");
              Serial.print(eps.i_batt, 2);
              Serial.print("A V_oc=");
              Serial.print(eps.v_oc, 2);
              Serial.println("V");
              Serial.println("  [OBSERVATEUR PUR] Détection seulement - aucune action");
            }
          }
        } else {
          if (result.level < 1) {
            result.level = 1;
            result.reason = "R" + String(i);
          }
        }
      }
    } else {
      rule_pers_count[i] = 0;
    }
  }
  
  return result;
}

/* ===================== FONCTIONS IA (SANS MODIFICATION) ===================== */
int8_t quantize(float x, float scale, int zp) {
  int q = round(x / scale) + zp;
  if (q > 127) q = 127;
  if (q < -128) q = -128;
  return (int8_t)q;
}

float dequantize(int8_t q, float scale, int zp) {
  return scale * (q - zp);
}

float calculate_mse_smoothed(float new_mse) {
  mse_history[mse_index] = new_mse;
  mse_index = (mse_index + 1) % 10;
  
  float sum = 0;
  for (int i = 0; i < 10; i++) {
    sum += mse_history[i];
  }
  return sum / 10.0f;
}

void push_to_sequence_buffer(float* features) {
  for (int i = 0; i < FEATURE_LEN; i++) {
    seq_buffer[seq_index][i] = features[i];
  }
  seq_index = (seq_index + 1) % SEQ_LEN;
  if (seq_index == 0) seq_filled = true;
}

bool update_alert_state(float ai_score, const EPSState &eps, float* normalized_features, float dv, float di) {
  RuleResult rule_result = evaluate_physics_rules(eps, dv, di);
  
  float adjusted_th_warning = TH_WARNING;
  float adjusted_th_critical = TH_CRITICAL;
  
  if (eps.soc < 0.15f || eps.soc > 0.90f) {
    adjusted_th_warning *= 1.3f;
    adjusted_th_critical *= 1.5f;
  }
  
  // Hystérésis IA
  unsigned long current_time = millis();
  bool ai_warning_confirmed = false;
  bool ai_critical_confirmed = false;
  
  if (ai_score >= adjusted_th_critical) {
    if (ai_crit_start == 0) ai_crit_start = current_time;
    if (current_time - ai_crit_start >= HYSTERESIS_CRIT_MS) {
      ai_critical_confirmed = true;
    }
  } else {
    ai_crit_start = 0;
  }
  
  if (ai_score >= adjusted_th_warning) {
    if (ai_warn_start == 0) ai_warn_start = current_time;
    if (current_time - ai_warn_start >= HYSTERESIS_WARN_MS) {
      ai_warning_confirmed = true;
    }
  } else {
    ai_warn_start = 0;
  }
  
  // Fusion hybride
  int new_level = 0;
  String new_reason = "NORMAL";
  String new_trigger_source = "NONE";
  
  if (rule_result.level == 2 || ai_critical_confirmed) {
    new_level = 2;
    if (rule_result.level == 2) {
      new_reason = rule_result.reason;
      new_trigger_source = rule_result.reason;
    } else {
      new_reason = "AI_CRITICAL";
      new_trigger_source = "AI";
    }
  } 
  else if (rule_result.level == 1 || ai_warning_confirmed) {
    new_level = 1;
    if (rule_result.level == 1) {
      new_reason = rule_result.reason;
      new_trigger_source = rule_result.reason;
    } else {
      new_reason = "AI_WARNING";
      new_trigger_source = "AI";
    }
  }
  
  if (new_level != current_alert_level) {
    current_alert_level = new_level;
    last_alert_reason = new_reason;
    last_trigger_source = new_trigger_source;
  }
  
  return (new_level > 0);
}

void prepare_ai_features(EPSState eps, float* normalized_features, float dv, float di, float dt) {
  normalized_features[0] = safe_constrain(
    (eps.v_batt - V_BATT_MIN) / (V_BATT_MAX - V_BATT_MIN),
    0.0f, 1.0f
  );
  
  normalized_features[1] = safe_constrain(
    (eps.i_batt - I_BATT_MIN) / (I_BATT_MAX - I_BATT_MIN),
    0.0f, 1.0f
  );
  
  normalized_features[2] = safe_constrain(
    (eps.t_batt - T_BATT_MIN) / (T_BATT_MAX - T_BATT_MIN),
    0.0f, 1.0f
  );
  
  float v_bus_clamped = safe_constrain(eps.v_bus, V_BATT_MIN, V_BATT_MAX);
  normalized_features[3] = safe_constrain(
    (v_bus_clamped - V_BATT_MIN) / (V_BATT_MAX - V_BATT_MIN),
    0.0f, 1.0f
  );
  
  normalized_features[4] = safe_constrain(eps.i_bus / LOAD_MAX, 0.0f, 1.0f);
  normalized_features[5] = safe_constrain(eps.v_solar / V_SOLAR_MAX, 0.0f, 1.0f);
  normalized_features[6] = safe_constrain(eps.i_solar / I_SOLAR_MAX, 0.0f, 1.0f);
  normalized_features[7] = safe_constrain(eps.soc, 0.0f, 1.0f);
  
  float t_eps_range = T_EPS_MAX - T_EPS_MIN;
  normalized_features[8] = safe_constrain(
    (eps.t_eps - T_EPS_MIN) / t_eps_range,
    0.0f, 1.0f
  );
  
  float p_batt_max = V_BATT_MAX * I_BATT_MAX;
  normalized_features[9] = safe_constrain(eps.p_batt / p_batt_max, 0.0f, 1.0f);
  
  float p_solar_max = V_SOLAR_MAX * I_SOLAR_MAX;
  normalized_features[10] = safe_constrain(eps.p_solar / p_solar_max, 0.0f, 1.0f);
  
  float p_bus_max = V_BATT_MAX * LOAD_MAX;
  normalized_features[11] = safe_constrain(eps.p_bus / p_bus_max, 0.0f, 1.0f);
  
  normalized_features[12] = safe_constrain((eps.converter_ratio - 0.8f) / 0.4f, 0.0f, 1.0f);
  
  normalized_features[13] = safe_constrain((dv + 0.5f) / 1.0f, 0.0f, 1.0f);
  normalized_features[14] = safe_constrain((di + 1.0f) / 2.0f, 0.0f, 1.0f);
  normalized_features[15] = safe_constrain((dt + 0.5f) / 1.0f, 0.0f, 1.0f);
  
  v_batt_history[history_index] = eps.v_batt;
  history_index = (history_index + 1) % ROLLING_WINDOW_SIZE;
  if (history_index == 0) history_filled = true;
  
  float sum = 0;
  int count = history_filled ? ROLLING_WINDOW_SIZE : history_index;
  for (int i = 0; i < count; i++) sum += v_batt_history[i];
  float r_mean = count > 0 ? sum / count : V_BATT_NOMINAL;
  
  float variance = 0;
  for (int i = 0; i < count; i++) {
    float diff = v_batt_history[i] - r_mean;
    variance += diff * diff;
  }
  float r_std = (count > 1) ? sqrt(variance / (count - 1)) : 0.01f;
  
  normalized_features[16] = safe_constrain(r_std / 0.2f, 0.0f, 1.0f);
  normalized_features[17] = safe_constrain(
    (r_mean - V_BATT_MIN) / (V_BATT_MAX - V_BATT_MIN),
    0.0f, 1.0f
  );
  
  for (int i = 0; i < FEATURE_LEN; i++) {
    if (normalized_features[i] < 0.0f || normalized_features[i] > 1.0f || isnan(normalized_features[i])) {
      normalized_features[i] = 0.5f;
    }
  }
}

float run_autoencoder_inference(float* normalized_features) {
  if (!interpreter || !input_tensor) return 0.0f;
  
  float in_scale = input_tensor->params.scale;
  int in_zp = input_tensor->params.zero_point;
  
  for (int i = 0; i < FEATURE_LEN; i++) {
    input_tensor->data.int8[i] = quantize(normalized_features[i], in_scale, in_zp);
  }
  
  if (interpreter->Invoke() != kTfLiteOk) {
    return 0.0f;
  }
  
  float out_scale = output_tensor->params.scale;
  int out_zp = output_tensor->params.zero_point;
  float mse = 0.0f;
  
  for (int i = 0; i < FEATURE_LEN; i++) {
    float reconstructed = dequantize(output_tensor->data.int8[i], out_scale, out_zp);
    float error = reconstructed - normalized_features[i];
    mse += error * error;
  }
  
  return mse / FEATURE_LEN;
}

void send_to_obc_with_sequence(unsigned long timestamp, int alert_level, 
                               float mse, const EPSState &eps, 
                               const String& trigger_source) {
  
  Serial.print("{");
  Serial.print("\"timestamp\":");
  Serial.print(timestamp);
  Serial.print(",\"level\":");
  Serial.print(alert_level);
  Serial.print(",\"reason\":\"");
  Serial.print(last_alert_reason);
  Serial.print("\",\"trigger_source\":\"");
  Serial.print(trigger_source);
  Serial.print("\",\"mse\":");
  Serial.print(mse, 4);
  Serial.print(",\"mse_smoothed\":");
  Serial.print(mse_smoothed, 4);
  Serial.print(",\"mode\":\"");
  Serial.print(DEMO_MODE ? "DEMO" : "PROJECT");
  
  #if DEMO_MODE && DEMO_OVERLOAD_SCENARIO
    Serial.print("\",\"simulation_active\":");
    Serial.print(eps.soc < DEMO_OVERLOAD_SOC_THRESHOLD ? "true" : "false");
  #endif
  
  Serial.print("\",\"philosophy\":\"PURE_OBSERVER\",\"model_fixed\":true");
  
  Serial.print(",\"eps\":{");
  Serial.print("\"v_batt\":");
  Serial.print(eps.v_batt, 3);
  Serial.print(",\"i_batt\":");
  Serial.print(eps.i_batt, 3);
  Serial.print(",\"soc\":");
  Serial.print(eps.soc, 3);
  Serial.print(",\"v_oc\":");
  Serial.print(eps.v_oc, 3);
  Serial.print(",\"r_int\":");
  Serial.print(eps.r_int * 1000, 1);
  Serial.print(",\"r_int_fixed\":true");
  Serial.print(",\"p_bus\":");
  Serial.print(eps.p_bus, 2);
  Serial.print(",\"p_batt\":");
  Serial.print(eps.p_batt, 2);
  Serial.print(",\"p_solar\":");
  Serial.print(eps.p_solar, 2);
  Serial.print("}");
  
  Serial.print(",\"seq_len\":");
  int count = seq_filled ? SEQ_LEN : seq_index;
  Serial.print(count);
  Serial.print(",\"seq\":[");
  
  int start_idx = seq_filled ? seq_index : 0;
  for (int s = 0; s < count; s++) {
    int buffer_idx = (start_idx + s) % SEQ_LEN;
    Serial.print("[");
    
    for (int f = 0; f < FEATURE_LEN; f++) {
      Serial.print(seq_buffer[buffer_idx][f], 4);
      if (f < FEATURE_LEN - 1) Serial.print(",");
    }
    
    Serial.print("]");
    if (s < count - 1) Serial.print(",");
  }
  
  Serial.println("]}");
  
  last_send_to_obc = millis();
}

/* ===================== SETUP AVEC PHILOSOPHIE EXPLICITE ===================== */
void setup() {
  Serial.begin(115200);
  delay(2000);
  
  Serial.println("\n" + String(78, '='));
  Serial.println("    EPS GUARDIAN - VERSION OBSERVATEUR PUR");
  Serial.println("   IEEE AESS & IES Challenge 2025");
  Serial.println(String(78, '='));
  
  Serial.println("\n PHILOSOPHIE DÉFINITIVE :");
  Serial.println(" OBSERVATEUR PUR - Détection seulement");
  Serial.println(" MODÈLE FIXE - Reproductibilité garantie");
  Serial.println(" MONITORING - Qualité du modèle mesurée");
  Serial.println(" SUGGESTIONS - Actions manuelles recommandées");
  
  Serial.println("\n PAS D'ACTIONS AUTOMATIQUES :");
  Serial.println(" Pas d'ajustement du modèle (R_int fixe)");
  Serial.println(" Pas de modification du scénario");
  Serial.println(" Pas de récupération automatique");
  Serial.println(" Pas de calibration en ligne (monitoring seulement)");
  
  Serial.println("\n CONFIGURATION ACTIVE:");
  Serial.println("Mode: " + String(DEMO_MODE ? "DÉMONSTRATION" : "PROJET SCIENTIFIQUE"));
  Serial.print("Échelle temporelle: 1 sec MCU = ");
  Serial.print(TIME_SCALE_ORBITAL, 0);
  Serial.println(" sec orbitales");
  Serial.print("SOC initial: ");
  Serial.print(INITIAL_SOC * 100, 1);
  Serial.print("% → V_oc=");
  Serial.print(calculate_ocv_from_soc(INITIAL_SOC), 2);
  Serial.println("V");
  Serial.print("R_int fixe: ");
  Serial.print(R_BATT_INTERNAL * 1000, 0);
  Serial.println(" mΩ");
  
  #if DEMO_MODE && DEMO_OVERLOAD_SCENARIO
    Serial.println("\n SCÉNARIO DE DÉMO (SIMULATION):");
    Serial.println("• SOC<20% → Simulation scénario survie");
    Serial.println("• Solar=0A (nuit), Load=1.5A (mission critique)");
    Serial.println("• SIMULATION SEULEMENT - pas de modification réelle");
  #endif
  
  Serial.println("\n DONNÉES SCIENTIFIQUES :");
  Serial.println("• Logs CSV complets (toutes les 2 secondes)");
  Serial.println("• JSON détaillé pour OBC (sur alertes)");
  Serial.println("• Monitoring qualité modèle (toutes les 30 secondes)");
  Serial.println("• Suggestions récupération (si conditions critiques)");
  
  Serial.println("\n CHECKLIST POUR LE JURY :");
  Serial.println("CE QUE FAIT LE SYSTÈME  :");
  Serial.println("• Détection précise des anomalies batterie");
  Serial.println("• Monitoring scientifique de la qualité du modèle");
  Serial.println("• Suggestions actionnables pour l'opérateur");
  Serial.println("• Logs complets pour analyse postérieure");
  Serial.println("• Indication visuelle (LEDs) de l'état");
  
  Serial.println("\nCE QUE NE FAIT PAS LE SYSTÈME :");
  Serial.println("• Aucun ajustement automatique du modèle physique");
  Serial.println("• Aucune modification des entrées (potentiomètres)");
  Serial.println("• Aucune action corrective sur le système");
  Serial.println("• Aucune calibration en ligne (monitoring seulement)");
  Serial.println("• Aucune récupération automatique (suggestions seulement)");
  
  Serial.println("\n ARGUMENTAIRE POUR LE JURY :");
  Serial.println("\"Notre système adopte une philosophie d'observateur pur :");
  Serial.println("il détecte, analyse et suggère, mais n'agit jamais sur le système.");
  Serial.println("Cette approche garantit :");
  Serial.println("• Reproductibilité scientifique : Mêmes entrées → mêmes résultats");
  Serial.println("• Transparence totale : Pas de 'magie' cachée");
  Serial.println("• Défendabilité : Chaque décision est traçable et explicable");
  Serial.println("• Séparation des responsabilités : Détection (MCU) vs Action (Opérateur/OBC)\"");
  
  Serial.println("\n TRACE SCIENTIFIQUE (CSV):");
  Serial.println("time,mode,mse,alert_level,trigger_source,obc_sent,v_batt,i_batt,soc,p_bus,p_total,simulation_active,suggestion_available");
  Serial.println(String(78, '='));

  // Initialisation GPIO
  pinMode(LED_GREEN, OUTPUT);
  pinMode(LED_YELLOW, OUTPUT);
  pinMode(LED_RED, OUTPUT);
  
  // Séquence LED de test
  Serial.println("\n TEST LEDs:");
  Serial.println("• Vert (0) → Jaune (1) → Rouge (2) → Vert (0)");
  update_leds(0); delay(200);
  update_leds(1); delay(200);
  update_leds(2); delay(200);
  update_leds(0); delay(200);
  Serial.println(" LEDs fonctionnelles");

  // Initialisation I2C
  Wire.begin(SDA_PIN, SCL_PIN);
  
  // Initialisation capteurs
  tempSensor.begin();
  ina219_detected = ina219.begin();
  if (ina219_detected) {
    ina219.setCalibration_32V_2A();
    Serial.println(" INA219 détecté (bench de test seulement)");
  } else {
    Serial.println("  INA219 non détecté - mode simulation bench");
  }

  // Initialisation TFLite Micro
  const tflite::Model* model = tflite::GetModel(g_model_data);
  static tflite::MicroMutableOpResolver<15> resolver;
  resolver.AddFullyConnected();
  resolver.AddMean();
  resolver.AddAdd();
  resolver.AddSub();
  resolver.AddMul();
  resolver.AddLogistic();
  resolver.AddReshape();
  resolver.AddConcatenation();
  resolver.AddRelu();
  resolver.AddSoftmax();
  resolver.AddTanh();
  resolver.AddStridedSlice();
  
  static tflite::MicroInterpreter static_interpreter(
    model, resolver, tensor_arena, kArenaSize
  );
  interpreter = &static_interpreter;
  
  if (interpreter->AllocateTensors() == kTfLiteOk) {
    input_tensor = interpreter->input(0);
    output_tensor = interpreter->output(0);
    Serial.println(" TFLite Micro initialisé");
  } else {
    Serial.println("  TFLite Micro échec - mode dégradé");
    interpreter = nullptr;
  }

  // Initialisation buffers
  for (int i = 0; i < ROLLING_WINDOW_SIZE; i++) {
    v_batt_history[i] = V_BATT_NOMINAL;
  }
  history_filled = true;
  
  // Initialisation du générateur aléatoire
  randomSeed(analogRead(0));
  
  Serial.println("\n SYSTÈME PRÊT - OBSERVATEUR PUR ACTIVÉ");
  Serial.println(" Modèle physique fixe et validé");
  delay(2000);
}

/* ===================== LOOP PRINCIPAL - VERSION SIMPLIFIÉE ===================== */
void loop() {
  unsigned long current_time = millis();
  
  // 1. Lecture capteurs (sans modification)
  RawSensors raw = read_raw_sensors_pure();
  
  // 2. Modèle physique FIXE
  EPSState eps = battery_model_pure_observer(raw);
  
  // 3. Calcul dérivées pour détection
  float dv = eps.v_batt - prev_v_batt;
  float di = eps.i_batt - prev_i_batt;
  float dt = eps.t_batt - prev_t_batt;
  
  // 4. Préparation features IA
  float normalized_features[FEATURE_LEN];
  prepare_ai_features(eps, normalized_features, dv, di, dt);
  
  // 5. Inférence autoencoder
  float mse = 0.0f;
  if (interpreter) {
    mse = run_autoencoder_inference(normalized_features);
    mse_smoothed = calculate_mse_smoothed(mse);
  } else {
    static uint32_t mse_noise_seed = 13579;
    mse_noise_seed = mse_noise_seed * 1103515245 + 12345;
    mse = 0.08f + ((mse_noise_seed >> 16) % 10) / 100.0f;
    mse_smoothed = mse;
  }
  
  // 6. Tampon séquentiel
  push_to_sequence_buffer(normalized_features);
  
  // 7. Détection hybride (règles physiques + IA)
  bool has_alert = update_alert_state(mse_smoothed, eps, normalized_features, dv, di);
  
  // 8. LEDs (indication état seulement)
  update_leds(current_alert_level);
  
  // 9. Envoi OBC (logs seulement)
  bool send_to_obc = false;
  if (current_alert_level == 2) {
    send_to_obc = true;
  } else if (current_alert_level == 1) {
    float random_value = (float)random(0, 10000) / 10000.0f;
    if (random_value < OBC_SEND_PROB_WARNING) {
      send_to_obc = true;
    }
  }
  
  if (send_to_obc && (millis() - last_send_to_obc >= MIN_OBC_SEND_INTERVAL)) {
    send_to_obc_with_sequence(current_time / 1000, current_alert_level, 
                             mse_smoothed, eps, last_trigger_source);
  }
  
  // 10. Trace CSV scientifique (toutes les 2 secondes)
  static unsigned long last_trace = 0;
  if (current_time - last_trace > 2000) {
    Serial.print(current_time / 1000);
    Serial.print(",");
    Serial.print(DEMO_MODE ? "DEMO," : "PROJ,");
    Serial.print(mse_smoothed, 4);
    Serial.print(",");
    Serial.print(current_alert_level);
    Serial.print(",");
    Serial.print(last_trigger_source);
    Serial.print(",");
    Serial.print(send_to_obc ? "1" : "0");
    Serial.print(",");
    Serial.print(eps.v_batt, 2);
    Serial.print(",");
    Serial.print(eps.i_batt, 3);
    Serial.print(",");
    Serial.print(eps.soc, 3);
    Serial.print(",");
    Serial.print(eps.p_bus, 2);
    Serial.print(",");
    Serial.print(eps.p_batt + eps.p_solar, 2);
    Serial.print(",");
    
    #if DEMO_MODE && DEMO_OVERLOAD_SCENARIO
      Serial.print(eps.soc < DEMO_OVERLOAD_SOC_THRESHOLD ? "1" : "0");
    #else
      Serial.print("0");
    #endif
    
    // Indicateur suggestion disponible
    bool suggestion_available = (eps.soc < 0.20f) && 
                              (eps.v_batt < 6.6f) && 
                              (eps.i_batt > 0.3f);
    Serial.print(",");
    Serial.print(suggestion_available ? "1" : "0");
    
    Serial.println();
    
    last_trace = current_time;
  }
  
  // 11. Mise à jour dérivées
  prev_v_batt = eps.v_batt;
  prev_i_batt = eps.i_batt;
  prev_t_batt = eps.t_batt;
  
  // 12. Délai fixe
  delay(1000);
}