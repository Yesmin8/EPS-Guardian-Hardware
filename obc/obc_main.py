#!/usr/bin/env python3
"""
OBC Principal - Version Professionnelle Complète
Architecture modulaire pour la surveillance EPS satellite
Avec RL Advisor et attente patiente du MCU
Support CSV et JSON
"""

import sys
import time
import serial
import json
import hashlib
import numpy as np
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple
import threading
import os

# Import des modules OBC
from obc_config import *
from obc_ai_inference import OBC_AI
from obc_fault_diagnosis import diagnoser
from obc_decision_engine import decision_engine
from obc_action_manager import action_manager, SystemAction
from obc_state_machine import state_machine
from obc_reinforcement_learning import (
    OBCReinforcementLearner, 
    RLState, 
    RLAction, 
    RLExperience,
    SafeRLValidator
)

class ProfessionalOBCSystem:
    """
    Système OBC professionnel avec architecture modulaire
    Respecte l'autonomie du MCU - Attente patiente
    Support CSV et JSON
    """
    
    def __init__(self):
        # Modules système
        self.serial_conn = None
        self.ai_engine = None
        self.rl_learner = None
        
        # Buffers et historiques
        self.message_buffer = []
        self.system_log = []
        self.performance_stats = {
            "messages_received": 0,
            "alerts_processed": 0,
            "avg_processing_time": 0,
            "start_time": time.time(),
            "rl_suggestions_used": 0,
            "rl_suggestions_rejected": 0,
            "waiting_time": 0,
        }
        
        # État OBC
        self.obc_state = "BOOT"
        self.episode_buffer = []
        self.episode_start_time = 0
        self.current_episode_type = None
        self.last_buffer_hash = None
        self.episode_analysis = None
        
        # RL tracking
        self.last_rl_state = None
        self.last_rl_action = None
        self.last_rl_suggestions = []
        self.rl_influence_active = False
        
        # MCU tracking
        self.mcu_ready = False
        self.first_message_received = False
        self.mcu_first_contact = 0
        self.waiting_start_time = 0
        
        # Statistiques d'attente
        self.waiting_stats = {
            "bytes_received": 0,
            "lines_read": 0,
            "invalid_messages": 0,
            "waiting_time": 0
        }
        
        # Recovery monitoring
        self.recovery_monitor_count = 0
        self.recovery_stable_count = 0
        self.recovery_start_time = 0
        self.RECOVERY_STABLE_THRESHOLD = 5
        self.RECOVERY_MONITOR_TIMEOUT = 300
        
        # Decision lock
        self.decision_locked = False
        self.lock_timestamp = 0
        
        # Configuration
        self.config = OBC_CONFIG.copy()
        
        # État interne
        self.running = False
        self.system_ready = False
        self.simulation_mode = False
        self.last_heartbeat = time.time()
        self.last_health_check = time.time()
        self.last_status = time.time()
        
        # Phase de mission
        self.mission_phase = "ORBITAL"
        
        # Debug
        self.debug_raw_data = []
        self.max_debug_lines = 5
    
    # ===================== INITIALISATION =====================
    
    def initialize(self):
        """Initialisation patiente du système"""
        print("\n" + "=" * 70)
        print(" OBC PROFESSIONNEL - DÉMARRAGE")
        print("=" * 70)
        
        print("\n RÈGLES ARCHITECTURALES:")
        print("   1. L'OBC attend le MCU, pas l'inverse")
        print("   2. Pas de timeout fatal")
        print("   3. Lecture passive pendant l'attente")
        print("   4. Activation progressive après premier contact")
        print("   5. Le MCU est autonome (R1-R3 appliquées au boot)")
        
        print("\n FORMATS SUPPORTÉS:")
        print("   • CSV: index,mode,mse_mcu,level,reason,...")
        print("   • JSON: {\"state\":\"NORMAL\",\"features\":[...],\"mse\":0.1}")
        
        try:
            # 1. Connexion série (tentative)
            self._connect_serial()
            
            # 2. Passer en attente MCU
            self._set_obc_state("WAITING_MCU")
            self.waiting_start_time = time.time()
            
            # 3. Initialisation passive des modules
            self._initialize_modules_passive()
            
            # 4. NE PAS attendre activement le MCU
            print("\n📡 OBC EN ATTENTE DU MCU...")
            print("   - MCU: Autonome, démarre ses propres règles")
            print("   - OBC: Observateur passif, prêt à recevoir")
            print("   - Attente: Illimitée (pas de timeout)")
            print("   - Premier message: Déclenchera l'activation")
            
            if not self.serial_conn or not self.serial_conn.is_open:
                print("     Port série non connecté - Mode simulation activé")
                self.simulation_mode = True
            
            self.system_ready = True
            return True
            
        except Exception as e:
            print(f"\n  Erreur initialisation OBC: {e}")
            print("   Continuation en mode résilient...")
            self.simulation_mode = True
            self._set_obc_state("WAITING_MCU")
            self.system_ready = True
            return True  # On ne s'arrête jamais
    
    def _connect_serial(self):
        """Tente une connexion série"""
        if not self.config.get("serial_port"):
            print("     Pas de port série configuré")
            return False
        
        port = self.config['serial_port']
        print(f"\n🔌 Connexion série: {port}...")
        
        # Attendre que Windows libère le port
        time.sleep(3)
        
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                print(f"    Tentative {attempt + 1}/{max_attempts}...")
                
                self.serial_conn = serial.Serial(
                    port=port,
                    baudrate=self.config['baudrate'],
                    timeout=self.config['timeout'],
                    write_timeout=2,
                    bytesize=serial.EIGHTBITS,
                    parity=serial.PARITY_NONE,
                    stopbits=serial.STOPBITS_ONE
                )
                
                time.sleep(2)
                
                if self.serial_conn.is_open:
                    self.serial_conn.reset_input_buffer()
                    self.serial_conn.reset_output_buffer()
                    print(f"    Connecté à {port}")
                    return True
                else:
                    print(f"    Port non ouvert")
                    
            except serial.SerialException as e:
                print(f"    Tentative {attempt + 1} échouée: {e}")
                if "PermissionError" in str(e) or "Accès refusé" in str(e):
                    print("    Port occupé - attente 3s avant nouvelle tentative...")
                    time.sleep(3)
                else:
                    break
            except Exception as e:
                print(f"    Erreur inattendue: {e}")
                break
        
        print(f"    Échec connexion après {max_attempts} tentatives")
        return False
    
    def _initialize_modules_passive(self):
        """Initialisation passive des modules"""
        print("\n🔧 Initialisation modules (mode passif)...")
        
        try:
            # IA (chargement seulement)
            self.ai_engine = OBC_AI(MODEL_PATH, SCALER_PATH)
            self.ai_engine.ready = False
            
            # RL Advisor
            if self.config.get('enable_rl', True):
                self.rl_learner = OBCReinforcementLearner()
                self.rl_learner.config["learning_enabled"] = False
                self.rl_learner.set_mission_phase(self.mission_phase)
                print("    RL: Initialisé (mode passif)")
            
            # Modules déterministes
            print("    Modules déterministes: Prêts")
            
        except Exception as e:
            print(f"     Erreur initialisation partielle: {e}")
    
    # ===================== PARSING DES MESSAGES =====================
    
    def _parse_message_passive(self, raw_line: str) -> Optional[Dict]:
        """Parse un message en mode passif - Support CSV ET JSON"""
        if not raw_line or raw_line.strip() == "":
            self.waiting_stats["invalid_messages"] += 1
            return None
        
        if len(self.debug_raw_data) < self.max_debug_lines:
            print(f"DEBUG RAW [{len(self.debug_raw_data)+1}]: '{raw_line}'")
            self.debug_raw_data.append(raw_line)
        
        if ',' in raw_line and len(raw_line) > 5:
            msg = self._parse_csv_message(raw_line)
            if msg:
                return msg
        
        if raw_line.startswith('{'):
            msg = self._parse_json_message(raw_line)
            if msg:
                return msg
        
        self.waiting_stats["invalid_messages"] += 1
        return None
    
    def _parse_csv_message(self, raw_line: str) -> Optional[Dict]:
        """Parse le format CSV de votre MCU réel"""
        try:
            parts = raw_line.strip().split(',')
            
            if len(parts) < 13:
                return None
            
            index = int(parts[0])
            mode = parts[1]
            mse_mcu = float(parts[2])
            level = int(parts[3])
            reason = parts[4]
            source = int(parts[5])
            v_batt = float(parts[6])
            i_batt = float(parts[7])
            soc = float(parts[8])
            v_bus = float(parts[9])
            i_load = float(parts[10])
            solar = int(parts[11])
            load = int(parts[12])
            
            if level == 2:
                state = "CRITICAL"
            elif level == 1:
                state = "WARNING"
            else:
                state = "NORMAL"
            
            t_batt_base = 25.0
            t_current_effect = max(0, i_batt - 1.0) * 5.0
            t_soc_effect = (soc - 0.5) * 4.0
            t_batt = t_batt_base + t_current_effect + t_soc_effect
            
            solar_active = solar > 0
            v_solar = 16.0 if solar_active else 0.0
            i_solar = 1.2 if solar_active else 0.0
            
            msg = {
                "state": state,
                "mse": mse_mcu,
                "features": [
                    v_batt,
                    i_batt,
                    t_batt,
                    v_solar,
                    i_solar,
                    v_bus,
                    i_load
                ],
                "msg_type": "HEARTBEAT" if state == "NORMAL" else "ALERT",
                "timestamp": time.time(),
                "raw_data": {
                    "index": index,
                    "mode": mode,
                    "reason": reason,
                    "source": source,
                    "soc": soc,
                    "solar": solar,
                    "load": load,
                    "level": level,
                    "mse_mcu": mse_mcu
                }
            }
            
            if len(self.debug_raw_data) < self.max_debug_lines:
                print(f"DEBUG CSV PARSED: state={state}, mse={mse_mcu:.3f}")
            
            return msg
            
        except Exception as e:
            if len(self.debug_raw_data) < self.max_debug_lines:
                print(f"DEBUG CSV ERROR: {e}")
            return None
    
    def _parse_json_message(self, raw_line: str) -> Optional[Dict]:
        """Parse le format JSON"""
        try:
            json_end = raw_line.rfind('}') + 1
            if json_end <= 0:
                return None
            
            json_str = raw_line[:json_end]
            msg = json.loads(json_str)
            
            if not all(key in msg for key in ["state", "features", "mse"]):
                return None
            
            if len(msg["features"]) != FEATURE_COUNT:
                return None
            
            msg["timestamp"] = time.time()
            msg["msg_type"] = "ALERT" if msg["state"] in ["WARNING", "CRITICAL"] else "HEARTBEAT"
            
            if len(self.debug_raw_data) < self.max_debug_lines:
                print(f"DEBUG JSON PARSED: state={msg['state']}, mse={msg['mse']:.3f}")
            
            return msg
            
        except json.JSONDecodeError:
            return None
        except Exception:
            return None
    
    # ===================== BOUCLE PRINCIPALE =====================
    
    def run(self):
        """Boucle principale avec attente patiente"""
        if not self.system_ready:
            print(" Système non initialisé")
            return
        
        print("\n" + "=" * 70)
        print(" DÉMARRAGE BOUCLE PRINCIPALE")
        print("=" * 70)
        
        print(f"\n ÉTAT INITIAL:")
        print(f"   OBC: {self.obc_state}")
        print(f"   MCU: {'PRÊT' if self.mcu_ready else 'EN ATTENTE'}")
        print(f"   Mode: {'SIMULATION' if self.simulation_mode else 'RÉEL'}")
        print(f"   RL: {'ACTIF' if self.config.get('enable_rl') else 'INACTIF'}")
        
        print("\n" + "-" * 70)
        print("Mode attente MCU actif... (Ctrl+C pour arrêter)")
        print("-" * 70 + "\n")
        
        self.running = True
        last_waiting_status = time.time()
        
        try:
            while self.running:
                current_time = time.time()
                
                if self.obc_state == "WAITING_MCU":
                    self._handle_waiting_mcu_state(current_time)
                    
                    if current_time - last_waiting_status > 5:
                        self._print_waiting_status(current_time)
                        last_waiting_status = current_time
                
                elif self.obc_state == "IDLE":
                    self._read_serial_data()
                    self._process_message_buffer()
                    
                    if current_time - self.last_health_check > 10:
                        self._check_system_health()
                        self.last_health_check = current_time
                    
                    if current_time - self.last_status > 30:
                        self._print_system_status()
                        self.last_status = current_time
                
                elif self.obc_state in ["ACCUMULATING", "ANALYZING", "DECIDING", 
                                       "RECOVERY_MONITORING", "ESCALATION"]:
                    self._read_serial_data()
                    self._process_message_buffer()
                
                self._check_state_timeouts(current_time)
                
                time.sleep(0.01)
                
        except KeyboardInterrupt:
            print("\n\n Arrêt demandé par l'utilisateur")
        except Exception as e:
            print(f"\n Erreur inattendue: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.shutdown()
    
    # ===================== GESTION DES ÉTATS =====================
    
    def _set_obc_state(self, new_state: str):
        """Transition d'état"""
        old_state = self.obc_state
        self.obc_state = new_state
        
        print(f"\n TRANSITION OBC: {old_state} → {new_state}")
        
        if new_state == "WAITING_MCU":
            self._enter_waiting_mcu_state()
        elif new_state == "IDLE":
            self._enter_idle_state()
        elif new_state == "ACCUMULATING":
            self._enter_accumulating_state()
        elif new_state == "ANALYZING":
            self._enter_analyzing_state()
        elif new_state == "DECIDING":
            self._enter_deciding_state()
        elif new_state == "RECOVERY_MONITORING":
            self._enter_recovery_monitoring_state()
        elif new_state == "ESCALATION":
            self._enter_escalation_state()
    
    def _enter_waiting_mcu_state(self):
        """État WAITING_MCU"""
        print("    En attente du MCU...")
        print("    Règles appliquées:")
        print("      • Pas de timeout fatal")
        print("      • Lecture série passive")
        print("      • Aucune décision prise")
        print("      • IA inactive")
        print("      • RL désactivé")
        
        if self.ai_engine:
            self.ai_engine.ready = False
        if self.rl_learner:
            self.rl_learner.config["learning_enabled"] = False
        
        self.message_buffer = []
        self.episode_buffer = []
        self.mcu_ready = False
        self.first_message_received = False
    
    def _enter_idle_state(self):
        """État IDLE"""
        print("    Système prêt - Attente nouvelle alerte")
        self.episode_buffer = []
        self.current_episode_type = None
        self.last_buffer_hash = None
        self.episode_analysis = None
        self.recovery_monitor_count = 0
        self.recovery_stable_count = 0
        self.decision_locked = False
        
        if self.ai_engine:
            self.ai_engine.ready = True
        if self.rl_learner and self.config.get('enable_rl'):
            self.rl_learner.config["learning_enabled"] = True
    
    def _enter_accumulating_state(self):
        """État ACCUMULATING"""
        print("    Début accumulation épisode")
        self.episode_start_time = time.time()
    
    def _enter_analyzing_state(self):
        """État ANALYZING"""
        print("    Analyse de l'épisode en cours")
    
    def _enter_deciding_state(self):
        """État DECIDING"""
        print("    Prise de décision en cours")
    
    def _enter_recovery_monitoring_state(self):
        """État RECOVERY_MONITORING"""
        print("\n DÉBUT RECOVERY MONITORING")
        print(f"   Timeout: {self.RECOVERY_MONITOR_TIMEOUT}s")
        print(f"   Stable threshold: {self.RECOVERY_STABLE_THRESHOLD} messages")
        
        self.recovery_monitor_count = 0
        self.recovery_stable_count = 0
        self.recovery_start_time = time.time()
    
    def _enter_escalation_state(self):
        """État ESCALATION"""
        print("\n ENTREE EN MODE ESCALATION")
        print("   Actions d'urgence nécessaires")
        self._send_escalation_command()
    
    def _check_state_timeouts(self, current_time: float):
        """Vérifie les timeouts d'état"""
        new_state = state_machine.check_timeout()
        if new_state:
            self._set_obc_state(new_state.value)
    
    # ===================== GESTION MCU =====================
    
    def _handle_waiting_mcu_state(self, current_time: float):
        """Gère l'état d'attente MCU"""
        if self.serial_conn and self.serial_conn.in_waiting > 0:
            try:
                raw = self.serial_conn.readline()
                line = raw.decode('utf-8', errors='ignore').strip()
                
                if line:
                    self.waiting_stats["bytes_received"] += len(raw)
                    self.waiting_stats["lines_read"] += 1
                    
                    msg = self._parse_message_passive(line)
                    
                    if msg:
                        self._on_first_mcu_message(msg, current_time)
                    else:
                        if self.waiting_stats["lines_read"] <= 10:
                            print(f"DEBUG WAIT: Ligne invalide '{line[:60]}...'")
                            
            except Exception as e:
                if self.waiting_stats["lines_read"] <= 5:
                    print(f"DEBUG WAIT Exception: {e}")
        
        elif self.simulation_mode and not self.first_message_received:
            if current_time - self.waiting_start_time > 2:
                msg = {
                    "state": "NORMAL",
                    "features": [7.4, 0.5, 25.0, 16.0, 1.2, 5.0, 1.0],
                    "mse": 0.1,
                    "msg_type": "HEARTBEAT"
                }
                self._on_first_mcu_message(msg, current_time)
        
        self.waiting_stats["waiting_time"] = current_time - self.waiting_start_time
    
    def _on_first_mcu_message(self, msg: Dict, timestamp: float):
        """Premier contact MCU établi"""
        self.first_message_received = True
        self.mcu_ready = True
        self.mcu_first_contact = timestamp
        
        print("\n" + "=" * 70)
        print(" PREMIER CONTACT MCU ÉTABLI")
        print("=" * 70)
        
        print(f"\n Message reçu après {self.waiting_stats['waiting_time']:.1f}s")
        print(f"   État MCU: {msg['state']}")
        print(f"   MSE MCU: {msg.get('mse', 'N/A')}")
        print(f"   Features: {len(msg['features'])} valeurs")
        
        print(f"\n Statistiques d'attente:")
        print(f"   Lignes lues: {self.waiting_stats['lines_read']}")
        print(f"   Bytes reçus: {self.waiting_stats['bytes_received']}")
        print(f"   Messages invalides: {self.waiting_stats['invalid_messages']}")
        
        print(f"\n Activation des modules OBC...")
        
        if self.ai_engine:
            self.ai_engine.ready = True
            print("    IA: Active")
        
        if self.rl_learner and self.config.get('enable_rl'):
            self.rl_learner.config["learning_enabled"] = True
            print("    RL: Mode advisor activé")
        
        print("    Modules déterministes: Actifs")
        
        self.last_health_check = time.time()
        self.last_status = time.time()
        self.last_heartbeat = time.time()
        
        if not self.simulation_mode:
            self.message_buffer.append(msg)
            self.ai_engine.add_sample(msg["features"])
        
        print(f"\n OBC PRÊT À FONCTIONNER")
        
        self._set_obc_state("IDLE")
    
    # ===================== TRAITEMENT DES MESSAGES =====================
    
    def _read_serial_data(self):
        """Lecture des données série"""
        if self.serial_conn and self.serial_conn.in_waiting > 0:
            try:
                raw = self.serial_conn.readline()
                line = raw.decode('utf-8', errors='ignore').strip()
                
                if line:
                    self.performance_stats["messages_received"] += 1
                    self._parse_and_buffer_message(line)
                    
            except UnicodeDecodeError:
                pass
            except Exception as e:
                if self.config['log_level'] == 'DEBUG':
                    print(f"DEBUG erreur lecture: {e}")
    
    def _parse_and_buffer_message(self, raw_line: str) -> Optional[Dict]:
        """Parse et bufferise un message"""
        msg = self._parse_message_passive(raw_line)
        
        if msg and not self.simulation_mode:
            self.message_buffer.append(msg)
            
            if len(self.message_buffer) > self.config['max_buffer_size']:
                self.message_buffer.pop(0)
            
            if self.ai_engine and self.ai_engine.ready:
                self.ai_engine.add_sample(msg["features"])
        
        return msg
    
    def _process_message_buffer(self):
        """Traite les messages en attente"""
        while self.message_buffer:
            msg = self.message_buffer.pop(0)
            self._process_single_message(msg)
    
    def _process_single_message(self, msg: Dict):
        """Traite un message individuel"""
        start_time = time.time()
        
        try:
            msg_type = msg.get("msg_type", "UNKNOWN")
            
            if msg_type == "HEARTBEAT":
                self._handle_heartbeat(msg)
            else:
                self.performance_stats["alerts_processed"] += 1
                self._handle_alert(msg)
            
            processing_time = time.time() - start_time
            self.performance_stats["avg_processing_time"] = (
                self.performance_stats["avg_processing_time"] * 0.9 + 
                processing_time * 0.1
            )
            
        except Exception as e:
            print(f" Erreur traitement message: {e}")
            if self.config['log_level'] == 'DEBUG':
                import traceback
                traceback.print_exc()
    
    def _handle_heartbeat(self, msg: Dict):
        """Gestion des heartbeats - CORRIGÉ"""
        self.last_heartbeat = time.time()
        
        if self.obc_state == "IDLE":
            if self.ai_engine and self.ai_engine.ready:
                self.ai_engine.add_sample(msg["features"])
            
            if msg["state"] in ["WARNING", "CRITICAL"]:
                self._set_obc_state("ACCUMULATING")
        
        elif self.obc_state == "ACCUMULATING":
            # CORRECTION CRITIQUE : Ne pas annuler CRITICAL sur un seul NORMAL
            if msg["state"] == "NORMAL":
                # Seulement pour WARNING, pas pour CRITICAL
                if self.current_episode_type == "WARNING":
                    print(f"     Annulation épisode WARNING: retour à NORMAL")
                    self._set_obc_state("IDLE")
                else:
                    # Pour CRITICAL, on continue l'accumulation
                    print(f"     CRITICAL en cours - Ignore NORMAL isolé")
        
        elif self.obc_state == "RECOVERY_MONITORING":
            self._handle_recovery_monitoring(msg)
        
        if self.config.get('buffer_only_alerts', False):
            if msg["state"] != "NORMAL" and self.ai_engine:
                self.ai_engine.add_sample(msg["features"])
        elif self.ai_engine:
            self.ai_engine.add_sample(msg["features"])
    
    def _handle_alert(self, msg: Dict):
        """Gestion d'une alerte"""
        print(f"\n" + "=" * 50)
        print(f" ALERTE MCU DETECTEE - OBC State: {self.obc_state}")
        print("=" * 50)
        
        print(f"\n{datetime.now().strftime('%H:%M:%S')}")
        print(f"État MCU: {msg['state']}")
        print(f"MSE MCU: {msg.get('mse', 0):.4f}")
        
        if self.obc_state == "IDLE":
            self._set_obc_state("ACCUMULATING")
            self.episode_buffer = [msg]
            self.current_episode_type = msg["state"]
            self.episode_start_time = time.time()
            print(f"\n DÉBUT ÉPISODE: {self.current_episode_type}")
            print(f"   Buffer: 1/30")
        
        elif self.obc_state == "ACCUMULATING":
            self.episode_buffer.append(msg)
            buffer_size = len(self.episode_buffer)
            print(f"\n   ACCUMULATION: {buffer_size}/30")
            
            if msg["state"] != self.current_episode_type:
                print(f"️    Changement type: {self.current_episode_type} → {msg['state']}")
                self.episode_buffer = [msg]
                self.current_episode_type = msg["state"]
                self.episode_start_time = time.time()
                return
            
            if buffer_size >= 30:
                self._set_obc_state("ANALYZING")
                self._perform_episode_analysis()
        
        elif self.obc_state in ["ANALYZING", "DECIDING"]:
            print(f"     Traitement en cours...")
        
        elif self.obc_state == "RECOVERY_MONITORING":
            print(f"\n RECOVERY MONITORING")
            print(f"   Alerte reçue pendant monitoring")
        
        elif self.obc_state == "ESCALATION":
            print(f"\n ESCALATION")
            print(f"   Alerte en mode urgence")
    
    # ===================== ANALYSE ÉPISODE =====================
    
    def _perform_episode_analysis(self):
        """Analyse complète de l'épisode"""
        print(f"\n ANALYSE ÉPISODE (30 échantillons)")
        
        buffer_str = str([msg["features"] for msg in self.episode_buffer])
        buffer_hash = hashlib.md5(buffer_str.encode()).hexdigest()[:8]
        
        if buffer_hash == self.last_buffer_hash:
            print(f"      Buffer déjà analysé")
            self._set_obc_state("IDLE")
            return
        
        self.last_buffer_hash = buffer_hash
        
        print(f"\n🤖 ANALYSE OBC (LSTM Autoencoder)...")
        try:
            features_list = [msg["features"] for msg in self.episode_buffer]
            sequence = np.array(features_list).reshape(1, 30, -1)
            
            # AFFICHER LES VALEURS MOYENNES
            print(f"   Valeurs moyennes sur l'épisode:")
            avg_features = np.mean(features_list, axis=0)
            for i, name in enumerate(FEATURE_NAMES):
                unit = "V" if "v_" in name else ("A" if "i_" in name else ("°C" if "t_" in name else ""))
                print(f"     {name:8}: {avg_features[i]:6.2f} {unit}")
            
            sequence_norm = self.ai_engine.scaler.transform(
                sequence.reshape(-1, sequence.shape[-1])
            )
            sequence_norm = sequence_norm.reshape(1, 30, -1)
            
            reconstructed = self.ai_engine.model.predict(sequence_norm, verbose=0)
            mse = np.mean((sequence_norm - reconstructed) ** 2)
            
            obc_state = "NORMAL"
            if mse > CRITICAL_THRESHOLD:
                obc_state = "CRITICAL"
            elif mse > WARNING_THRESHOLD:
                obc_state = "WARNING"
            
            print(f"\n   MSE OBC: {mse:.4f}")
            print(f"   Seuil CRITICAL: {CRITICAL_THRESHOLD:.4f}")
            print(f"   Ratio MSE/Seuil: {mse/CRITICAL_THRESHOLD:.1f}x")
            print(f"   État OBC: {obc_state}")
            
            self.episode_analysis = {
                "mse": mse,
                "obc_state": obc_state,
                "episode_type": self.current_episode_type,
                "buffer_hash": buffer_hash,
                "last_msg": self.episode_buffer[-1],
                "avg_features": avg_features.tolist()
            }
            
            self._set_obc_state("DECIDING")
            self._perform_episode_decision()
            
        except Exception as e:
            print(f" Erreur analyse OBC: {e}")
            self._set_obc_state("IDLE")
    
    def _perform_episode_decision(self):
        """Décision et actions pour l'épisode"""
        if not hasattr(self, 'episode_analysis'):
            print(f"     Aucune analyse disponible")
            self._set_obc_state("IDLE")
            return
        
        print(f"\n DÉCISION ÉPISODE")
        
        last_msg = self.episode_analysis["last_msg"]
        
        fault_diagnosis = diagnoser.diagnose(last_msg["features"], last_msg["state"])
        print(f"   Type défaut: {fault_diagnosis.fault_type}")
        print(f"   Confiance: {fault_diagnosis.confidence:.2%}")
        
        rl_state = RLState(
            features=np.array(last_msg["features"]),
            mcu_state=last_msg["state"],
            mcu_mse=last_msg.get("mse", 0.0),
            obc_mse=self.episode_analysis["mse"],
            fault_type=fault_diagnosis.fault_type,
            system_state=state_machine.current_state.name,
            episode_type=self.current_episode_type
        )
        
        self.last_rl_state = rl_state
        
        rl_suggestions = []
        if self.rl_learner and self.config.get('enable_rl', True):
            rl_suggestions = self.rl_learner.get_safe_suggestions(rl_state, n_suggestions=3)
            self.last_rl_suggestions = rl_suggestions
            
            if rl_suggestions:
                print(f"\n    Suggestions RL (sûres):")
                for i, (action, confidence, safety_info) in enumerate(rl_suggestions[:3], 1):
                    print(f"     {i}. {action} (confiance: {confidence:.1%})")
        
        rl_action_list = [(action, conf) for action, conf, _ in rl_suggestions]
        
        decision = decision_engine.make_decision(
            mcu_data=last_msg,
            obc_mse=self.episode_analysis["mse"],
            fault_diagnosis=fault_diagnosis,
            rl_suggestions=rl_action_list,
            debug=(self.config['log_level'] == 'DEBUG')
        )
        
        print(f"\n    Décision finale: {decision.decision_type}")
        
        if hasattr(decision, 'metadata') and decision.metadata.get('rl_influenced', False):
            print(f"    Influence RL détectée")
            self.rl_influence_active = True
            self.performance_stats["rl_suggestions_used"] += 1
        else:
            self.rl_influence_active = False
        
        actions = action_manager.determine_actions(
            decision=decision.decision_type,
            fault_diagnosis=fault_diagnosis
        )
        
        if actions and self.rl_learner:
            self.last_rl_action = RLAction(
                action_type=actions[0].action.value,
                priority=actions[0].priority,
                parameters=actions[0].parameters
            )
        
        self._print_decision_summary(decision, actions, rl_suggestions)
        
        print(f"\n EXECUTION ACTIONS...")
        action_executed = False
        for action_cmd in actions:
            if action_cmd.action != SystemAction.NO_ACTION:
                success = self._execute_and_send_action(action_cmd)
                if success:
                    print(f"    {action_cmd.action.value} → MCU")
                    action_executed = True
                    
                    if action_executed and self.rl_learner:
                        self._update_rl_learning(success, last_msg)
        
        if decision.decision_type == "CONFIRMED_CRITICAL":
            self.decision_locked = True
            self.lock_timestamp = time.time()
            print(f"\n VERROU APPLIQUE")
            print(f"   Décision critique confirmée")
        
        self._set_obc_state("RECOVERY_MONITORING")
    
    def _update_rl_learning(self, action_success: bool, mcu_msg: Dict):
        """Met à jour l'apprentissage RL - CORRIGÉ"""
        if not self.rl_learner or not self.config.get('enable_rl', True):
            return
        
        if self.last_rl_state is None or self.last_rl_action is None:
            return
        
        valid_state = RLState(
            features=np.array(mcu_msg["features"]),
            mcu_state=mcu_msg["state"],
            mcu_mse=mcu_msg.get("mse", 0.0),
            obc_mse=self.episode_analysis["mse"] if hasattr(self, 'episode_analysis') else mcu_msg.get("mse", 0.0),
            fault_type="BUS_OVERLOAD",
            system_state="CRITICAL",
            episode_type=mcu_msg["state"] if mcu_msg["state"] in ["WARNING", "CRITICAL"] else "NORMAL"
        )
        
        is_valid, reason = SafeRLValidator.validate_action(
            self.last_rl_action.action_type,
            valid_state,
            self.mission_phase
        )
        
        if not is_valid:
            print(f"    RL: Action non sûre ({reason}) - Expérience ignorée")
            return
        
        next_state = RLState(
            features=np.array(mcu_msg["features"]),
            mcu_state=mcu_msg["state"],
            mcu_mse=mcu_msg.get("mse", 0.0),
            obc_mse=self.episode_analysis["mse"] if hasattr(self, 'episode_analysis') else mcu_msg.get("mse", 0.0),
            fault_type="",
            system_state=state_machine.current_state.name,
            episode_type=mcu_msg["state"] if mcu_msg["state"] in ["WARNING", "CRITICAL"] else "NORMAL"
        )
        
        reward = self.rl_learner.calculate_mission_reward(
            state=self.last_rl_state,
            action=self.last_rl_action,
            next_state=next_state,
            success=action_success
        )
        
        experience = RLExperience(
            state=self.last_rl_state,
            action=self.last_rl_action,
            reward=reward,
            next_state=next_state,
            timestamp=time.time(),
            success=action_success,
            mission_phase=self.mission_phase
        )
        
        self.rl_learner.store_experience(experience)
        
        if len(self.rl_learner.experience_buffer) >= 32:
            self.rl_learner.learn_from_buffer(batch_size=16)
        
        print(f"    RL: Récompense={reward:.1f}, Expériences={len(self.rl_learner.experience_buffer)}")
    
    # ===================== RECOVERY MONITORING =====================
    
    def _handle_recovery_monitoring(self, msg: Dict):
        """Gestion recovery monitoring - CORRIGÉ"""
        self.recovery_monitor_count += 1
        elapsed = time.time() - self.recovery_start_time
        
        print(f"\n RECOVERY MONITOR #{self.recovery_monitor_count}")
        print(f"   Élapsed: {elapsed:.0f}/{self.RECOVERY_MONITOR_TIMEOUT}s")
        print(f"   État MCU: {msg['state']}")
        
        if elapsed > self.RECOVERY_MONITOR_TIMEOUT:
            print(f"     TIMEOUT - Passage en ESCALATION")
            self._set_obc_state("ESCALATION")
            return
        
        if msg["state"] == "NORMAL":
            self.recovery_stable_count += 1
            print(f"   Stable count: {self.recovery_stable_count}/{self.RECOVERY_STABLE_THRESHOLD}")
            
            if self.current_episode_type == "CRITICAL":
                required_stable = self.RECOVERY_STABLE_THRESHOLD * 2
                if self.recovery_stable_count >= required_stable:
                    print(f"     STABILISATION RÉUSSIE (CRITICAL) - Retour à IDLE")
                    self._set_obc_state("IDLE")
            else:
                if self.recovery_stable_count >= self.RECOVERY_STABLE_THRESHOLD:
                    print(f"     STABILISATION RÉUSSIE (WARNING) - Retour à IDLE")
                    self._set_obc_state("IDLE")
        else:
            self.recovery_stable_count = 0
            print(f"️    Alerte reçue - Reset stable count")
            
            if msg["state"] == "CRITICAL":
                print(f"     NOUVELLE ALERTE CRITIQUE - Passage en ESCALATION")
                self._set_obc_state("ESCALATION")
    
    # ===================== ACTIONS =====================
    
    def _execute_and_send_action(self, action_cmd) -> bool:
        """Exécute une action et l'envoie au MCU"""
        local_success = action_manager.execute_action(action_cmd)
        
        if not local_success:
            return False
        
        try:
            if self.serial_conn and self.serial_conn.is_open:
                cmd = f"CMD,{action_cmd.action.value},{int(time.time())}\n"
                self.serial_conn.write(cmd.encode())
                
                print(f"    Commande envoyée au MCU: {action_cmd.action.value}")
                return True
            else:
                print("    ERREUR: Port série non connecté")
                return False
                
        except Exception as e:
            print(f"    ERREUR envoi au MCU: {e}")
            return False
    
    def _send_escalation_command(self):
        """Envoie une commande d'escalation"""
        try:
            if self.serial_conn and self.serial_conn.is_open:
                cmd = f"CMD,FULL_SYSTEM_RESET,{int(time.time())},ESCALATION\n"
                self.serial_conn.write(cmd.encode())
                print(f"    Commande FULL_SYSTEM_RESET envoyée")
                
                time.sleep(2)
                self._set_obc_state("RECOVERY_MONITORING")
                
        except Exception as e:
            print(f"    ERREUR commande escalade: {e}")
    
    # ===================== AFFICHAGE =====================
    
    def _print_waiting_status(self, current_time: float):
        """Affiche le statut d'attente"""
        wait_time = current_time - self.waiting_start_time
        
        print(f"\n STATUT ATTENTE MCU - {datetime.now().strftime('%H:%M:%S')}")
        print(f"   Temps d'attente: {wait_time:.1f}s")
        print(f"   Lignes lues: {self.waiting_stats['lines_read']}")
        print(f"   Messages invalides: {self.waiting_stats['invalid_messages']}")
        
        if self.serial_conn:
            status = "OUVERT" if self.serial_conn.is_open else "FERMÉ"
            in_waiting = self.serial_conn.in_waiting
            print(f"   Port série: {status} ({in_waiting} bytes)")
        else:
            print(f"   Port série: NON CONNECTÉ")
        
        if self.simulation_mode:
            print(f"   Mode: SIMULATION")
        
        if self.debug_raw_data:
            print(f"\n   Premières lignes reçues:")
            for i, line in enumerate(self.debug_raw_data[:3], 1):
                print(f"     {i}. {line[:60]}...")
        
        print(f"\n   Prochain statut dans 5s...")
    
    def _print_system_status(self):
        """Affiche le statut système complet"""
        uptime = time.time() - self.performance_stats["start_time"]
        hours = int(uptime // 3600)
        minutes = int((uptime % 3600) // 60)
        seconds = int(uptime % 60)
        
        stats = decision_engine.get_statistics()
        
        print(f"\n" + "-" * 70)
        print(f" STATUT SYSTEME - {datetime.now().strftime('%H:%M:%S')}")
        print("-" * 70)
        
        print(f"\n ÉTAT OBC:")
        print(f"   {self.obc_state}")
        
        if self.obc_state == "ACCUMULATING":
            print(f"   Buffer épisode: {len(self.episode_buffer)}/30")
            print(f"   Type: {self.current_episode_type}")
        
        elif self.obc_state == "RECOVERY_MONITORING":
            elapsed = time.time() - self.recovery_start_time
            remaining = max(0, self.RECOVERY_MONITOR_TIMEOUT - elapsed)
            print(f"   Monitoring: {self.recovery_monitor_count} messages")
            print(f"   Stable: {self.recovery_stable_count}/{self.RECOVERY_STABLE_THRESHOLD}")
            print(f"   Restant: {remaining:.0f}s")
        
        elif self.obc_state == "ESCALATION":
            print(f"    MODE URGENCE ACTIF")
        
        print(f"\n  TEMPS:")
        print(f"   Uptime: {hours:02d}:{minutes:02d}:{seconds:02d}")
        
        print(f"\n PERFORMANCE:")
        print(f"   Messages: {self.performance_stats['messages_received']}")
        print(f"   Alertes: {self.performance_stats['alerts_processed']}")
        print(f"   Temps moyen: {self.performance_stats['avg_processing_time']*1000:.1f}ms")
        
        print(f"\n DÉCISIONS:")
        print(f"   Confirmées: {stats['confirmation_rate']:.1%}")
        print(f"   Faux positifs: {stats['false_positive_rate']:.1%}")
        print(f"   Critiques: {stats['critical']}")
        
        if self.rl_learner and self.config.get('enable_rl', True):
            print(f"\n RL ADVISOR:")
            rl_stats = self.rl_learner.get_policy_summary()
            print(f"   États appris: {rl_stats.get('total_states', 0)}")
            print(f"   Expériences: {rl_stats.get('total_experiences', 0)}")
            print(f"   Suggestions utilisées: {self.performance_stats['rl_suggestions_used']}")
            
            if 'top_actions' in rl_stats:
                print(f"   Actions fréquentes:")
                for action, count in rl_stats['top_actions'][:3]:
                    print(f"     • {action}: {count}")
        
        print(f"\n ÉTAT SYSTÈME:")
        state_info = state_machine.get_state_info()
        print(f"   {state_info['current_state']} ({state_info['duration_seconds']:.0f}s)")
        
        print(f"\n IA:")
        buffer_size = len(self.ai_engine.buffer) if self.ai_engine else 0
        print(f"   Buffer: {buffer_size}/{SEQUENCE_LENGTH}")
        ready = "PRÊT" if buffer_size == SEQUENCE_LENGTH else "EN ATTENTE"
        print(f"   Statut: {ready}")
        
        print(f"\n VERROU:")
        if self.decision_locked:
            lock_age = time.time() - self.lock_timestamp
            print(f"   ACTIF (depuis {lock_age:.0f}s)")
        else:
            print(f"   INACTIF")
        
        print(f"\n🔌 CONNEXION:")
        if self.first_message_received:
            contact_age = time.time() - self.mcu_first_contact
            print(f"   MCU: Connecté (depuis {contact_age:.1f}s)")
        else:
            print(f"   MCU:  En attente...")
        
        print(f"\n" + "-" * 70 + "\n")
    
    def _print_decision_summary(self, decision, actions, rl_suggestions=None):
        """Affiche un résumé de décision"""
        print("\n" + "=" * 70)
        print(" RESUME DECISION")
        print("=" * 70)
        
        print(f"\n DIAGNOSTIC FINAL:")
        print(f"   Severite:        {decision.severity}")
        print(f"   Type defaut:     {decision.fault_type}")
        print(f"   Confiance:       {decision.confidence:.2%}")
        print(f"   Etat systeme:    {decision.system_state}")
        print(f"   Type decision:   {decision.decision_type}")
        
        if hasattr(decision, 'metadata'):
            if decision.metadata.get('rl_influenced', False):
                print(f"   Influence RL:    ✓ ({decision.metadata.get('rl_action', 'N/A')})")
        
        print(f"\n INDICATEURS CLES:")
        indicators = decision.indicators
        print(f"   Vbatt: {indicators.get('v_batt', 0):.2f} V | "
              f"Ibatt: {indicators.get('i_batt', 0):.3f} A | "
              f"Tbatt: {indicators.get('t_batt', 0):.1f} °C")
        print(f"   Pbatt: {indicators.get('p_batt', 0):.2f} W | "
              f"Psolar: {indicators.get('p_solar', 0):.3f} W")
        
        if rl_suggestions and len(rl_suggestions) > 0:
            print(f"\n SUGGESTIONS RL DISPONIBLES:")
            for i, (action, confidence, safety_info) in enumerate(rl_suggestions[:3], 1):
                status = "✓" if i == 1 and self.rl_influence_active else " "
                print(f"   {status} {action:25} (confiance: {confidence:.1%})")
        
        print(f"\n ACTIONS PLANIFIEES:")
        if actions:
            for i, action in enumerate(actions, 1):
                print(f"   {i}. {action.action.value} (Prio: {action.priority}/10)")
                print(f"      -> {action.description}")
        else:
            print("   Aucune action requise")
        
        print(f"\n CONTEXTE SYSTEME:")
        state_info = state_machine.get_state_info()
        print(f"   Etat actuel:    {state_info['current_state']}")
        print(f"   Duree etat:     {state_info['duration_seconds']:.1f}s")
        print(f"   Phase mission:  {self.mission_phase}")
        
        print("\n" + "=" * 70)
    
    # ===================== SANTÉ SYSTÈME =====================
    
    def _check_system_health(self):
        """Vérification santé"""
        current_time = time.time()
        
        if not self.serial_conn or not self.serial_conn.is_open:
            if self.config.get('auto_reconnect', True) and not self.simulation_mode:
                print(" Port série fermé - tentative reconnexion...")
                self._connect_serial()
        
        if self.first_message_received and not self.simulation_mode:
            if current_time - self.last_heartbeat > 60:
                print("  Pas de heartbeat MCU depuis 60s")
                self.last_heartbeat = current_time
    
    # ===================== JOURNALISATION =====================
    
    def _log_event(self, decision, actions, msg, action_executed=True):
        """Journalise un événement"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "obc_state": self.obc_state,
            "decision": {
                "type": decision.decision_type,
                "severity": decision.severity,
                "fault_type": decision.fault_type,
                "confidence": decision.confidence,
            },
            "actions": [a.action.value for a in actions],
            "action_executed": action_executed,
            "mcu_state": msg["state"],
            "system_state": state_machine.current_state.name,
            "decision_locked": self.decision_locked,
            "rl_influenced": self.rl_influence_active,
        }
        
        self.system_log.append(log_entry)
        
        if len(self.system_log) % 10 == 0:
            self._save_log_to_file()
    
    def _save_log_to_file(self):
        """Sauvegarde le log"""
        try:
            with open("obc_system_log.json", "w") as f:
                json.dump(self.system_log, f, indent=2)
        except:
            pass
    
    # ===================== ARRÊT =====================
    
    def shutdown(self):
        """Arrêt propre"""
        print("\n" + "=" * 70)
        print(" ARRET DU SYSTEME OBC")
        print("=" * 70)
        
        self.running = False
        
        if self.rl_learner:
            self.rl_learner.save_model()
            print("    Modèle RL sauvegardé")
        
        if self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.close()
            print("    Port série fermé")
        
        self._save_log_to_file()
        
        uptime = time.time() - self.performance_stats["start_time"]
        stats = decision_engine.get_statistics()
        
        print(f"\n STATISTIQUES FINALES:")
        print(f"   Durée: {uptime:.1f}s")
        print(f"   Messages: {self.performance_stats['messages_received']}")
        print(f"   Alertes: {self.performance_stats['alerts_processed']}")
        print(f"   Confirmées: {stats['confirmed']} ({stats['confirmation_rate']:.1%})")
        
        if self.rl_learner:
            print(f"\n STATISTIQUES RL:")
            print(f"   Suggestions utilisées: {self.performance_stats['rl_suggestions_used']}")
            print(f"   États appris: {len(self.rl_learner.q_table)}")
            print(f"   Expériences: {self.rl_learner.stats['total_experiences']}")
        
        print(f"\n ÉTAT OBC FINAL: {self.obc_state}")
        
        if self.system_log:
            print(f"\n DERNIERS EVENEMENTS:")
            for event in self.system_log[-3:]:
                time_str = event['timestamp'].split('T')[1][:8]
                state = event.get('obc_state', 'UNKNOWN')
                decision = event.get('decision', {}).get('type', 'UNKNOWN')
                print(f"   {time_str}: {state} - {decision}")
        
        print(f"\n" + "=" * 70)
        print(f" SYSTEME OBC ARRETE")
        print("=" * 70)

# ===================== MAIN =====================

def main():
    """Point d'entrée principal"""
    print("\n" + "=" * 70)
    print(" OBC PROFESSIONNEL - EPS GUARDIAN")
    print("=" * 70)
    print("Architecture modulaire avec RL Advisor")
    print("Attente patiente du MCU - Pas de timeout fatal")
    print("Support CSV et JSON")
    print("=" * 70)
    
    system = ProfessionalOBCSystem()
    
    system.config.update({
        "serial_port": "COM6",
        "log_level": "INFO",
        "enable_rl": True,
        "rl_advisor_mode": True,
    })
    
    if not system.initialize():
        print("\n  Initialisation avec avertissements - Continuation...")
    
    system.run()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
