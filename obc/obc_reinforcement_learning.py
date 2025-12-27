"""
Couche d'apprentissage par renforcement pour OBC
Mode ADVISOR uniquement - Ne prend jamais de décision critique
"""

import numpy as np
import json
import pickle
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import hashlib
import time

@dataclass
class RLState:
    """État pour l'apprentissage par renforcement"""
    features: np.ndarray  # 7 features EPS
    mcu_state: str       # État MCU (NORMAL, WARNING, CRITICAL)
    mcu_mse: float       # MSE du MCU
    obc_mse: float       # MSE de l'OBC
    fault_type: str      # Type de défaut
    system_state: str    # État système (NOMINAL, SAFE, etc.)
    episode_type: str    # Type d'épisode (WARNING, CRITICAL)

@dataclass 
class RLAction:
    """Action pour l'apprentissage par renforcement"""
    action_type: str     # Type d'action (ENTER_SAFE_MODE, etc.)
    priority: int        # Priorité (1-10)
    parameters: Dict[str, float]  # Paramètres d'action

@dataclass
class RLExperience:
    """Expérience stockée pour l'apprentissage"""
    state: RLState
    action: RLAction
    reward: float
    next_state: RLState
    timestamp: float
    success: bool
    mission_phase: str   # Phase de mission

class SafeRLValidator:
    """Valide que les actions RL sont sûres avant exécution"""
    
    # Règles de sécurité invariantes
    SAFETY_RULES = {
        "MONITOR": {
            "allowed_states": ["NORMAL", "WARNING", "CRITICAL"],
            "min_battery": 0.05,
            "max_temp": 80,
        },
        "MONITOR_CLOSELY": {
            "allowed_states": ["WARNING", "CRITICAL"],
            "min_battery": 0.10,
            "max_temp": 75,
        },
        "ENTER_SAFE_MODE": {
            "allowed_states": ["WARNING", "CRITICAL"],
            "min_battery": 0.15,
            "max_temp": 65,
            "requires_confirmation": False,
        },
        "POWER_SAVING_MODE": {
            "allowed_states": ["WARNING", "NORMAL"],
            "min_battery": 0.20,
            "max_temp": 60,
            "requires_confirmation": False,
        },
        "ISOLATE_LOAD": {
            "allowed_states": ["WARNING", "CRITICAL"],
            "min_battery": 0.15,
            "max_temp": 70,
            "requires_confirmation": True,
        },
        "SWITCH_TO_BATTERY": {
            "allowed_states": ["WARNING", "CRITICAL"],
            "min_battery": 0.25,
            "max_temp": 60,
            "requires_confirmation": True,
        },
        "SHUTDOWN_NON_CRITICAL": {
            "allowed_states": ["CRITICAL"],
            "min_battery": 0.10,
            "max_temp": 75,
            "requires_confirmation": True,
        },
        "IGNORE_SENSOR": {
            "allowed_states": ["NORMAL", "WARNING"],
            "min_battery": 0.30,
            "max_temp": 50,
            "requires_confirmation": True,
        }
    }
    
    # Actions interdites dans certaines phases
    FORBIDDEN_ACTIONS_BY_PHASE = {
        "LAUNCH": ["SHUTDOWN_NON_CRITICAL", "IGNORE_SENSOR"],
        "ORBITAL": [],
        "ECLIPSE": ["IGNORE_SENSOR", "SWITCH_TO_BATTERY"],
        "SAFE_MODE": ["IGNORE_SENSOR"],
        "CRITICAL": ["IGNORE_SENSOR", "MONITOR"]  # En cas critique, ne pas ignorer
    }
    
    @staticmethod
    def validate_action(action: str, state: RLState, mission_phase: str = "ORBITAL") -> Tuple[bool, str]:
        """
        Vérifie si une action est sûre dans l'état courant
        Retourne: (est_valide, raison)
        """
        if action not in SafeRLValidator.SAFETY_RULES:
            return False, f"Action inconnue: {action}"
        
        rules = SafeRLValidator.SAFETY_RULES[action]
        
        # 1. Vérification état système
        if state.system_state not in rules.get("allowed_states", ["NORMAL"]):
            return False, f"État système {state.system_state} non autorisé pour {action}"
        
        # 2. Vérification phase de mission
        forbidden = SafeRLValidator.FORBIDDEN_ACTIONS_BY_PHASE.get(mission_phase, [])
        if action in forbidden:
            return False, f"Action {action} interdite en phase {mission_phase}"
        
        # 3. Vérification SOC (estimé)
        soc = SafeRLValidator._estimate_soc(state.features[0])
        min_soc = rules.get("min_battery", 0.1)
        if soc < min_soc:
            return False, f"SOC trop bas ({soc:.1%} < {min_soc:.1%})"
        
        # 4. Vérification température
        temp = state.features[2]  # t_batt
        max_temp = rules.get("max_temp", 70)
        if temp > max_temp:
            return False, f"Température trop élevée ({temp:.1f}°C > {max_temp}°C)"
        
        # 5. Vérification état MCU
        if state.mcu_state == "CRITICAL" and action == "MONITOR":
            return False, "Action MONITOR interdite en état CRITICAL MCU"
        
        return True, "OK"
    
    @staticmethod
    def _estimate_soc(v_batt: float) -> float:
        """Estime le SOC à partir de la tension batterie"""
        # Courbe de décharge Li-ion simplifiée
        if v_batt >= 8.2: return 1.0
        elif v_batt >= 7.8: return 0.85
        elif v_batt >= 7.4: return 0.65
        elif v_batt >= 7.0: return 0.40
        elif v_batt >= 6.6: return 0.20
        elif v_batt >= 6.4: return 0.10
        else: return 0.05
    
    @staticmethod
    def filter_safe_actions(actions: List[str], state: RLState, 
                           mission_phase: str = "ORBITAL") -> List[str]:
        """Filtre une liste d'actions pour ne garder que les sûres"""
        safe_actions = []
        for action in actions:
            is_valid, reason = SafeRLValidator.validate_action(action, state, mission_phase)
            if is_valid:
                safe_actions.append(action)
        return safe_actions

class OBCReinforcementLearner:
    """
    Apprentissage par renforcement pour améliorer les décisions OBC
    Mode ADVISOR uniquement - Ne prend jamais de décision seule
    """
    
    def __init__(self, config_path: str = "config/rl_config.json"):
        self.q_table = {}  # Table Q pour états discrets
        self.experience_buffer = []  # Buffer d'expériences
        self.learning_history = []   # Historique d'apprentissage
        
        # Hyperparamètres
        self.alpha = 0.1      # Taux d'apprentissage
        self.gamma = 0.9      # Facteur d'actualisation
        self.epsilon = 0.15   # Exploration vs exploitation
        self.max_experiences = 2000  # Taille max du buffer
        
        # Configuration
        self.config = self._load_config(config_path)
        
        # Validateur de sécurité
        self.validator = SafeRLValidator()
        
        # Statistiques
        self.stats = {
            "total_experiences": 0,
            "updates": 0,
            "rewards_total": 0,
            "best_action_count": 0,
            "exploration_count": 0,
            "exploitation_count": 0,
            "safe_actions_suggested": 0,
            "unsafe_actions_blocked": 0,
            "rl_influence_count": 0,
        }
        
        # Actions disponibles
        self.available_actions = [
            "MONITOR", "MONITOR_CLOSELY", "ENTER_SAFE_MODE",
            "POWER_SAVING_MODE", "ISOLATE_LOAD", "SWITCH_TO_BATTERY",
            "SHUTDOWN_NON_CRITICAL", "IGNORE_SENSOR"
        ]
        
        # Phase de mission courante
        self.current_mission_phase = "ORBITAL"
        
        print(f" Couche RL initialisée (Mode ADVISOR)")
        print(f"   Actions disponibles: {len(self.available_actions)}")
        print(f"   Sécurité: Activée")
        print(f"   Phase mission: {self.current_mission_phase}")
    
    def _load_config(self, config_path: str) -> Dict:
        """Charge la configuration RL"""
        default_config = {
            "learning_enabled": True,
            "advisor_mode": True,  # Mode advisor uniquement
            "alpha": 0.1,
            "gamma": 0.9,
            "epsilon": 0.15,
            "max_experiences": 2000,
            "min_confidence_threshold": 0.7,  # Seuil minimum pour considérer une suggestion
            "reward_config": {
                "successful_action": 15,
                "partial_success": 8,
                "no_change": 0,
                "worsening": -8,
                "critical_worsening": -20,
                "false_positive": -5,
                "true_positive": 10,
                "early_detection_bonus": 5,
                "conservative_penalty": -3,
                "safety_bonus": 10,  # Bonus pour actions sûres
                "mission_bonus": 5,   # Bonus pour adaptation à la mission
            },
            "state_discretization": {
                "mse_bins": 5,
                "soc_bins": 5,
                "temp_bins": 4,
                "fault_bins": 3
            },
            "exploration_schedule": {
                "initial_epsilon": 0.3,
                "final_epsilon": 0.05,
                "decay_steps": 1000
            },
        }
        
        try:
            with open(config_path, 'r') as f:
                user_config = json.load(f)
                default_config.update(user_config)
        except FileNotFoundError:
            print(f"  Fichier de config RL non trouvé, utilisation des valeurs par défaut")
        
        return default_config
    
    def set_mission_phase(self, phase: str):
        """Définit la phase de mission courante"""
        valid_phases = ["LAUNCH", "ORBITAL", "ECLIPSE", "SAFE_MODE", "CRITICAL"]
        if phase in valid_phases:
            self.current_mission_phase = phase
            print(f" RL: Phase mission changée → {phase}")
        else:
            print(f"  Phase mission invalide: {phase}")
    
    def discretize_state(self, state: RLState) -> str:
        """
        Discrétise l'état pour la table Q
        Retourne une clé de hachage pour l'état
        """
        # Extraire les features importantes
        features = state.features
        
        # SOC (à partir de v_batt)
        soc = self._estimate_soc(features[0])  # v_batt
        
        # Température
        temp = features[2]  # t_batt
        
        # MSE combiné
        mse_combined = (state.mcu_mse + state.obc_mse) / 2
        
        # Discrétisation
        soc_bin = int(min(soc * 5, 4))
        temp_bin = self._discretize_temp(temp)
        mse_bin = self._discretize_mse(mse_combined)
        
        # État MCU
        mcu_state_map = {"NORMAL": 0, "WARNING": 1, "CRITICAL": 2}
        mcu_state_code = mcu_state_map.get(state.mcu_state, 0)
        
        # Type de défaut (simplifié)
        fault_severity = self._get_fault_severity(state.fault_type)
        
        # État système
        system_state_map = {"NOMINAL": 0, "SAFE": 1, "DEGRADED": 2, "CRITICAL": 3}
        system_state_code = system_state_map.get(state.system_state, 0)
        
        # Créer la clé d'état
        state_key = f"{soc_bin}{temp_bin}{mse_bin}{mcu_state_code}{fault_severity}{system_state_code}"
        
        return state_key
    
    def _estimate_soc(self, v_batt: float) -> float:
        """Estime le SOC à partir de la tension batterie"""
        if v_batt >= 8.2: return 1.0
        elif v_batt >= 7.8: return 0.85
        elif v_batt >= 7.4: return 0.65
        elif v_batt >= 7.0: return 0.40
        elif v_batt >= 6.6: return 0.20
        elif v_batt >= 6.4: return 0.10
        else: return 0.05
    
    def _discretize_temp(self, temp: float) -> int:
        """Discrétise la température"""
        if temp < 0: return 0    # Très froid
        elif temp < 25: return 1 # Froid
        elif temp < 45: return 2 # Normal
        elif temp < 60: return 3 # Chaud
        else: return 4           # Très chaud
    
    def _discretize_mse(self, mse: float) -> int:
        """Discrétise le MSE"""
        if mse < 0.5: return 0   # Très bon
        elif mse < 1.0: return 1 # Bon
        elif mse < 1.5: return 2 # Moyen
        elif mse < 2.0: return 3 # Mauvais
        else: return 4           # Très mauvais
    
    def _get_fault_severity(self, fault_type: str) -> int:
        """Retourne la sévérité du défaut (0-2)"""
        critical_faults = ["BATTERY_OVERHEAT", "BATTERY_DEEP_DISCHARGE", 
                          "BATTERY_OVERCURRENT", "SOLAR_SHORT_CIRCUIT"]
        warning_faults = ["BATTERY_OVERVOLTAGE", "BATTERY_LOW_TEMP", 
                         "SOLAR_PANEL_FAILURE", "BUS_OVERLOAD"]
        
        if fault_type in critical_faults:
            return 2
        elif fault_type in warning_faults:
            return 1
        else:
            return 0
    
    def get_safe_suggestions(self, state: RLState, n_suggestions: int = 3) -> List[Tuple[str, float, str]]:
        """
        Retourne les suggestions RL SÛRES pour un état donné
        Format: (action, confidence, safety_info)
        """
        state_key = self.discretize_state(state)
        
        # Si l'état n'existe pas ou pas d'apprentissage
        if state_key not in self.q_table:
            return [("MONITOR", 0.5, "suggestion par défaut")]
        
        # 1. Obtenir toutes les actions avec leurs valeurs Q
        q_values = self.q_table[state_key]
        
        # 2. Trier par valeur Q décroissante
        sorted_actions = sorted(q_values.items(), key=lambda x: x[1], reverse=True)
        
        # 3. Filtrer et valider les actions sûres
        suggestions = []
        for action, q_value in sorted_actions:
            # Vérifier la sécurité
            is_valid, safety_info = self.validator.validate_action(
                action, state, self.current_mission_phase
            )
            
            if is_valid:
                # Convertir Q-value en confiance (0-1)
                confidence = min(max(q_value / 20.0, 0.1), 1.0)
                suggestions.append((action, confidence, safety_info))
                
                if len(suggestions) >= n_suggestions:
                    break
        
        # 4. Si pas de suggestions sûres, retourner monitor par défaut
        if not suggestions:
            self.stats["unsafe_actions_blocked"] += 1
            return [("MONITOR", 0.5, "pas de suggestion sûre disponible")]
        
        self.stats["safe_actions_suggested"] += len(suggestions)
        return suggestions
    
    def select_safe_action_for_exploration(self, state: RLState) -> Tuple[str, str]:
        """
        Sélectionne une action sûre pour exploration
        Utilisé uniquement pour générer de l'expérience
        """
        # Filtrer les actions sûres
        safe_actions = self.validator.filter_safe_actions(
            self.available_actions, state, self.current_mission_phase
        )
        
        if not safe_actions:
            return "MONITOR", "exploration safe par défaut"
        
        # Choisir aléatoirement parmi les actions sûres
        action = np.random.choice(safe_actions)
        return action, f"exploration safe parmi {len(safe_actions)} actions"
    
    def calculate_mission_reward(self, state: RLState, action: RLAction,
                               next_state: RLState, success: bool) -> float:
        """
        Calcule la récompense adaptée à la mission
        """
        reward = 0
        
        # Récompense de base pour succès
        if success:
            reward += self.config["reward_config"]["successful_action"]
        else:
            reward += self.config["reward_config"]["partial_success"]
        
        # Vérifier sécurité de l'action
        is_valid, safety_reason = self.validator.validate_action(
            action.action_type, state, self.current_mission_phase
        )
        
        if is_valid:
            reward += self.config["reward_config"]["safety_bonus"]
        else:
            reward -= 20  # Pénalité importante pour action non sûre
        
        # Récompense basée sur l'amélioration de l'état
        state_improvement = self._calculate_state_improvement(state, next_state)
        
        if state_improvement > 0.3:
            reward += 10
        elif state_improvement > 0.1:
            reward += 5
        elif state_improvement < -0.3:
            reward -= 15
        elif state_improvement < -0.1:
            reward -= 8
        
        # Bonus pour détection précoce
        if state.mcu_state == "WARNING" and next_state.mcu_state == "NORMAL":
            reward += self.config["reward_config"]["early_detection_bonus"]
        
        # Adaptation à la phase de mission
        if self._is_action_mission_appropriate(action.action_type):
            reward += self.config["reward_config"]["mission_bonus"]
        
        # Encourage la simplicité quand c'est possible
        if action.action_type == "MONITOR" and state.mcu_state == "NORMAL":
            reward += 2
        
        return reward
    
    def _calculate_state_improvement(self, state: RLState, next_state: RLState) -> float:
        """Calcule l'amélioration de l'état"""
        improvement = 0
        
        # Amélioration du MSE
        mse_improvement = state.obc_mse - next_state.obc_mse
        improvement += mse_improvement * 2
        
        # Amélioration de l'état MCU
        state_values = {"NORMAL": 0, "WARNING": 1, "CRITICAL": 2}
        current_val = state_values.get(state.mcu_state, 1)
        next_val = state_values.get(next_state.mcu_state, 1)
        improvement += (current_val - next_val) * 3
        
        # Amélioration SOC (estimée)
        current_soc = self._estimate_soc(state.features[0])
        next_soc = self._estimate_soc(next_state.features[0])
        improvement += (next_soc - current_soc) * 5
        
        return improvement / 10.0  # Normaliser
    
    def _is_action_mission_appropriate(self, action: str) -> bool:
        """Vérifie si l'action est appropriée pour la phase de mission"""
        inappropriate_actions = {
            "LAUNCH": ["SHUTDOWN_NON_CRITICAL", "IGNORE_SENSOR"],
            "ECLIPSE": ["IGNORE_SENSOR", "SWITCH_TO_BATTERY"],
            "SAFE_MODE": ["IGNORE_SENSOR"],
        }
        
        forbidden = inappropriate_actions.get(self.current_mission_phase, [])
        return action not in forbidden
    
    def store_experience(self, experience: RLExperience):
        """
        Stocke une expérience dans le buffer
        """
        # Vérifier que l'action était sûre
        is_valid, _ = self.validator.validate_action(
            experience.action.action_type,
            experience.state,
            experience.mission_phase
        )
        
        if not is_valid:
            print(f"  Expérience rejetée: action non sûre")
            return
        
        self.experience_buffer.append(experience)
        self.stats["total_experiences"] += 1
        
        # Limiter la taille du buffer
        if len(self.experience_buffer) > self.max_experiences:
            self.experience_buffer.pop(0)
        
        # Mettre à jour la table Q
        self._update_q_table(experience)
    
    def _update_q_table(self, experience: RLExperience):
        """
        Met à jour la table Q avec l'expérience
        """
        state_key = self.discretize_state(experience.state)
        next_state_key = self.discretize_state(experience.next_state)
        action = experience.action.action_type
        
        # Initialiser l'état si nécessaire
        if state_key not in self.q_table:
            self.q_table[state_key] = {}
        
        # Initialiser l'action si nécessaire
        if action not in self.q_table[state_key]:
            self.q_table[state_key][action] = 0
        
        # Valeur Q max pour le prochain état
        max_next_q = 0
        if next_state_key in self.q_table:
            # Ne prendre que les actions sûres pour le calcul
            safe_actions_next = self.validator.filter_safe_actions(
                list(self.q_table[next_state_key].keys()),
                experience.next_state,
                experience.mission_phase
            )
            if safe_actions_next:
                max_next_q = max(self.q_table[next_state_key][a] for a in safe_actions_next)
        
        # Mise à jour Q-learning
        current_q = self.q_table[state_key][action]
        new_q = current_q + self.alpha * (
            experience.reward + self.gamma * max_next_q - current_q
        )
        
        # Limiter les valeurs Q pour éviter l'explosion
        self.q_table[state_key][action] = np.clip(new_q, -50, 50)
        
        self.stats["updates"] += 1
        self.stats["rewards_total"] += experience.reward
        
        # Journaliser l'apprentissage
        self.learning_history.append({
            "timestamp": datetime.now().isoformat(),
            "state": state_key,
            "action": action,
            "reward": experience.reward,
            "new_q": new_q,
            "success": experience.success,
            "mission_phase": experience.mission_phase,
        })
    
    def learn_from_buffer(self, batch_size: int = 32):
        """
        Apprentissage par batch à partir du buffer d'expériences
        """
        if len(self.experience_buffer) < batch_size:
            return
        
        # Échantillonner aléatoirement du buffer
        indices = np.random.choice(len(self.experience_buffer), batch_size, replace=False)
        
        for idx in indices:
            exp = self.experience_buffer[idx]
            self._update_q_table(exp)
        
        print(f" RL: Apprentissage batch ({batch_size} expériences)")
    
    def get_policy_summary(self) -> Dict:
        """
        Retourne un résumé de la politique apprise
        """
        summary = {
            "total_states": len(self.q_table),
            "total_experiences": self.stats["total_experiences"],
            "updates": self.stats["updates"],
            "avg_reward": self.stats["rewards_total"] / max(self.stats["updates"], 1),
            "safe_actions_suggested": self.stats["safe_actions_suggested"],
            "unsafe_actions_blocked": self.stats["unsafe_actions_blocked"],
            "exploration_rate": self.stats["exploration_count"] / 
                              max(self.stats["exploration_count"] + self.stats["exploitation_count"], 1),
            "best_action_rate": self.stats["best_action_count"] / 
                              max(self.stats["exploitation_count"], 1),
            "advisor_mode": self.config.get("advisor_mode", True),
            "mission_phase": self.current_mission_phase,
        }
        
        # Actions les plus fréquentes
        action_counts = {}
        for state_actions in self.q_table.values():
            for action in state_actions:
                action_counts[action] = action_counts.get(action, 0) + 1
        
        if action_counts:
            summary["top_actions"] = sorted(
                action_counts.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:5]
        
        return summary
    
    def save_model(self, filepath: str = "models/obc_rl_model.pkl"):
        """Sauvegarde le modèle RL"""
        model_data = {
            "q_table": self.q_table,
            "experience_buffer": self.experience_buffer[:500],  # Garder seulement 500
            "learning_history": self.learning_history[-1000:],  # Garder 1000 derniers
            "stats": self.stats,
            "config": self.config,
            "mission_phase": self.current_mission_phase,
        }
        
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            print(f"  Modèle RL sauvegardé: {filepath}")
        except Exception as e:
            print(f"  Erreur sauvegarde RL: {e}")
    
    def load_model(self, filepath: str = "models/obc_rl_model.pkl"):
        """Charge le modèle RL"""
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.q_table = model_data.get("q_table", {})
            self.experience_buffer = model_data.get("experience_buffer", [])
            self.learning_history = model_data.get("learning_history", [])
            self.stats = model_data.get("stats", self.stats)
            self.config = model_data.get("config", self.config)
            self.current_mission_phase = model_data.get("mission_phase", "ORBITAL")
            
            print(f"   Modèle RL chargé: {filepath}")
            print(f"   États: {len(self.q_table)}, Expériences: {len(self.experience_buffer)}")
            print(f"   Phase mission: {self.current_mission_phase}")
            
        except FileNotFoundError:
            print(f"  Fichier modèle RL non trouvé: {filepath}")
        except Exception as e:
            print(f"  Erreur chargement RL: {e}")
    
    def print_status(self):
        """Affiche le statut du RL"""
        print(f"\n STATUT APPRENTISSAGE PAR RENFORCEMENT")
        print(f"   Mode: {'ADVISOR' if self.config.get('advisor_mode', True) else 'ACTOR'}")
        print(f"   États appris: {len(self.q_table)}")
        print(f"   Expériences: {self.stats['total_experiences']}")
        print(f"   Actions sûres suggérées: {self.stats['safe_actions_suggested']}")
        print(f"   Actions bloquées: {self.stats['unsafe_actions_blocked']}")
        
        # Politique actuelle (exemples)
        if len(self.q_table) > 0:
            print(f"\n   Exemple de politique (top 3 états):")
            states_sample = list(self.q_table.keys())[:3]
            for state in states_sample:
                # Trouver la meilleure action sûre
                safe_actions = []
                for action in self.q_table[state]:
                    # Simulation d'un état minimal pour validation
                    dummy_state = RLState(
                        features=np.array([7.5, 1.0, 25.0, 10.0, 5.0, 0.1, 0.5]),
                        mcu_state="NORMAL",
                        mcu_mse=0.5,
                        obc_mse=0.5,
                        fault_type="NONE",
                        system_state="NOMINAL",
                        episode_type="NORMAL"
                    )
                    is_valid, _ = self.validator.validate_action(action, dummy_state)
                    if is_valid:
                        safe_actions.append((action, self.q_table[state][action]))
                
                if safe_actions:
                    best_action = max(safe_actions, key=lambda x: x[1])
                    print(f"     État {state[:6]}... → {best_action[0]} (Q={best_action[1]:.2f})")