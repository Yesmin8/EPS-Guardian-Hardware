"""
Gestionnaire d'actions système
Traduit les décisions en actions concrètes
"""
import time  
from enum import Enum
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from obc_fault_diagnosis import FaultDiagnosis

class SystemAction(Enum):
    """Actions système disponibles"""
    NO_ACTION = "NO_ACTION"
    MONITOR = "MONITOR"
    MONITOR_CLOSELY = "MONITOR_CLOSELY"
    ENTER_SAFE_MODE = "ENTER_SAFE_MODE"
    POWER_SAVING_MODE = "POWER_SAVING_MODE"
    ISOLATE_LOAD = "ISOLATE_LOAD"
    SWITCH_TO_BATTERY = "SWITCH_TO_BATTERY"
    SHUTDOWN_NON_CRITICAL = "SHUTDOWN_NON_CRITICAL"
    IGNORE_SENSOR = "IGNORE_SENSOR"
    FULL_SYSTEM_RESET = "FULL_SYSTEM_RESET"

@dataclass
class ActionCommand:
    """Commande d'action complète"""
    action: SystemAction
    priority: int  # 1-10, 10 = plus urgent
    parameters: Dict[str, Any]
    description: str
    expected_duration: float  # secondes

class ActionManager:
    """Gestionnaire d'actions"""
    
    def __init__(self):
        self.action_history = []
        self.action_mapping = self._create_action_mapping()
    
    def _create_action_mapping(self) -> Dict[str, List[SystemAction]]:
        """Crée le mapping décision -> actions"""
        return {
            "CONFIRMED_CRITICAL": [
                SystemAction.ENTER_SAFE_MODE,
                SystemAction.ISOLATE_LOAD,
                SystemAction.SHUTDOWN_NON_CRITICAL
            ],
            "CRITICAL_RULE_R1": [
                SystemAction.ENTER_SAFE_MODE,
                SystemAction.SHUTDOWN_NON_CRITICAL
            ],
            "CRITICAL_RULE_R2": [
                SystemAction.SWITCH_TO_BATTERY,
                SystemAction.POWER_SAVING_MODE
            ],
            "CRITICAL_RULE_R3": [
                SystemAction.ISOLATE_LOAD,
                SystemAction.ENTER_SAFE_MODE
            ],
            "CONFIRMED_WARNING": [
                SystemAction.MONITOR_CLOSELY,
                SystemAction.POWER_SAVING_MODE
            ],
            "CONFIRMED_NORMAL": [
                SystemAction.MONITOR
            ],
            "MCU_CRITICAL_CONFLICT": [
                SystemAction.MONITOR_CLOSELY,
                SystemAction.ENTER_SAFE_MODE
            ],
            "REJECTED_MCU_WARNING": [
                SystemAction.MONITOR,
                SystemAction.IGNORE_SENSOR
            ],
            "NORMAL_RL_MONITOR": [
                SystemAction.MONITOR
            ],
            "NORMAL_RL_POWER_SAVING_MODE": [
                SystemAction.POWER_SAVING_MODE
            ]
        }
    
    def determine_actions(self, decision: str, 
                         fault_diagnosis: FaultDiagnosis) -> List[ActionCommand]:
        """
        Détermine les actions à partir d'une décision
        
        Args:
            decision: Type de décision
            fault_diagnosis: Diagnostic associé
            
        Returns:
            List[ActionCommand]: Liste d'actions ordonnées
        """
        # Actions par défaut
        if decision not in self.action_mapping:
            return [self._create_monitor_command()]
        
        # Obtenir les actions de base
        base_actions = self.action_mapping[decision]
        
        # Adapter en fonction du diagnostic
        adapted_actions = self._adapt_actions_to_fault(
            base_actions, fault_diagnosis
        )
        
        # Créer les commandes
        commands = []
        for i, action in enumerate(adapted_actions):
            command = self._create_action_command(
                action=action,
                priority=self._calculate_priority(decision, i),
                fault_diagnosis=fault_diagnosis
            )
            commands.append(command)
        
        return commands
    
    def _adapt_actions_to_fault(self, actions: List[SystemAction],
                               fault_diagnosis: FaultDiagnosis) -> List[SystemAction]:
        """Adapte les actions au type de défaut"""
        adapted = actions.copy()
        
        # Adaptation pour surchauffe batterie
        if fault_diagnosis.fault_type == "BATTERY_OVERHEAT":
            if SystemAction.SWITCH_TO_BATTERY in adapted:
                adapted.remove(SystemAction.SWITCH_TO_BATTERY)
            if SystemAction.POWER_SAVING_MODE not in adapted:
                adapted.insert(1, SystemAction.POWER_SAVING_MODE)
        
        # Adaptation pour panneau solaire
        elif fault_diagnosis.fault_type == "SOLAR_PANEL_FAILURE":
            if SystemAction.SWITCH_TO_BATTERY not in adapted:
                adapted.insert(0, SystemAction.SWITCH_TO_BATTERY)
        
        return adapted
    
    def _create_action_command(self, action: SystemAction, priority: int,
                              fault_diagnosis: FaultDiagnosis) -> ActionCommand:
        """Crée une commande d'action complète"""
        
        # Paramètres par défaut
        base_params = {
            "fault_type": fault_diagnosis.fault_type,
            "confidence": fault_diagnosis.confidence,
            "timestamp": time.time()
        }
        
        # Paramètres spécifiques par action
        action_params = {
            SystemAction.NO_ACTION: {
                "params": {},
                "desc": "Aucune action requise",
                "duration": 0.0
            },
            SystemAction.MONITOR: {
                "params": {"interval": 1.0, "duration": 60.0},
                "desc": "Surveillance standard",
                "duration": 60.0
            },
            SystemAction.MONITOR_CLOSELY: {
                "params": {"interval": 0.5, "duration": 300.0},
                "desc": "Surveillance accrue",
                "duration": 300.0
            },
            SystemAction.ENTER_SAFE_MODE: {
                "params": {"level": "FULL", "preserve_data": True},
                "desc": "Passage en mode sécurité",
                "duration": 10.0
            },
            SystemAction.POWER_SAVING_MODE: {
                "params": {"reduction": 50, "non_critical": True},
                "desc": "Réduction consommation énergie",
                "duration": 30.0
            },
            SystemAction.ISOLATE_LOAD: {
                "params": {"loads": ["NON_CRITICAL"], "duration": 300},
                "desc": "Isolement charges non critiques",
                "duration": 5.0
            },
            SystemAction.SWITCH_TO_BATTERY: {
                "params": {"source": "BATTERY_ONLY", "duration": 600},
                "desc": "Basculer sur alimentation batterie",
                "duration": 2.0
            },
            SystemAction.SHUTDOWN_NON_CRITICAL: {
                "params": {"systems": ["COMM", "PAYLOAD"], "graceful": True},
                "desc": "Arrêt systèmes non critiques",
                "duration": 15.0
            },
            SystemAction.IGNORE_SENSOR: {
                "params": {"sensor_id": "UNKNOWN", "duration": 3600},
                "desc": "Ignorer capteur défaillant",
                "duration": 1.0
            },
            SystemAction.FULL_SYSTEM_RESET: {
                "params": {"level": "HARD", "backup": True},
                "desc": "Reset complet du système",
                "duration": 30.0
            }
        }
        
        config = action_params.get(action, action_params[SystemAction.NO_ACTION])
        
        # Fusionner les paramètres
        parameters = {**base_params, **config["params"]}
        
        return ActionCommand(
            action=action,
            priority=priority,
            parameters=parameters,
            description=config["desc"],
            expected_duration=config["duration"]
        )
    
    def _create_monitor_command(self) -> ActionCommand:
        """Crée une commande de monitoring par défaut"""
        return ActionCommand(
            action=SystemAction.MONITOR,
            priority=1,
            parameters={"interval": 1.0, "duration": 60.0},
            description="Surveillance standard",
            expected_duration=60.0
        )
    
    def _calculate_priority(self, decision: str, action_index: int) -> int:
        """Calcule la priorité d'une action"""
        base_priority = {
            "CONFIRMED_CRITICAL": 10,
            "CRITICAL_RULE_R1": 10,
            "CRITICAL_RULE_R2": 9,
            "CRITICAL_RULE_R3": 9,
            "CONFIRMED_WARNING": 7,
            "MCU_CRITICAL_CONFLICT": 8,
            "REJECTED_MCU_WARNING": 3,
            "CONFIRMED_NORMAL": 1,
            "NORMAL_RL_MONITOR": 1,
            "NORMAL_RL_POWER_SAVING_MODE": 2
        }.get(decision, 5)
        
        # Réduire la priorité pour les actions secondaires
        return max(base_priority - action_index, 1)
    
    def execute_action(self, command: ActionCommand) -> bool:
        """
        Exécute une action localement (simulation)
        
        Args:
            command: Commande à exécuter
            
        Returns:
            bool: Succès de l'exécution
        """
        try:
            # Simulation d'exécution
            action = command.action.value
            params = command.parameters
            
            # Log de l'action
            action_log = {
                "timestamp": time.time(),
                "action": action,
                "priority": command.priority,
                "parameters": params,
                "description": command.description
            }
            
            self.action_history.append(action_log)
            if len(self.action_history) > 100:
                self.action_history.pop(0)
            
            print(f"    Action exécutée: {action}")
            if command.priority >= 7:
                print(f"      {command.description}")
            
            return True
            
        except Exception as e:
            print(f"    Erreur exécution action: {e}")
            return False
    
    def get_action_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques d'actions"""
        if not self.action_history:
            return {"total": 0, "by_type": {}, "by_priority": {}}
        
        stats = {
            "total": len(self.action_history),
            "by_type": {},
            "by_priority": {},
            "recent": self.action_history[-5:] if len(self.action_history) >= 5 else self.action_history
        }
        
        for log in self.action_history:
            # Par type
            action_type = log["action"]
            stats["by_type"][action_type] = stats["by_type"].get(action_type, 0) + 1
            
            # Par priorité
            priority = log["priority"]
            stats["by_priority"][priority] = stats["by_priority"].get(priority, 0) + 1
        
        return stats

# Instance globale
action_manager = ActionManager()