"""
Machine à états du système OBC
"""

import time
from enum import Enum
from typing import Dict, Any, Optional
from dataclasses import dataclass

class OBCState(Enum):
    """États du système OBC"""
    BOOT = "BOOT"
    WAITING_MCU = "WAITING_MCU"
    IDLE = "IDLE"
    ACCUMULATING = "ACCUMULATING"
    ANALYZING = "ANALYZING"
    DECIDING = "DECIDING"
    RECOVERY_MONITORING = "RECOVERY_MONITORING"
    ESCALATION = "ESCALATION"

@dataclass
class StateInfo:
    """Informations sur un état"""
    name: str
    entered_at: float
    description: str
    allowed_transitions: list

class StateMachine:
    """Machine à états OBC"""
    
    def __init__(self):
        self.current_state = OBCState.BOOT
        self.state_history = []
        self.transition_rules = self._create_transition_rules()
        
        # Enregistrer l'état initial
        self._enter_state(OBCState.BOOT)
    
    def _create_transition_rules(self) -> Dict[OBCState, Dict[str, Any]]:
        """Crée les règles de transition"""
        return {
            OBCState.BOOT: {
                "description": "Démarrage système",
                "allowed_transitions": [OBCState.WAITING_MCU],
                "timeout": None
            },
            OBCState.WAITING_MCU: {
                "description": "Attente premier contact MCU",
                "allowed_transitions": [OBCState.IDLE],
                "timeout": None  # Attente illimitée
            },
            OBCState.IDLE: {
                "description": "En attente d'alerte",
                "allowed_transitions": [OBCState.ACCUMULATING, OBCState.WAITING_MCU],
                "timeout": None
            },
            OBCState.ACCUMULATING: {
                "description": "Accumulation données épisode",
                "allowed_transitions": [OBCState.ANALYZING, OBCState.IDLE],
                "timeout": 60  # 60 secondes max
            },
            OBCState.ANALYZING: {
                "description": "Analyse IA de l'épisode",
                "allowed_transitions": [OBCState.DECIDING, OBCState.IDLE],
                "timeout": 10  # 10 secondes max
            },
            OBCState.DECIDING: {
                "description": "Prise de décision",
                "allowed_transitions": [OBCState.RECOVERY_MONITORING, OBCState.IDLE],
                "timeout": 5  # 5 secondes max
            },
            OBCState.RECOVERY_MONITORING: {
                "description": "Surveillance post-action",
                "allowed_transitions": [OBCState.IDLE, OBCState.ESCALATION],
                "timeout": 300  # 5 minutes max
            },
            OBCState.ESCALATION: {
                "description": "Mode urgence",
                "allowed_transitions": [OBCState.RECOVERY_MONITORING, OBCState.WAITING_MCU],
                "timeout": 30  # 30 secondes max
            }
        }
    
    def _enter_state(self, state: OBCState):
        """Entre dans un nouvel état"""
        state_info = StateInfo(
            name=state.value,
            entered_at=time.time(),
            description=self.transition_rules[state]["description"],
            allowed_transitions=[s.value for s in self.transition_rules[state]["allowed_transitions"]]
        )
        
        self.state_history.append(state_info)
        if len(self.state_history) > 50:
            self.state_history.pop(0)
        
        print(f"\n ENTREE ETAT: {state.value}")
        print(f"   Description: {state_info.description}")
        print(f"   Transitions autorisées: {', '.join(state_info.allowed_transitions)}")
    
    def transition_to(self, new_state: OBCState) -> bool:
        """
        Tente une transition d'état
        
        Args:
            new_state: Nouvel état souhaité
            
        Returns:
            bool: Succès de la transition
        """
        current_rules = self.transition_rules[self.current_state]
        
        # Vérifier si la transition est autorisée
        if new_state not in current_rules["allowed_transitions"]:
            print(f"  Transition refusée: {self.current_state.value} → {new_state.value}")
            print(f"   Transitions autorisées: {[s.value for s in current_rules['allowed_transitions']]}")
            return False
        
        # Vérifier timeout si applicable
        if current_rules["timeout"] is not None:
            time_in_state = time.time() - self.state_history[-1].entered_at
            if time_in_state > current_rules["timeout"]:
                print(f"  Timeout état {self.current_state.value} ({time_in_state:.1f}s)")
                # Forcer la transition vers un état de récupération
                if self.current_state != OBCState.ESCALATION:
                    new_state = OBCState.ESCALATION
        
        # Effectuer la transition
        old_state = self.current_state
        self.current_state = new_state
        
        print(f"\n TRANSITION: {old_state.value} → {new_state.value}")
        
        # Enregistrer le nouvel état
        self._enter_state(new_state)
        
        return True
    
    def force_transition(self, new_state: OBCState, reason: str = ""):
        """
        Force une transition (pour urgences)
        
        Args:
            new_state: Nouvel état
            reason: Raison de la force
        """
        old_state = self.current_state
        self.current_state = new_state
        
        print(f"\n  TRANSITION FORCÉE: {old_state.value} → {new_state.value}")
        if reason:
            print(f"   Raison: {reason}")
        
        self._enter_state(new_state)
    
    def get_state_info(self) -> Dict[str, Any]:
        """Retourne les informations sur l'état courant"""
        if not self.state_history:
            return {"current_state": "UNKNOWN", "duration_seconds": 0}
        
        current = self.state_history[-1]
        duration = time.time() - current.entered_at
        
        return {
            "current_state": current.name,
            "duration_seconds": duration,
            "description": current.description,
            "time_in_state": f"{duration:.1f}s",
            "allowed_transitions": current.allowed_transitions
        }
    
    def get_state_history(self, count: int = 10) -> list:
        """Retourne l'historique des états"""
        return self.state_history[-count:] if len(self.state_history) >= count else self.state_history
    
    def check_timeout(self) -> Optional[OBCState]:
        """
        Vérifie les timeouts d'état
        
        Returns:
            Optional[OBCState]: Nouvel état si timeout, sinon None
        """
        current_rules = self.transition_rules[self.current_state]
        
        if current_rules["timeout"] is None:
            return None
        
        time_in_state = time.time() - self.state_history[-1].entered_at
        
        if time_in_state > current_rules["timeout"]:
            print(f" Timeout état {self.current_state.value} ({time_in_state:.1f}s)")
            
            # Déterminer le prochain état selon les règles
            if self.current_state == OBCState.RECOVERY_MONITORING:
                return OBCState.ESCALATION
            elif self.current_state == OBCState.ESCALATION:
                return OBCState.WAITING_MCU
            elif self.current_state in [OBCState.ACCUMULATING, OBCState.ANALYZING, OBCState.DECIDING]:
                return OBCState.IDLE
        
        return None

# Instance globale
state_machine = StateMachine()