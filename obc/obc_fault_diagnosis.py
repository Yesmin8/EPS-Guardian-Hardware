"""
Module de diagnostic des défauts EPS
Basé sur les règles physiques et l'analyse des features
"""

import numpy as np
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
import obc_config as config
PHYSICAL_LIMITS = config.BATTERY_CONFIG  # ou créez un mapping
FEATURE_NAMES = ["v_batt", "i_batt", "t_batt", "v_solar", "i_solar", "v_bus", "i_load"]

@dataclass
class FaultDiagnosis:
    """Résultat de diagnostic"""
    fault_type: str
    confidence: float
    severity: str  # MINOR, MODERATE, SEVERE, CRITICAL
    affected_features: List[str]
    description: str
    recommendation: str

class FaultDiagnoser:
    """Diagnostiqueur de défauts EPS"""
    
    def __init__(self):
        self.diagnosis_history = []
        self.fault_patterns = self._load_fault_patterns()
        
    def _load_fault_patterns(self) -> Dict[str, Dict]:
        """Charge les patterns de défauts connus"""
        return {
            "BATTERY_OVERHEAT": {
                "conditions": [
                    lambda f: f[2] > 60,  # t_batt > 60°C
                    lambda f: f[1] > 2.0,  # i_batt > 2A
                ],
                "severity": "CRITICAL",
                "description": "Surchauffe batterie détectée",
                "recommendation": "Réduire charge, activer refroidissement"
            },
            "BATTERY_UNDERVOLTAGE": {
                "conditions": [
                    lambda f: f[0] < 6.5,  # v_batt < 6.5V
                    lambda f: f[1] > 0.5,   # i_batt > 0.5A (décharge)
                ],
                "severity": "SEVERE",
                "description": "Tension batterie trop basse",
                "recommendation": "Charger batterie, réduire consommation"
            },
            "SOLAR_PANEL_FAILURE": {
                "conditions": [
                    lambda f: f[3] < 5.0,   # v_solar < 5V
                    lambda f: f[4] < 0.1,   # i_solar < 0.1A
                    lambda f: f[0] > 7.0,   # v_batt > 7V (batterie OK)
                ],
                "severity": "MODERATE",
                "description": "Panneau solaire non fonctionnel",
                "recommendation": "Basculer sur batterie, vérifier connexions"
            },
            "BUS_OVERLOAD": {
                "conditions": [
                    lambda f: f[6] > 1.5,   # i_load > 1.5A
                    lambda f: f[5] < 6.8,   # v_bus < 6.8V
                ],
                "severity": "SEVERE",
                "description": "Surcharge bus principal",
                "recommendation": "Isoler charges non critiques"
            },
            "BATTERY_OVERVOLTAGE": {
                "conditions": [
                    lambda f: f[0] > 8.2,   # v_batt > 8.2V
                    lambda f: f[4] > 2.0,   # i_solar > 2A
                ],
                "severity": "MODERATE",
                "description": "Surtension batterie",
                "recommendation": "Limiter charge solaire"
            },
            "NORMAL": {
                "conditions": [],
                "severity": "MINOR",
                "description": "Système nominal",
                "recommendation": "Continuer monitoring"
            }
        }
    
    def diagnose(self, features: List[float], mcu_state: str) -> FaultDiagnosis:
        """
        Diagnostique un défaut basé sur les features
        
        Args:
            features: Liste de 7 valeurs
            mcu_state: État rapporté par le MCU
            
        Returns:
            FaultDiagnosis: Diagnostic complet
        """
        if len(features) != 7:
            return self._create_unknown_diagnosis()
        
        # Vérifier chaque pattern de défaut
        best_match = None
        highest_confidence = 0.0
        
        for fault_type, pattern in self.fault_patterns.items():
            if fault_type == "NORMAL":
                continue
                
            confidence = self._calculate_fault_confidence(features, pattern["conditions"])
            
            if confidence > highest_confidence and confidence > 0.5:
                highest_confidence = confidence
                best_match = (fault_type, pattern)
        
        # Si pas de défaut détecté
        if best_match is None:
            return FaultDiagnosis(
                fault_type="NORMAL",
                confidence=0.9,
                severity="MINOR",
                affected_features=[],
                description="Système nominal",
                recommendation="Continuer monitoring"
            )
        
        fault_type, pattern = best_match
        
        # Déterminer les features affectées
        affected_features = self._identify_affected_features(features, fault_type)
        
        # Créer le diagnostic
        diagnosis = FaultDiagnosis(
            fault_type=fault_type,
            confidence=highest_confidence,
            severity=pattern["severity"],
            affected_features=affected_features,
            description=pattern["description"],
            recommendation=pattern["recommendation"]
        )
        
        # Ajouter à l'historique
        self.diagnosis_history.append(diagnosis)
        if len(self.diagnosis_history) > 100:
            self.diagnosis_history.pop(0)
        
        return diagnosis
    
    def _calculate_fault_confidence(self, features: List[float], 
                                  conditions: List) -> float:
        """Calcule la confiance d'un défaut"""
        if not conditions:
            return 0.0
        
        confidence = 0.0
        weights = [1.0, 0.8, 0.6]  # Poids décroissants
        
        for i, condition in enumerate(conditions[:3]):
            try:
                if condition(features):
                    confidence += weights[min(i, len(weights)-1)]
            except Exception:
                continue
        
        return min(confidence / len(conditions), 1.0)
    
    def _identify_affected_features(self, features: List[float], 
                                   fault_type: str) -> List[str]:
        """Identifie les features affectées par un défaut"""
        affected = []
        
        # Mapping défaut -> features concernées
        fault_to_features = {
            "BATTERY_OVERHEAT": ["t_batt", "i_batt"],
            "BATTERY_UNDERVOLTAGE": ["v_batt", "i_batt"],
            "SOLAR_PANEL_FAILURE": ["v_solar", "i_solar"],
            "BUS_OVERLOAD": ["i_load", "v_bus"],
            "BATTERY_OVERVOLTAGE": ["v_batt", "i_solar"],
        }
        
        feature_indices = fault_to_features.get(fault_type, [])
        for feature_name in feature_indices:
            if feature_name in FEATURE_NAMES:
                affected.append(feature_name)
        
        return affected
    
    def _create_unknown_diagnosis(self) -> FaultDiagnosis:
        """Crée un diagnostic inconnu"""
        return FaultDiagnosis(
            fault_type="UNKNOWN",
            confidence=0.0,
            severity="MODERATE",
            affected_features=[],
            description="Données insuffisantes pour diagnostic",
            recommendation="Collecter plus de données"
        )
    
    def get_diagnosis_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques de diagnostic"""
        if not self.diagnosis_history:
            return {"total": 0, "by_severity": {}, "by_type": {}}
        
        stats = {
            "total": len(self.diagnosis_history),
            "by_severity": {},
            "by_type": {}
        }
        
        for diagnosis in self.diagnosis_history:
            # Par sévérité
            stats["by_severity"][diagnosis.severity] = \
                stats["by_severity"].get(diagnosis.severity, 0) + 1
            
            # Par type
            stats["by_type"][diagnosis.fault_type] = \
                stats["by_type"].get(diagnosis.fault_type, 0) + 1
        
        return stats

# Instance globale
diagnoser = FaultDiagnoser()