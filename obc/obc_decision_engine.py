"""
Moteur de décision hybride
Combine règles déterministes, IA et suggestions RL
"""

from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import time
from obc_config import DECISION_CONFIDENCE_THRESHOLD
from obc_fault_diagnosis import FaultDiagnosis

@dataclass
class Decision:
    """Décision finale"""
    decision_type: str
    severity: str
    fault_type: str
    confidence: float
    system_state: str
    indicators: Dict[str, float]
    metadata: Dict[str, Any]

class DecisionEngine:
    """Moteur de décision principal"""
    
    def __init__(self):
        self.decision_history = []
        self.statistics = {
            "total_decisions": 0,
            "confirmed": 0,
            "rejected": 0,
            "critical": 0,
            "warning": 0,
            "normal": 0,
            "rl_influenced": 0
        }
        
        # Seuils de décision
        self.thresholds = {
            "critical_mse": 1.5,
            "warning_mse": 1.0,
            "min_confidence": 0.6,
            "rl_confidence_threshold": 0.7
        }
        
        # Règles déterministes prioritaires (R1, R2, R3)
        self.critical_rules = [
            self._check_rule_battery_overheat,
            self._check_rule_battery_deep_discharge,
            self._check_rule_system_critical
        ]
    
    def _check_rule_battery_overheat(self, features: List[float]) -> bool:
        """Règle R1: Surchauffe batterie"""
        # t_batt > 65°C ET i_batt > 1.0A (décharge)
        return features[2] > 65 and features[1] > 1.0
    
    def _check_rule_battery_deep_discharge(self, features: List[float]) -> bool:
        """Règle R2: Décharge profonde"""
        # v_batt < 6.2V ET i_batt > 0.2A (décharge)
        return features[0] < 6.2 and features[1] > 0.2
    
    def _check_rule_system_critical(self, features: List[float]) -> bool:
        """Règle R3: Court-circuit possible"""
        # i_load > 2.5A ET v_bus < 6.0V
        return features[6] > 2.5 and features[5] < 6.0
    
    def make_decision(self, 
                     mcu_data: Dict[str, Any],
                     obc_mse: float,
                     fault_diagnosis: FaultDiagnosis,
                     rl_suggestions: Optional[List[Tuple[str, float]]] = None,
                     debug: bool = False) -> Decision:
        """
        Prend une décision hybride
        
        Args:
            mcu_data: Données du MCU
            obc_mse: MSE calculé par l'IA OBC
            fault_diagnosis: Diagnostic de défaut
            rl_suggestions: Suggestions du RL (action, confiance)
            debug: Mode debug
            
        Returns:
            Decision: Décision finale
        """
        start_time = time.time()
        
        if debug:
            print(f"\n[DEBUG DECISION] Début analyse...")
            print(f"  MCU état: {mcu_data['state']}")
            print(f"  OBC MSE: {obc_mse:.4f}")
            print(f"  Défaut: {fault_diagnosis.fault_type}")
            print(f"  Confiance défaut: {fault_diagnosis.confidence:.2%}")
        
        # 1. VÉRIFICATION RÈGLES CRITIQUES (priorité absolue)
        critical_decision = self._check_critical_rules(mcu_data, fault_diagnosis)
        if critical_decision:
            decision = critical_decision
            if debug:
                print(f"  → Décision critique (règle prioritaire)")
        
        # 2. DÉCISION BASÉE SUR CONCORDANCE MCU-OBC
        elif mcu_data["state"] == "CRITICAL" and obc_mse > self.thresholds["critical_mse"]:
            decision = self._create_confirmed_decision(
                "CONFIRMED_CRITICAL", "CRITICAL", fault_diagnosis, mcu_data
            )
            if debug:
                print(f"  → CRITICAL confirmé (concordance MCU+OBC)")
        
        elif mcu_data["state"] == "WARNING" and obc_mse > self.thresholds["warning_mse"]:
            decision = self._create_confirmed_decision(
                "CONFIRMED_WARNING", "WARNING", fault_diagnosis, mcu_data
            )
            if debug:
                print(f"  → WARNING confirmé (concordance MCU+OBC)")
        
        # 3. CONFLIT MCU-OBC (nécessite analyse plus poussée)
        elif mcu_data["state"] in ["WARNING", "CRITICAL"] and obc_mse < self.thresholds["warning_mse"]:
            decision = self._handle_conflict(mcu_data, obc_mse, fault_diagnosis, debug)
        
        # 4. ÉTAT NORMAL AVEC SUGGESTIONS RL
        elif mcu_data["state"] == "NORMAL" and rl_suggestions:
            decision = self._handle_normal_with_rl(
                mcu_data, obc_mse, fault_diagnosis, rl_suggestions, debug
            )
        
        # 5. DÉFAUT NORMAL
        else:
            decision = self._create_confirmed_decision(
                "CONFIRMED_NORMAL", "NORMAL", fault_diagnosis, mcu_data
            )
            if debug:
                print(f"  → État NORMAL confirmé")
        
        # Mettre à jour les métadonnées
        decision.metadata["processing_time"] = time.time() - start_time
        decision.metadata["mcu_mse"] = mcu_data.get("mse", 0.0)
        decision.metadata["obc_mse"] = obc_mse
        
        # Enregistrer la décision
        self._record_decision(decision)
        
        if debug:
            print(f"  Décision finale: {decision.decision_type}")
            print(f"  Confiance: {decision.confidence:.2%}")
            print(f"  Temps traitement: {decision.metadata['processing_time']:.3f}s")
        
        return decision
    
    def _check_critical_rules(self, mcu_data: Dict, 
                            fault_diagnosis: FaultDiagnosis) -> Optional[Decision]:
        """Vérifie les règles critiques prioritaires"""
        features = mcu_data["features"]
        
        # R1: Batterie en surchauffe
        if self._check_rule_battery_overheat(features):
            return self._create_confirmed_decision(
                "CRITICAL_RULE_R1", "CRITICAL", fault_diagnosis, mcu_data,
                rule="R1 (Surchauffe batterie >65°C)"
            )
        
        # R2: Décharge profonde
        if self._check_rule_battery_deep_discharge(features):
            return self._create_confirmed_decision(
                "CRITICAL_RULE_R2", "CRITICAL", fault_diagnosis, mcu_data,
                rule="R2 (Décharge profonde <6.2V)"
            )
        
        # R3: Court-circuit
        if self._check_rule_system_critical(features):
            return self._create_confirmed_decision(
                "CRITICAL_RULE_R3", "CRITICAL", fault_diagnosis, mcu_data,
                rule="R3 (Court-circuit possible)"
            )
        
        return None
    
    def _create_confirmed_decision(self, decision_type: str, severity: str,
                                 fault_diagnosis: FaultDiagnosis,
                                 mcu_data: Dict,
                                 **metadata) -> Decision:
        """Crée une décision confirmée"""
        # Calculer la confiance
        if severity == "CRITICAL":
            confidence = max(fault_diagnosis.confidence, 0.8)
        elif severity == "WARNING":
            confidence = max(fault_diagnosis.confidence, 0.6)
        else:
            confidence = max(1.0 - fault_diagnosis.confidence, 0.5)
        
        # Extraire les indicateurs
        indicators = {}
        features = mcu_data["features"]
        feature_names = ["v_batt", "i_batt", "t_batt", "v_solar", 
                        "i_solar", "v_bus", "i_load"]
        
        for i, name in enumerate(feature_names[:len(features)]):
            indicators[name] = features[i]
        
        # Calculer la puissance
        indicators["p_batt"] = features[0] * features[1]
        indicators["p_solar"] = features[3] * features[4]
        
        # Déterminer l'état système
        system_state = self._determine_system_state(severity, fault_diagnosis.severity)
        
        # Métadonnées
        base_metadata = {
            "fault_severity": fault_diagnosis.severity,
            "affected_features": fault_diagnosis.affected_features,
            "rl_influenced": False,
            **metadata
        }
        
        return Decision(
            decision_type=decision_type,
            severity=severity,
            fault_type=fault_diagnosis.fault_type,
            confidence=confidence,
            system_state=system_state,
            indicators=indicators,
            metadata=base_metadata
        )
    
    def _handle_conflict(self, mcu_data: Dict, obc_mse: float,
                        fault_diagnosis: FaultDiagnosis,
                        debug: bool) -> Decision:
        """Gère un conflit entre MCU et OBC"""
        if debug:
            print(f"  ⚠️  Conflit détecté: MCU={mcu_data['state']}, OBC MSE={obc_mse:.4f}")
        
        # Priorité au MCU pour CRITICAL
        if mcu_data["state"] == "CRITICAL":
            if debug:
                print(f"  → Priorité MCU (CRITICAL)")
            return self._create_confirmed_decision(
                "MCU_CRITICAL_CONFLICT", "CRITICAL", fault_diagnosis, mcu_data,
                conflict_type="MCU_CRITICAL_OBC_NORMAL"
            )
        
        # Pour WARNING, analyser plus finement
        conflict_confidence = self._analyze_conflict(mcu_data, obc_mse, fault_diagnosis)
        
        if conflict_confidence > 0.7:
            if debug:
                print(f"  → Conflit résolu en faveur MCU")
            return self._create_confirmed_decision(
                "CONFIRMED_WARNING_CONFLICT", "WARNING", fault_diagnosis, mcu_data,
                conflict_confidence=conflict_confidence
            )
        else:
            if debug:
                print(f"  → Conflit résolu en faveur OBC (faux positif probable)")
            return self._create_confirmed_decision(
                "REJECTED_MCU_WARNING", "NORMAL", fault_diagnosis, mcu_data,
                conflict_confidence=conflict_confidence
            )
    
    def _analyze_conflict(self, mcu_data: Dict, obc_mse: float,
                         fault_diagnosis: FaultDiagnosis) -> float:
        """Analyse un conflit MCU-OBC"""
        confidence = 0.0
        
        # Facteur 1: Type de défaut
        severe_faults = ["BATTERY_OVERHEAT", "BATTERY_DEEP_DISCHARGE", 
                        "BUS_OVERLOAD", "SOLAR_SHORT_CIRCUIT"]
        if fault_diagnosis.fault_type in severe_faults:
            confidence += 0.4
        
        # Facteur 2: MSE MCU
        mcu_mse = mcu_data.get("mse", 0.0)
        if mcu_mse > 1.0:
            confidence += 0.3
        
        # Facteur 3: Valeurs aberrantes
        features = mcu_data["features"]
        if features[2] > 50 or features[0] < 6.5:  # Temp ou tension critique
            confidence += 0.3
        
        return min(confidence, 1.0)
    
    def _handle_normal_with_rl(self, mcu_data: Dict, obc_mse: float,
                              fault_diagnosis: FaultDiagnosis,
                              rl_suggestions: List[Tuple[str, float]],
                              debug: bool) -> Decision:
        """Gère un état normal avec suggestions RL"""
        best_action, best_confidence = rl_suggestions[0]
        
        # Seuil de confiance RL
        if best_confidence >= self.thresholds["rl_confidence_threshold"]:
            # Créer une décision influencée par le RL
            decision = self._create_confirmed_decision(
                f"NORMAL_RL_{best_action}", "NORMAL", fault_diagnosis, mcu_data
            )
            
            # Mettre à jour les métadonnées
            decision.metadata["rl_influenced"] = True
            decision.metadata["rl_action"] = best_action
            decision.metadata["rl_confidence"] = best_confidence
            decision.metadata["rl_suggestions"] = rl_suggestions
            
            self.statistics["rl_influenced"] += 1
            
            if debug:
                print(f"  → Décision influencée par RL: {best_action}")
            
            return decision
        
        # RL pas assez confiant
        return self._create_confirmed_decision(
            "CONFIRMED_NORMAL", "NORMAL", fault_diagnosis, mcu_data
        )
    
    def _determine_system_state(self, decision_severity: str, 
                               fault_severity: str) -> str:
        """Détermine l'état système"""
        if decision_severity == "CRITICAL" or fault_severity == "CRITICAL":
            return "CRITICAL"
        elif decision_severity == "WARNING" or fault_severity == "SEVERE":
            return "DEGRADED"
        elif decision_severity == "NORMAL" and fault_severity in ["MODERATE", "MINOR"]:
            return "SAFE"
        else:
            return "NOMINAL"
    
    def _record_decision(self, decision: Decision):
        """Enregistre une décision dans l'historique"""
        self.decision_history.append(decision)
        self.statistics["total_decisions"] += 1
        
        # Mettre à jour les statistiques
        if "CRITICAL" in decision.decision_type:
            self.statistics["critical"] += 1
        elif "WARNING" in decision.decision_type:
            self.statistics["warning"] += 1
        else:
            self.statistics["normal"] += 1
        
        if "CONFIRMED" in decision.decision_type:
            self.statistics["confirmed"] += 1
        elif "REJECTED" in decision.decision_type:
            self.statistics["rejected"] += 1
        
        # Garder seulement les 100 dernières décisions
        if len(self.decision_history) > 100:
            self.decision_history.pop(0)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Retourne les statistiques de décision"""
        total = self.statistics["total_decisions"]
        
        stats = self.statistics.copy()
        if total > 0:
            stats["confirmation_rate"] = stats["confirmed"] / total
            stats["false_positive_rate"] = stats["rejected"] / total
            stats["critical_rate"] = stats["critical"] / total
            stats["rl_influence_rate"] = stats["rl_influenced"] / total
        else:
            stats["confirmation_rate"] = 0.0
            stats["false_positive_rate"] = 0.0
            stats["critical_rate"] = 0.0
            stats["rl_influence_rate"] = 0.0
        
        return stats
    
    def get_recent_decisions(self, count: int = 10) -> List[Decision]:
        """Retourne les décisions récentes"""
        return self.decision_history[-count:] if self.decision_history else []

# Instance globale
decision_engine = DecisionEngine()