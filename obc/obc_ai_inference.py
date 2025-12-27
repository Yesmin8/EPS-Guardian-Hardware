"""
Module d'inférence IA - LSTM Autoencoder
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.losses import MeanSquaredError  # <=== AJOUT IMPORT
import pickle
import joblib
from typing import List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class OBC_AI:
    """
    Interface pour le modèle LSTM Autoencoder
    """
    
    def __init__(self, model_path: str, scaler_path: str):
        """
        Initialise le modèle IA
        
        Args:
            model_path: Chemin vers le modèle .h5
            scaler_path: Chemin vers le scaler .pkl
        """
        self.model_path = model_path
        self.scaler_path = scaler_path
        
        # Buffer pour accumulation
        self.buffer = []
        self.sequence_length = 30
        self.feature_count = 7
        
        # Charger le modèle
        self.model = self._load_model()
        
        # Charger le scaler
        self.scaler = self._load_scaler()
        
        # Statut
        self.ready = False
        
        # Statistiques
        self.inference_count = 0
        self.avg_inference_time = 0
        
        print(f"   IA initialisée: {model_path}")
        print(f"   Séquence: {self.sequence_length} steps")
        print(f"   Features: {self.feature_count}")
    
    def _load_model(self) -> keras.Model:
        """Charge le modèle LSTM Autoencoder"""
        try:
            # PROBLÈME 3 FIX: Ajouter custom_objects pour 'mse'
            custom_objects = {
                'mse': MeanSquaredError(),
            }
            
            model = keras.models.load_model(
                self.model_path,
                custom_objects=custom_objects,  # <=== AJOUT
                compile=False  # On n'a pas besoin de compilation pour l'inférence
            )
            print(f"   Modèle chargé: {self.model_path}")
            return model
        except Exception as e:
            print(f" Erreur chargement modèle: {e}")
            # Créer un modèle factice pour tests
            print("  Modèle factice créé (mode test)")
            return self._create_dummy_model()
    
    def _create_dummy_model(self) -> keras.Model:
        """Crée un modèle factice pour tests"""
        inputs = keras.Input(shape=(self.sequence_length, self.feature_count))
        encoded = keras.layers.LSTM(16, return_sequences=True)(inputs)
        decoded = keras.layers.LSTM(self.feature_count, return_sequences=True)(encoded)
        model = keras.Model(inputs, decoded)
        # Pas de compile pour le mode test
        print("  Modèle factice créé (mode test)")
        return model
    
    def _load_scaler(self):
        """Charge le scaler pour normalisation"""
        try:
            # Essayer d'abord avec joblib (plus robuste)
            try:
                scaler = joblib.load(self.scaler_path)
                print(f"   Scaler chargé (joblib): {self.scaler_path}")
                return scaler
            except:
                # Essayer avec pickle
                with open(self.scaler_path, 'rb') as f:
                    scaler = pickle.load(f)
                print(f"   Scaler chargé (pickle): {self.scaler_path}")
                return scaler
        except Exception as e:
            print(f" Erreur chargement scaler: {e}")
            # Créer un scaler factice
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            scaler.mean_ = np.zeros(self.feature_count)
            scaler.scale_ = np.ones(self.feature_count)
            scaler.var_ = np.ones(self.feature_count)
            scaler.n_samples_seen_ = 100
            print("  Scaler factice créé (mode test)")
            return scaler
    
    def add_sample(self, features: List[float]) -> None:
        """
        Ajoute un échantillon au buffer
        
        Args:
            features: Liste de 7 valeurs
        """
        if len(features) != self.feature_count:
            print(f"  Features attendues: {self.feature_count}, reçues: {len(features)}")
            return
        
        self.buffer.append(features)
        
        # Garder seulement la dernière séquence
        if len(self.buffer) > self.sequence_length:
            self.buffer.pop(0)
    
    def is_ready(self) -> bool:
        """Vérifie si le buffer est plein"""
        return len(self.buffer) >= self.sequence_length
    
    def get_current_buffer_size(self) -> int:
        """Retourne la taille actuelle du buffer"""
        return len(self.buffer)
    
    def predict(self) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Exécute l'inférence sur le buffer courant
        
        Returns:
            mse: Erreur quadratique moyenne
            original: Données originales
            reconstructed: Données reconstruites
        """
        if not self.is_ready():
            raise ValueError(f"Buffer incomplet: {len(self.buffer)}/{self.sequence_length}")
        
        import time
        start_time = time.time()
        
        # Convertir en numpy array
        sequence = np.array(self.buffer[-self.sequence_length:])
        original = sequence.copy()
        
        # Normaliser
        try:
            sequence_norm = self.scaler.transform(sequence)
        except AttributeError:
            # Scaler factice - pas de transformation
            sequence_norm = sequence
        
        sequence_norm = sequence_norm.reshape(1, self.sequence_length, self.feature_count)
        
        # Prédiction
        reconstructed_norm = self.model.predict(sequence_norm, verbose=0)
        
        # Dénormaliser
        try:
            reconstructed = self.scaler.inverse_transform(
                reconstructed_norm.reshape(-1, self.feature_count)
            ).reshape(self.sequence_length, self.feature_count)
        except AttributeError:
            # Scaler factice - pas de dénormalisation
            reconstructed = reconstructed_norm.reshape(self.sequence_length, self.feature_count)
        
        # Calcul MSE
        mse = np.mean((original - reconstructed) ** 2)
        
        # Mettre à jour les stats
        inference_time = time.time() - start_time
        self.inference_count += 1
        self.avg_inference_time = (self.avg_inference_time * 0.9 + 
                                  inference_time * 0.1)
        
        return mse, original, reconstructed
    
    def analyze_anomaly(self, mse: float) -> dict:
        """
        Analyse une anomalie détectée
        
        Args:
            mse: Valeur MSE calculée
            
        Returns:
            dict: Analyse détaillée
        """
        analysis = {
            "mse": mse,
            "anomaly_level": "NORMAL",
            "confidence": 0.0,
            "features_affected": [],
            "recommendation": "Continuer monitoring"
        }
        
        if mse > 1.5:
            analysis["anomaly_level"] = "CRITICAL"
            analysis["confidence"] = min(mse / 3.0, 0.95)
            analysis["recommendation"] = "Action immédiate requise"
        elif mse > 1.0:
            analysis["anomaly_level"] = "WARNING"
            analysis["confidence"] = min(mse / 2.0, 0.85)
            analysis["recommendation"] = "Surveillance accrue"
        else:
            analysis["anomaly_level"] = "NORMAL"
            analysis["confidence"] = max(1.0 - mse, 0.5)
        
        return analysis
    
    def reset_buffer(self) -> None:
        """Réinitialise le buffer"""
        self.buffer = []
        print(" Buffer IA réinitialisé")
    
    def get_stats(self) -> dict:
        """Retourne les statistiques d'inférence"""
        return {
            "inference_count": self.inference_count,
            "avg_inference_time": self.avg_inference_time,
            "buffer_size": len(self.buffer),
            "buffer_ready": self.is_ready(),
            "model_loaded": self.model is not None,
            "scaler_loaded": self.scaler is not None
        }