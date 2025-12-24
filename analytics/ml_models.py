"""
Machine Learning Models for Agricultural Analysis

Implements:
- Irrigation classification using ML
- Micro climate prediction
- Weed detection using Gaussian Mixture Model
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, mean_squared_error
import pickle
import os
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class IrrigationClassifier:
    """
    ML-based irrigation classification using satellite and IoT data.
    
    Classifies fields as:
    - irrigated
    - partially_irrigated
    - rainfed
    """
    
    def __init__(self, model_path: str = None):
        self.model = None
        self.scaler = StandardScaler()
        self.model_path = model_path or 'models/irrigation_classifier.pkl'
        self.feature_names = [
            'ndvi_mean', 'ndvi_std', 'ndvi_max', 'ndvi_min',
            'ndwi_mean', 'ndwi_std',
            'soil_moisture_mean', 'soil_moisture_std',
            'temperature_mean', 'precipitation_total',
            'elevation', 'slope'
        ]
    
    def prepare_features(self, 
                         ndvi_series: List[float],
                         ndwi_series: List[float] = None,
                         soil_moisture_series: List[float] = None,
                         temperature_series: List[float] = None,
                         precipitation_total: float = None,
                         elevation: float = None,
                         slope: float = None) -> np.ndarray:
        """
        Prepare feature vector from input data.
        """
        features = []
        
        # NDVI features
        ndvi = np.array(ndvi_series) if ndvi_series else np.array([0.5])
        features.extend([
            np.mean(ndvi),
            np.std(ndvi),
            np.max(ndvi),
            np.min(ndvi)
        ])
        
        # NDWI features
        ndwi = np.array(ndwi_series) if ndwi_series else np.array([0])
        features.extend([
            np.mean(ndwi),
            np.std(ndwi)
        ])
        
        # Soil moisture features
        sm = np.array(soil_moisture_series) if soil_moisture_series else np.array([30])
        features.extend([
            np.mean(sm),
            np.std(sm)
        ])
        
        # Weather features
        temp = np.array(temperature_series) if temperature_series else np.array([25])
        features.append(np.mean(temp))
        features.append(precipitation_total if precipitation_total is not None else 500)
        
        # Terrain features
        features.append(elevation if elevation is not None else 100)
        features.append(slope if slope is not None else 2)
        
        return np.array(features).reshape(1, -1)
    
    def train(self, X: np.ndarray, y: np.ndarray, save_model: bool = True) -> Dict:
        """
        Train the irrigation classifier.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Labels (0=rainfed, 1=partially_irrigated, 2=irrigated)
            save_model: Whether to save trained model
            
        Returns:
            Training metrics
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train Random Forest
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        
        # Get feature importance
        importance = dict(zip(self.feature_names, self.model.feature_importances_))
        
        if save_model:
            self.save_model()
        
        return {
            'accuracy': float(self.model.score(X_test_scaled, y_test)),
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
            'feature_importance': importance,
            'n_samples': len(y)
        }
    
    def predict(self, **kwargs) -> Dict:
        """
        Predict irrigation status for a field.
        
        Returns:
            Prediction with probability
        """
        if self.model is None:
            # Use rule-based fallback if no model
            return self._rule_based_prediction(**kwargs)
        
        features = self.prepare_features(**kwargs)
        features_scaled = self.scaler.transform(features)
        
        prediction = self.model.predict(features_scaled)[0]
        probabilities = self.model.predict_proba(features_scaled)[0]
        
        labels = ['rainfed', 'partially_irrigated', 'irrigated']
        
        return {
            'classification': labels[prediction],
            'confidence': float(max(probabilities)),
            'probabilities': {
                label: float(prob) 
                for label, prob in zip(labels, probabilities)
            }
        }
    
    def _rule_based_prediction(self, 
                               ndvi_series: List[float] = None,
                               ndwi_series: List[float] = None,
                               soil_moisture_series: List[float] = None,
                               **kwargs) -> Dict:
        """
        Rule-based fallback prediction when no model is trained.
        """
        # Calculate indicators
        ndvi_std = np.std(ndvi_series) if ndvi_series else 0.1
        ndwi_mean = np.mean(ndwi_series) if ndwi_series else 0
        sm_mean = np.mean(soil_moisture_series) if soil_moisture_series else 30
        sm_std = np.std(soil_moisture_series) if soil_moisture_series else 10
        
        # Irrigated fields have:
        # - Low NDVI variance (consistent greenness)
        # - Higher NDWI (more water content)
        # - Higher soil moisture with low variance
        
        score = 0
        
        if ndvi_std < 0.15:
            score += 1
        if ndwi_mean > 0.1:
            score += 1
        if sm_mean > 40:
            score += 1
        if sm_std < 10:
            score += 1
        
        if score >= 3:
            classification = 'irrigated'
            confidence = 0.7 + (score - 3) * 0.1
        elif score >= 2:
            classification = 'partially_irrigated'
            confidence = 0.6
        else:
            classification = 'rainfed'
            confidence = 0.6 + (2 - score) * 0.1
        
        return {
            'classification': classification,
            'confidence': float(min(confidence, 1.0)),
            'method': 'rule_based',
            'probabilities': {
                'rainfed': 0.33,
                'partially_irrigated': 0.33,
                'irrigated': 0.34
            }
        }
    
    def save_model(self):
        """Save model to disk."""
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        with open(self.model_path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'scaler': self.scaler
            }, f)
    
    def load_model(self):
        """Load model from disk."""
        if os.path.exists(self.model_path):
            with open(self.model_path, 'rb') as f:
                data = pickle.load(f)
                self.model = data['model']
                self.scaler = data['scaler']
            return True
        return False


class MicroClimatePredictor:
    """
    Predict micro climate parameters (temperature, humidity) at field level
    using global weather data and local IoT sensor observations.
    """
    
    def __init__(self, model_path: str = None):
        self.temp_model = None
        self.humidity_model = None
        self.scaler = StandardScaler()
        self.model_path = model_path or 'models/microclimate_predictor.pkl'
    
    def prepare_features(self,
                         global_temp: float,
                         global_humidity: float,
                         global_pressure: float,
                         wind_speed: float,
                         cloud_cover: float,
                         elevation: float,
                         slope: float,
                         aspect: float,
                         distance_to_water: float,
                         ndvi: float,
                         hour_of_day: int,
                         day_of_year: int) -> np.ndarray:
        """
        Prepare feature vector for prediction.
        """
        # Convert hour to cyclic features
        hour_sin = np.sin(2 * np.pi * hour_of_day / 24)
        hour_cos = np.cos(2 * np.pi * hour_of_day / 24)
        
        # Convert day to cyclic features
        day_sin = np.sin(2 * np.pi * day_of_year / 365)
        day_cos = np.cos(2 * np.pi * day_of_year / 365)
        
        features = [
            global_temp, global_humidity, global_pressure,
            wind_speed, cloud_cover,
            elevation, slope, aspect, distance_to_water,
            ndvi,
            hour_sin, hour_cos, day_sin, day_cos
        ]
        
        return np.array(features).reshape(1, -1)
    
    def train(self, X: np.ndarray, y_temp: np.ndarray, y_humidity: np.ndarray) -> Dict:
        """
        Train micro climate prediction models.
        
        Args:
            X: Feature matrix
            y_temp: Local temperature labels
            y_humidity: Local humidity labels
            
        Returns:
            Training metrics
        """
        X_train, X_test, y_temp_train, y_temp_test = train_test_split(
            X, y_temp, test_size=0.2, random_state=42
        )
        _, _, y_hum_train, y_hum_test = train_test_split(
            X, y_humidity, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train temperature model
        self.temp_model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=5,
            random_state=42
        )
        self.temp_model.fit(X_train_scaled, y_temp_train)
        
        # Train humidity model
        self.humidity_model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=5,
            random_state=42
        )
        self.humidity_model.fit(X_train_scaled, y_hum_train)
        
        # Evaluate
        temp_pred = self.temp_model.predict(X_test_scaled)
        hum_pred = self.humidity_model.predict(X_test_scaled)
        
        return {
            'temperature_rmse': float(np.sqrt(mean_squared_error(y_temp_test, temp_pred))),
            'humidity_rmse': float(np.sqrt(mean_squared_error(y_hum_test, hum_pred))),
            'temperature_r2': float(self.temp_model.score(X_test_scaled, y_temp_test)),
            'humidity_r2': float(self.humidity_model.score(X_test_scaled, y_hum_test)),
            'n_samples': len(y_temp)
        }
    
    def predict(self, **kwargs) -> Dict:
        """
        Predict local micro climate.
        """
        features = self.prepare_features(**kwargs)
        
        if self.temp_model is None or self.humidity_model is None:
            # Use estimation fallback
            return self._estimate_microclimate(**kwargs)
        
        features_scaled = self.scaler.transform(features)
        
        temp_pred = self.temp_model.predict(features_scaled)[0]
        hum_pred = self.humidity_model.predict(features_scaled)[0]
        
        return {
            'local_temperature': float(temp_pred),
            'local_humidity': float(hum_pred),
            'temperature_adjustment': float(temp_pred - kwargs.get('global_temp', temp_pred)),
            'humidity_adjustment': float(hum_pred - kwargs.get('global_humidity', hum_pred)),
            'method': 'ml_prediction'
        }
    
    def _estimate_microclimate(self, 
                               global_temp: float,
                               global_humidity: float,
                               elevation: float = 100,
                               ndvi: float = 0.5,
                               **kwargs) -> Dict:
        """
        Simple estimation when no model is trained.
        
        Uses:
        - Lapse rate for elevation (-0.65°C per 100m)
        - Vegetation cooling effect
        """
        # Elevation adjustment (lapse rate)
        elevation_adjustment = -0.0065 * elevation
        
        # Vegetation cooling (evapotranspiration)
        veg_cooling = -1.5 * ndvi  # Up to -1.5°C for dense vegetation
        
        local_temp = global_temp + elevation_adjustment + veg_cooling
        
        # Humidity increases with vegetation
        humidity_increase = 5 * ndvi
        local_humidity = min(100, global_humidity + humidity_increase)
        
        return {
            'local_temperature': float(local_temp),
            'local_humidity': float(local_humidity),
            'temperature_adjustment': float(elevation_adjustment + veg_cooling),
            'humidity_adjustment': float(humidity_increase),
            'method': 'estimation',
            'factors': {
                'elevation_effect': float(elevation_adjustment),
                'vegetation_effect': float(veg_cooling)
            }
        }


class WeedDetector:
    """
    Detect weeds in crop fields using Gaussian Mixture Model clustering
    on spectral data from drone/satellite imagery.
    """
    
    def __init__(self, n_components: int = 3):
        """
        Initialize weed detector.
        
        Args:
            n_components: Number of clusters (typically 3: crop, weed, soil)
        """
        self.n_components = n_components
        self.gmm = None
        self.scaler = StandardScaler()
        self.cluster_labels = {}  # Map cluster to type
    
    def prepare_features(self, 
                         red: np.ndarray,
                         green: np.ndarray,
                         blue: np.ndarray,
                         nir: np.ndarray = None,
                         red_edge: np.ndarray = None) -> np.ndarray:
        """
        Prepare feature matrix from spectral bands.
        
        Args:
            red, green, blue: RGB bands
            nir: Near-infrared band (optional)
            red_edge: Red edge band (optional)
            
        Returns:
            Feature matrix (n_pixels, n_features)
        """
        # Flatten arrays
        r = red.flatten()
        g = green.flatten()
        b = blue.flatten()
        
        # Calculate vegetation indices
        # Excess Green Index
        egi = 2 * g - r - b
        
        # Excess Red Index
        eri = 1.4 * r - g
        
        # Excess Green minus Excess Red
        exgr = egi - eri
        
        features = [r, g, b, egi, exgr]
        
        if nir is not None:
            n = nir.flatten()
            # NDVI
            ndvi = (n - r) / (n + r + 1e-10)
            features.extend([n, ndvi])
        
        if red_edge is not None:
            re = red_edge.flatten()
            # NDRE (Red Edge NDVI)
            if nir is not None:
                ndre = (nir.flatten() - re) / (nir.flatten() + re + 1e-10)
                features.append(ndre)
        
        return np.column_stack(features)
    
    def fit(self, features: np.ndarray) -> Dict:
        """
        Fit GMM to spectral features.
        
        Args:
            features: Feature matrix from prepare_features()
            
        Returns:
            Clustering results
        """
        # Remove invalid pixels
        valid_mask = ~np.any(np.isnan(features), axis=1)
        valid_features = features[valid_mask]
        
        if len(valid_features) < self.n_components * 10:
            return {'error': 'Insufficient valid pixels'}
        
        # Scale features
        features_scaled = self.scaler.fit_transform(valid_features)
        
        # Fit GMM
        self.gmm = GaussianMixture(
            n_components=self.n_components,
            covariance_type='full',
            random_state=42,
            n_init=5
        )
        self.gmm.fit(features_scaled)
        
        # Get cluster assignments
        labels = self.gmm.predict(features_scaled)
        
        # Identify cluster types based on spectral characteristics
        self._identify_clusters(valid_features, labels)
        
        return {
            'n_clusters': self.n_components,
            'cluster_sizes': {
                int(i): int(np.sum(labels == i)) 
                for i in range(self.n_components)
            },
            'cluster_types': self.cluster_labels,
            'bic': float(self.gmm.bic(features_scaled)),
            'aic': float(self.gmm.aic(features_scaled))
        }
    
    def _identify_clusters(self, features: np.ndarray, labels: np.ndarray):
        """
        Identify cluster types based on spectral characteristics.
        
        Uses:
        - High EGI + High NDVI = Crop
        - High EGI + Lower NDVI = Weed
        - Low EGI = Soil/Background
        """
        cluster_means = {}
        
        for i in range(self.n_components):
            mask = labels == i
            cluster_features = features[mask]
            
            # Assuming features are [r, g, b, egi, exgr, ...]
            mean_egi = np.mean(cluster_features[:, 3])
            mean_exgr = np.mean(cluster_features[:, 4])
            
            cluster_means[i] = {
                'egi': mean_egi,
                'exgr': mean_exgr
            }
        
        # Sort by EGI (greenness)
        sorted_clusters = sorted(cluster_means.items(), key=lambda x: x[1]['egi'])
        
        # Assign labels
        self.cluster_labels = {
            sorted_clusters[0][0]: 'soil',
            sorted_clusters[-1][0]: 'crop'
        }
        
        # Middle clusters are potential weeds
        for i, (cluster_id, _) in enumerate(sorted_clusters[1:-1]):
            if sorted_clusters[cluster_id][1]['exgr'] < 0:
                self.cluster_labels[cluster_id] = 'weed'
            else:
                self.cluster_labels[cluster_id] = 'mixed_vegetation'
    
    def predict(self, features: np.ndarray, original_shape: Tuple = None) -> Dict:
        """
        Predict weed locations in new imagery.
        
        Args:
            features: Feature matrix
            original_shape: Original image shape for reshaping results
            
        Returns:
            Weed detection results with map
        """
        if self.gmm is None:
            return {'error': 'Model not fitted. Call fit() first.'}
        
        # Handle invalid pixels
        valid_mask = ~np.any(np.isnan(features), axis=1)
        
        # Initialize full predictions
        full_predictions = np.full(len(features), -1)
        
        if np.any(valid_mask):
            features_scaled = self.scaler.transform(features[valid_mask])
            predictions = self.gmm.predict(features_scaled)
            full_predictions[valid_mask] = predictions
        
        # Create semantic map
        semantic_map = np.array([
            self.cluster_labels.get(p, 'unknown') for p in full_predictions
        ])
        
        # Calculate statistics
        total_pixels = len(features)
        weed_pixels = np.sum(semantic_map == 'weed')
        crop_pixels = np.sum(semantic_map == 'crop')
        soil_pixels = np.sum(semantic_map == 'soil')
        
        results = {
            'weed_percentage': float(weed_pixels / total_pixels * 100),
            'crop_percentage': float(crop_pixels / total_pixels * 100),
            'soil_percentage': float(soil_pixels / total_pixels * 100),
            'weed_pixels': int(weed_pixels),
            'total_pixels': int(total_pixels),
            'cluster_labels': full_predictions.tolist(),
            'semantic_labels': semantic_map.tolist()
        }
        
        if original_shape:
            results['weed_map_shape'] = original_shape
            # Could reshape here if needed
        
        return results
    
    def detect_from_image(self, 
                          red: np.ndarray,
                          green: np.ndarray,
                          blue: np.ndarray,
                          nir: np.ndarray = None) -> Dict:
        """
        Full pipeline: prepare features, fit, and detect weeds.
        
        Args:
            red, green, blue: RGB bands as 2D arrays
            nir: Optional NIR band
            
        Returns:
            Detection results with weed map
        """
        original_shape = red.shape
        
        # Prepare features
        features = self.prepare_features(red, green, blue, nir)
        
        # Fit model
        fit_result = self.fit(features)
        
        if 'error' in fit_result:
            return fit_result
        
        # Predict
        predictions = self.predict(features, original_shape)
        predictions['model_info'] = fit_result
        
        return predictions


def classify_field_irrigation(field_id: int) -> Dict:
    """
    Classify irrigation status for a specific field.
    """
    from analytics.models import AnalyticsResult
    from iot.models import SensorReading
    from core.models import FieldBoundary
    
    try:
        field = FieldBoundary.objects.get(id=field_id)
    except FieldBoundary.DoesNotExist:
        return {'error': f'Field {field_id} not found'}
    
    # Get NDVI data
    ndvi_results = AnalyticsResult.objects.filter(
        field=field
    ).values_list('avg_ndvi', flat=True)
    
    # Get NDWI data
    ndwi_results = AnalyticsResult.objects.filter(
        field=field
    ).values_list('avg_ndwi', flat=True)
    
    # Get soil moisture from IoT
    soil_moisture = SensorReading.objects.filter(
        device__field=field,
        sensor_type='moisture'
    ).values_list('value', flat=True)
    
    classifier = IrrigationClassifier()
    classifier.load_model()  # Try to load pre-trained model
    
    result = classifier.predict(
        ndvi_series=list(filter(None, ndvi_results)),
        ndwi_series=list(filter(None, ndwi_results)),
        soil_moisture_series=list(soil_moisture)
    )
    
    result['field_id'] = field_id
    result['field_name'] = field.name
    
    return result
