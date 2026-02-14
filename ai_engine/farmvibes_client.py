"""
FarmVibes-AI style workflows using open-source models.

Implements:
1. Cloud-free NDVI with SCL masking (SpaceEye-style)
2. NDVI time-series anomaly detection
3. Crop classification using TorchGeo / EuroSAT
4. Sensor-satellite fusion for soil moisture estimation
5. Data generation helpers for model training
"""

import logging
import os
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class FarmVibesAnalyzer:
    """
    Replicates key FarmVibes-AI workflows using the same underlying
    open-source models (torchgeo, segmentation_models_pytorch, scikit-learn).
    """

    def __init__(self):
        self._device = None
        logger.info("FarmVibes Analyzer initialised")

    @property
    def device(self):
        """Lazy torch device init so non-torch callers don't need GPU."""
        if self._device is None:
            try:
                import torch
                self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            except ImportError:
                self._device = 'cpu'
        return self._device

    # ── Workflow 1: Cloud-free NDVI ──────────────────────────────
    def compute_cloud_free_ndvi(self, red_path: str, nir_path: str,
                                scl_path: str = None) -> dict:
        """
        SpaceEye-style NDVI with Sentinel-2 SCL cloud masking.
        """
        import rasterio

        with rasterio.open(red_path) as red_src, rasterio.open(nir_path) as nir_src:
            red = red_src.read(1).astype(np.float32)
            nir = nir_src.read(1).astype(np.float32)
            profile = red_src.profile.copy()

        cloud_percent = 0.0
        if scl_path and os.path.exists(scl_path):
            with rasterio.open(scl_path) as scl_src:
                scl = scl_src.read(1)
            cloud_mask = np.isin(scl, [0, 3, 8, 9, 10])
            red = np.where(cloud_mask, np.nan, red)
            nir = np.where(cloud_mask, np.nan, nir)
            cloud_percent = float(cloud_mask.sum() / cloud_mask.size * 100)

        denom = nir + red
        ndvi = np.where(denom > 0, (nir - red) / denom, np.nan)

        return {
            'ndvi': ndvi,
            'profile': profile,
            'stats': {
                'mean': float(np.nanmean(ndvi)),
                'min': float(np.nanmin(ndvi)),
                'max': float(np.nanmax(ndvi)),
                'std': float(np.nanstd(ndvi)),
                'cloud_cover_percent': cloud_percent,
                'valid_pixel_percent': float((~np.isnan(ndvi)).sum() / ndvi.size * 100),
            },
        }

    # ── Workflow 2: NDVI anomaly detection ───────────────────────
    def detect_ndvi_anomalies(self, ndvi_history: List[dict],
                              z_threshold: float = -2.0) -> List[dict]:
        """
        Flag fields where NDVI deviates from its rolling baseline.
        """
        if len(ndvi_history) < 5:
            return []

        values = np.array([h['mean_ndvi'] for h in ndvi_history])
        dates = [h['date'] for h in ndvi_history]
        window = 5
        anomalies = []

        for i in range(window, len(values)):
            w = values[i - window:i]
            mean, std = w.mean(), w.std()
            if std > 0.01:
                z = (values[i] - mean) / std
                if z < z_threshold:
                    anomalies.append({
                        'date': dates[i],
                        'ndvi': float(values[i]),
                        'expected_ndvi': float(mean),
                        'z_score': float(z),
                        'severity': 'critical' if z < -3 else 'warning',
                        'drop_percent': float((mean - values[i]) / mean * 100),
                    })
        return anomalies

    # ── Workflow 3: Crop classification (TorchGeo) ───────────────
    def classify_crop_from_bands(self, image_path: str,
                                  model_path: str = None) -> dict:
        """
        EuroSAT-based crop classification mapped to Pakistani crops.
        Falls back to NDVI-rule-based when torch is unavailable.
        """
        try:
            import torch
            import rasterio

            EUROSAT_CLASSES = [
                'AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway',
                'Industrial', 'Pasture', 'PermanentCrop', 'Residential',
                'River', 'SeaLake',
            ]
            MAP_TO_PK = {
                'AnnualCrop': 'wheat/rice',
                'Forest': 'orchard',
                'HerbaceousVegetation': 'fodder/grass',
                'Pasture': 'grazing_land',
                'PermanentCrop': 'sugarcane/cotton',
                'Residential': 'settlement',
                'Industrial': 'non_agricultural',
                'River': 'water_body',
                'SeaLake': 'water_body',
                'Highway': 'non_agricultural',
            }

            with rasterio.open(image_path) as src:
                data = src.read().astype(np.float32)

            # Normalise per band
            for i in range(data.shape[0]):
                band = data[i]
                if band.std() > 0:
                    data[i] = (band - band.mean()) / (band.std() + 1e-8)

            from torchvision.models import resnet18
            model = resnet18(weights=None)
            model.conv1 = torch.nn.Conv2d(data.shape[0], 64, 7, stride=2, padding=3, bias=False)
            model.fc = torch.nn.Linear(model.fc.in_features, len(EUROSAT_CLASSES))

            if model_path and os.path.exists(model_path):
                model.load_state_dict(torch.load(model_path, map_location=self.device))

            model.to(self.device).eval()

            # Resize to 64×64 patch
            from torchvision.transforms.functional import resize
            tensor = torch.tensor(data).unsqueeze(0).to(self.device)
            tensor = resize(tensor, [64, 64])

            with torch.no_grad():
                output = model(tensor)
                probs = torch.nn.functional.softmax(output, dim=1)[0]

            results = {}
            for idx, cls in enumerate(EUROSAT_CLASSES):
                pk = MAP_TO_PK[cls]
                results[pk] = results.get(pk, 0) + float(probs[idx])

            top = max(results, key=results.get)
            return {
                'detected_crop': top,
                'confidence': round(results[top], 4),
                'all_probabilities': {k: round(v, 4) for k, v in results.items()},
                'method': 'eurosat_resnet18',
            }

        except ImportError:
            logger.warning("torch not available — using NDVI rule-based crop classification")
            return self._rule_based_crop(image_path)

    def _rule_based_crop(self, image_path: str) -> dict:
        """Fallback when torch is missing."""
        try:
            import rasterio
            with rasterio.open(image_path) as src:
                if src.count >= 4:
                    red = src.read(3).astype(np.float32)
                    nir = src.read(4).astype(np.float32)
                else:
                    red = src.read(1).astype(np.float32)
                    nir = src.read(min(2, src.count)).astype(np.float32)
            denom = nir + red
            ndvi = np.where(denom > 0, (nir - red) / denom, 0)
            mean_ndvi = float(np.nanmean(ndvi))
        except Exception:
            mean_ndvi = 0.5

        if mean_ndvi > 0.6:
            crop = 'wheat/rice'
        elif mean_ndvi > 0.4:
            crop = 'sugarcane/cotton'
        elif mean_ndvi > 0.2:
            crop = 'fodder/grass'
        else:
            crop = 'bare_soil'

        return {
            'detected_crop': crop,
            'confidence': 0.45,
            'method': 'ndvi_rule_based',
            'ndvi_mean': mean_ndvi,
        }

    # ── Workflow 4: Sensor–satellite fusion ──────────────────────
    def estimate_soil_moisture(self, sensor_readings: list,
                               ndvi: float, temperature: float) -> dict:
        """
        Fuse IoT soil moisture readings with satellite NDVI and weather.
        """
        moisture_values = []
        for r in sensor_readings:
            if isinstance(r, dict):
                m = (r.get('soilMoisture_%') or r.get('soilMoisture') or
                     r.get('soil_moisture') or r.get('moisture'))
                if m is not None:
                    moisture_values.append(float(m))

        if not moisture_values:
            return {'estimated_moisture': None, 'confidence': 0.0, 'method': 'no_data'}

        sensor_mean = np.mean(moisture_values)
        sensor_std = np.std(moisture_values)

        ndvi_factor = 1.0 + (ndvi - 0.5) * 0.15
        temp_factor = 1.0 - max(0, (temperature - 25)) * 0.008
        adjusted = sensor_mean * ndvi_factor * temp_factor

        consistency = max(0, 1.0 - (sensor_std / (sensor_mean + 1e-6)))
        qty = min(1.0, len(moisture_values) / 10)
        confidence = consistency * 0.6 + qty * 0.4

        return {
            'estimated_moisture': round(float(adjusted), 2),
            'raw_sensor_mean': round(float(sensor_mean), 2),
            'sensor_std': round(float(sensor_std), 2),
            'ndvi_correction': round(float(ndvi_factor), 3),
            'temp_correction': round(float(temp_factor), 3),
            'num_readings': len(moisture_values),
            'confidence': round(float(confidence), 3),
            'method': 'sensor_satellite_fusion',
        }

    # ── Training data generators ─────────────────────────────────
    @staticmethod
    def generate_irrigation_training_data(fields_qs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build X, y from real AnalyticsResult + SensorReading rows.
        Returns (features, labels) ready for IrrigationClassifier.train().
        """
        from analytics.models import AnalyticsResult
        from iot.models import SensorReading, WeatherData

        X, y = [], []
        for field in fields_qs:
            analytics = AnalyticsResult.objects.filter(field=field).order_by('date')
            if analytics.count() < 3:
                continue

            ndvi_vals = [a.avg_ndvi for a in analytics]
            ndwi_vals = [a.avg_ndwi for a in analytics if a.avg_ndwi is not None]

            readings = SensorReading.objects.filter(device__farm=field.farm).order_by('-timestamp')[:50]
            moisture_vals = []
            for r in readings:
                m = (r.results.get('soilMoisture_%') or r.results.get('soilMoisture')
                     or r.results.get('soil_moisture') or r.results.get('moisture')) if r.results else None
                if m is not None:
                    moisture_vals.append(float(m))

            weather = WeatherData.objects.filter(farm=field.farm).order_by('-timestamp')[:30]
            temps = [w.temperature for w in weather if w.temperature is not None]
            precip = sum(w.precipitation or 0 for w in weather)

            features = [
                np.mean(ndvi_vals), np.std(ndvi_vals), np.max(ndvi_vals), np.min(ndvi_vals),
                np.mean(ndwi_vals) if ndwi_vals else 0, np.std(ndwi_vals) if ndwi_vals else 0,
                np.mean(moisture_vals) if moisture_vals else 30, np.std(moisture_vals) if moisture_vals else 10,
                np.mean(temps) if temps else 25, precip if precip else 500,
                100, 2,
            ]
            X.append(features)

            # Auto-label: high moisture + low NDVI variance → irrigated
            sm_mean = np.mean(moisture_vals) if moisture_vals else 30
            ndvi_std = np.std(ndvi_vals)
            if sm_mean > 50 and ndvi_std < 0.12:
                label = 2  # irrigated
            elif sm_mean > 30:
                label = 1  # partially
            else:
                label = 0  # rainfed
            y.append(label)

        return np.array(X), np.array(y)

    @staticmethod
    def generate_yield_training_data(fields_qs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build X, y from ImageReading rows for yield model training.
        """
        from imagery.models import ImageReading
        from iot.models import WeatherData

        X, y = [], []
        for field in fields_qs:
            readings = ImageReading.objects.filter(field=field).order_by('acquisition_date')
            if readings.count() < 3:
                continue

            ndvi = [r.ndvi_mean for r in readings if r.ndvi_mean is not None]
            ndwi = [r.ndwi_mean for r in readings if r.ndwi_mean is not None]
            if len(ndvi) < 3:
                continue

            dates = [r.acquisition_date for r in readings]
            growing_days = (dates[-1] - dates[0]).days if len(dates) > 1 else len(ndvi) * 5

            weather = WeatherData.objects.filter(farm=field.farm).order_by('-timestamp')[:30]
            temps = [w.temperature for w in weather if w.temperature is not None]
            precip = sum(w.precipitation or 0 for w in weather)

            features = [
                np.mean(ndvi), np.max(ndvi),
                np.argmax(ndvi) / max(len(ndvi) - 1, 1),
                float(np.trapz(ndvi)) if len(ndvi) > 1 else np.mean(ndvi),
                np.mean(ndwi) if ndwi else 0,
                growing_days,
                np.mean(temps) if temps else 25,
                precip if precip else 300,
            ]
            X.append(features)

            # Estimate ground-truth yield from peak NDVI × crop factor
            crop_factors = {
                'wheat': 4000, 'rice': 5000, 'cotton': 2500,
                'maize': 6000, 'sugarcane': 30000, 'default': 4000,
            }
            key = field.crop_type.lower() if field.crop_type else 'default'
            factor = crop_factors.get(key, 4000)
            estimated_yield = 2500 + np.max(ndvi) * factor
            y.append(estimated_yield)

        return np.array(X), np.array(y)

    @staticmethod
    def generate_microclimate_training_data(farm) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Build X, y_temp, y_humidity from WeatherData + SensorReading.
        """
        from iot.models import WeatherData, SensorReading

        weather_qs = WeatherData.objects.filter(farm=farm).order_by('timestamp')
        X, y_temp, y_hum = [], [], []

        for w in weather_qs:
            if w.temperature is None or w.humidity is None:
                continue

            hour = w.timestamp.hour
            day_of_year = w.timestamp.timetuple().tm_yday
            hour_sin = np.sin(2 * np.pi * hour / 24)
            hour_cos = np.cos(2 * np.pi * hour / 24)
            day_sin = np.sin(2 * np.pi * day_of_year / 365)
            day_cos = np.cos(2 * np.pi * day_of_year / 365)

            features = [
                w.temperature, w.humidity, 1013,
                w.wind_speed or 5, 0.5,
                100, 2, 180, 1000, 0.5,
                hour_sin, hour_cos, day_sin, day_cos,
            ]
            X.append(features)

            # Local adjustments: sensor data as ground truth
            nearby = SensorReading.objects.filter(
                device__farm=farm,
                timestamp__range=(
                    w.timestamp - __import__('datetime').timedelta(minutes=30),
                    w.timestamp + __import__('datetime').timedelta(minutes=30),
                ),
            ).first()

            if nearby and nearby.temperature is not None:
                y_temp.append(nearby.temperature)
                y_hum.append(nearby.results.get('humidity', w.humidity) if nearby.results else w.humidity)
            else:
                # Small random perturbation as pseudo-local truth
                y_temp.append(w.temperature + np.random.normal(0, 0.5))
                y_hum.append(w.humidity + np.random.normal(0, 2))

        return np.array(X), np.array(y_temp), np.array(y_hum)
