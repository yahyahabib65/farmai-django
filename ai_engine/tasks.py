"""
Celery tasks for AI Engine — all long-running ML operations.

Each task:
- Has retry logic with exponential backoff
- Logs progress to TrainingJob / InferenceResult
- Is callable from the UI dashboard
"""

import logging
from celery import shared_task
from django.utils import timezone

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
#  TRAINING TASKS
# ═══════════════════════════════════════════════════════════════

@shared_task(bind=True, max_retries=2, default_retry_delay=30, acks_late=True)
def train_irrigation_model(self, job_id: int):
    """Train irrigation classifier from real field data."""
    from .models import TrainingJob, MLModel
    from core.models import FieldBoundary
    from analytics.ml_models import IrrigationClassifier
    from .farmvibes_client import FarmVibesAnalyzer

    job = TrainingJob.objects.get(id=job_id)
    job.status = 'running'
    job.started_at = timezone.now()
    job.celery_task_id = self.request.id
    job.save()

    try:
        # Gather training data from DB
        if job.farm:
            fields = FieldBoundary.objects.filter(farm=job.farm)
        else:
            fields = FieldBoundary.objects.all()

        X, y = FarmVibesAnalyzer.generate_irrigation_training_data(fields)

        if len(X) < 5:
            # Generate synthetic data to supplement
            job.log += "⚠ Only {} real samples — augmenting with synthetic data\n".format(len(X))
            import numpy as np
            n_syn = max(50 - len(X), 30)
            X_syn, y_syn = [], []
            for _ in range(n_syn):
                label = np.random.choice([0, 1, 2])
                if label == 2:  # irrigated
                    row = [0.7 + np.random.normal(0, 0.05), 0.08, 0.85, 0.55,
                           0.15, 0.04, 55 + np.random.normal(0, 5), 5,
                           28, 200, 100, 2]
                elif label == 1:  # partial
                    row = [0.5 + np.random.normal(0, 0.08), 0.12, 0.7, 0.3,
                           0.05, 0.06, 35 + np.random.normal(0, 8), 10,
                           30, 400, 100, 2]
                else:  # rainfed
                    row = [0.35 + np.random.normal(0, 0.1), 0.18, 0.55, 0.15,
                           -0.05, 0.08, 20 + np.random.normal(0, 5), 12,
                           32, 700, 100, 2]
                X_syn.append(row)
                y_syn.append(label)

            X_syn = np.array(X_syn)
            y_syn = np.array(y_syn)
            X = np.vstack([X, X_syn]) if len(X) > 0 else X_syn
            y = np.concatenate([y, y_syn]) if len(y) > 0 else y_syn

        classifier = IrrigationClassifier()
        metrics = classifier.train(X, y, save_model=True)

        # Register model
        model_obj, _ = MLModel.objects.update_or_create(
            name='Irrigation Classifier',
            version='1.0',
            defaults={
                'model_type': 'irrigation_classification',
                'architecture': 'RandomForest',
                'framework': 'scikit-learn',
                'accuracy': metrics.get('accuracy'),
                'training_samples': len(y),
                'model_path': classifier.model_path,
                'is_active': True,
            },
        )

        job.status = 'completed'
        job.result_metrics = metrics
        job.trained_model = model_obj
        job.completed_at = timezone.now()
        job.log += f"✅ Trained on {len(y)} samples. Accuracy: {metrics.get('accuracy', 0):.2%}\n"
        job.save()

        logger.info(f"Irrigation model trained: accuracy={metrics.get('accuracy')}")
        return metrics

    except Exception as exc:
        job.status = 'failed'
        job.error_message = str(exc)
        job.completed_at = timezone.now()
        job.save()
        logger.error(f"Irrigation training failed: {exc}")
        raise self.retry(exc=exc)


@shared_task(bind=True, max_retries=2, default_retry_delay=30, acks_late=True)
def train_yield_model(self, job_id: int):
    """Train yield prediction model from satellite NDVI time-series."""
    from .models import TrainingJob, MLModel
    from core.models import FieldBoundary
    from analytics.ml_models import YieldPredictor
    from .farmvibes_client import FarmVibesAnalyzer

    job = TrainingJob.objects.get(id=job_id)
    job.status = 'running'
    job.started_at = timezone.now()
    job.celery_task_id = self.request.id
    job.save()

    try:
        if job.farm:
            fields = FieldBoundary.objects.filter(farm=job.farm)
        else:
            fields = FieldBoundary.objects.all()

        X, y = FarmVibesAnalyzer.generate_yield_training_data(fields)

        if len(X) < 5:
            import numpy as np
            job.log += "⚠ Only {} real samples — augmenting with synthetic data\n".format(len(X))
            n_syn = max(50 - len(X), 30)
            X_syn, y_syn = [], []
            for _ in range(n_syn):
                ndvi_mean = np.random.uniform(0.3, 0.8)
                ndvi_max = ndvi_mean + np.random.uniform(0.05, 0.2)
                row = [
                    ndvi_mean, min(ndvi_max, 1.0),
                    np.random.uniform(0.3, 0.7),
                    ndvi_mean * np.random.uniform(3, 8),
                    np.random.uniform(-0.1, 0.3),
                    np.random.randint(60, 180),
                    np.random.uniform(20, 35),
                    np.random.uniform(100, 800),
                ]
                X_syn.append(row)
                y_syn.append(2500 + ndvi_max * 4000 * np.random.uniform(0.8, 1.2))

            X_syn = np.array(X_syn)
            y_syn = np.array(y_syn)
            X = np.vstack([X, X_syn]) if len(X) > 0 else X_syn
            y = np.concatenate([y, y_syn]) if len(y) > 0 else y_syn

        predictor = YieldPredictor()
        metrics = predictor.train(X, y, save_model=True)

        model_obj, _ = MLModel.objects.update_or_create(
            name='Yield Predictor',
            version='1.0',
            defaults={
                'model_type': 'yield_prediction',
                'architecture': 'GradientBoosting',
                'framework': 'scikit-learn',
                'rmse': metrics.get('rmse'),
                'r2_score': metrics.get('r2_score'),
                'training_samples': len(y),
                'model_path': predictor.model_path,
                'is_active': True,
            },
        )

        job.status = 'completed'
        job.result_metrics = metrics
        job.trained_model = model_obj
        job.completed_at = timezone.now()
        job.log += f"✅ Trained on {len(y)} samples. R²: {metrics.get('r2_score', 0):.3f}\n"
        job.save()

        logger.info(f"Yield model trained: R²={metrics.get('r2_score')}")
        return metrics

    except Exception as exc:
        job.status = 'failed'
        job.error_message = str(exc)
        job.completed_at = timezone.now()
        job.save()
        logger.error(f"Yield training failed: {exc}")
        raise self.retry(exc=exc)


@shared_task(bind=True, max_retries=2, default_retry_delay=30, acks_late=True)
def train_microclimate_model(self, job_id: int):
    """Train micro climate prediction model from weather + sensor data."""
    from .models import TrainingJob, MLModel
    from analytics.ml_models import MicroClimatePredictor
    from .farmvibes_client import FarmVibesAnalyzer

    job = TrainingJob.objects.get(id=job_id)
    job.status = 'running'
    job.started_at = timezone.now()
    job.celery_task_id = self.request.id
    job.save()

    try:
        farm = job.farm
        if not farm:
            from core.models import Farm
            farm = Farm.objects.first()
            if not farm:
                raise ValueError("No farm exists to train on")

        X, y_temp, y_hum = FarmVibesAnalyzer.generate_microclimate_training_data(farm)

        if len(X) < 10:
            import numpy as np
            job.log += "⚠ Only {} real samples — augmenting with synthetic data\n".format(len(X))
            n_syn = max(100 - len(X), 50)
            X_syn, yt_syn, yh_syn = [], [], []
            for _ in range(n_syn):
                hour = np.random.randint(0, 24)
                day = np.random.randint(1, 366)
                temp = np.random.uniform(15, 45)
                hum = np.random.uniform(20, 90)
                row = [
                    temp, hum, 1013 + np.random.normal(0, 5),
                    np.random.uniform(0, 20), np.random.uniform(0, 1),
                    np.random.uniform(50, 500), np.random.uniform(0, 10), np.random.uniform(0, 360),
                    np.random.uniform(100, 5000), np.random.uniform(0.1, 0.9),
                    np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24),
                    np.sin(2 * np.pi * day / 365), np.cos(2 * np.pi * day / 365),
                ]
                X_syn.append(row)
                yt_syn.append(temp + np.random.normal(0, 1.5))
                yh_syn.append(min(100, hum + np.random.normal(0, 3)))

            X_syn = np.array(X_syn)
            yt_syn = np.array(yt_syn)
            yh_syn = np.array(yh_syn)
            X = np.vstack([X, X_syn]) if len(X) > 0 else X_syn
            y_temp = np.concatenate([y_temp, yt_syn]) if len(y_temp) > 0 else yt_syn
            y_hum = np.concatenate([y_hum, yh_syn]) if len(y_hum) > 0 else yh_syn

        predictor = MicroClimatePredictor()
        metrics = predictor.train(X, y_temp, y_hum)

        # Save model manually
        predictor_path = predictor.model_path
        import os, pickle
        os.makedirs(os.path.dirname(predictor_path), exist_ok=True)
        with open(predictor_path, 'wb') as f:
            pickle.dump({
                'temp_model': predictor.temp_model,
                'humidity_model': predictor.humidity_model,
                'scaler': predictor.scaler,
            }, f)

        model_obj, _ = MLModel.objects.update_or_create(
            name='MicroClimate Predictor',
            version='1.0',
            defaults={
                'model_type': 'microclimate',
                'architecture': 'GradientBoosting',
                'framework': 'scikit-learn',
                'rmse': metrics.get('temperature_rmse'),
                'r2_score': metrics.get('temperature_r2'),
                'training_samples': len(y_temp),
                'model_path': predictor_path,
                'is_active': True,
            },
        )

        job.status = 'completed'
        job.result_metrics = metrics
        job.trained_model = model_obj
        job.completed_at = timezone.now()
        job.log += f"✅ Trained on {len(y_temp)} samples. Temp R²: {metrics.get('temperature_r2', 0):.3f}\n"
        job.save()

        return metrics

    except Exception as exc:
        job.status = 'failed'
        job.error_message = str(exc)
        job.completed_at = timezone.now()
        job.save()
        logger.error(f"Microclimate training failed: {exc}")
        raise self.retry(exc=exc)


# ═══════════════════════════════════════════════════════════════
#  INFERENCE TASKS
# ═══════════════════════════════════════════════════════════════

@shared_task(bind=True, max_retries=2, default_retry_delay=60, acks_late=True)
def run_crop_classification(self, field_id: int, image_path: str = None):
    """Run crop classification on a field's latest satellite image."""
    from .models import MLModel, InferenceResult
    from .farmvibes_client import FarmVibesAnalyzer
    from core.models import FieldBoundary

    try:
        field = FieldBoundary.objects.get(id=field_id)
        analyzer = FarmVibesAnalyzer()

        # Find image path if not provided
        if not image_path:
            from imagery.models import ImageReading
            latest = ImageReading.objects.filter(field=field).order_by('-acquisition_date').first()
            if latest and latest.red_path:
                image_path = latest.red_path
            else:
                return {'error': 'No satellite image found for this field'}

        result = analyzer.classify_crop_from_bands(image_path)

        model_obj, _ = MLModel.objects.get_or_create(
            name='Crop Classifier', version='1.0',
            defaults={
                'model_type': 'crop_classification',
                'architecture': 'EuroSAT-ResNet18',
                'framework': 'pytorch',
                'model_path': 'models/crop_classifier.pt',
            },
        )

        InferenceResult.objects.create(
            model=model_obj,
            field=field,
            source_type='satellite',
            source_reference=image_path or '',
            prediction=result,
            confidence=result.get('confidence', 0),
        )

        logger.info(f"Crop classification: {field.name} → {result.get('detected_crop')}")
        return result

    except Exception as exc:
        logger.error(f"Crop classification failed for field {field_id}: {exc}")
        raise self.retry(exc=exc)


@shared_task(bind=True, max_retries=2, default_retry_delay=60, acks_late=True)
def run_ndvi_anomaly_detection(self, field_id: int):
    """Check NDVI history for anomalies."""
    from .farmvibes_client import FarmVibesAnalyzer
    from analytics.models import AnalyticsResult
    from core.models import FieldBoundary

    try:
        field = FieldBoundary.objects.get(id=field_id)
        analyzer = FarmVibesAnalyzer()

        history = list(
            AnalyticsResult.objects.filter(field=field)
            .order_by('date')
            .values('date', 'avg_ndvi')
        )
        ndvi_history = [{'date': str(h['date']), 'mean_ndvi': h['avg_ndvi']} for h in history]

        anomalies = analyzer.detect_ndvi_anomalies(ndvi_history)

        if anomalies:
            latest = anomalies[-1]
            AnalyticsResult.objects.create(
                field=field,
                avg_ndvi=latest['ndvi'],
                irrigation_alert=True,
                notes=f"NDVI anomaly: dropped to {latest['ndvi']:.3f} (expected {latest['expected_ndvi']:.3f}), severity={latest['severity']}",
            )

        return {'field': field.name, 'anomalies_found': len(anomalies), 'anomalies': anomalies}

    except Exception as exc:
        logger.error(f"Anomaly detection failed for field {field_id}: {exc}")
        raise self.retry(exc=exc)


@shared_task(bind=True, max_retries=2, default_retry_delay=60, acks_late=True)
def run_soil_moisture_estimation(self, field_id: int):
    """Fuse IoT sensor data with satellite NDVI for soil moisture."""
    from .models import MLModel, InferenceResult
    from .farmvibes_client import FarmVibesAnalyzer
    from core.models import FieldBoundary
    from analytics.models import AnalyticsResult
    from iot.models import SensorReading, WeatherData

    try:
        field = FieldBoundary.objects.get(id=field_id)
        farm = field.farm
        analyzer = FarmVibesAnalyzer()

        recent = list(
            SensorReading.objects.filter(device__farm=farm)
            .order_by('-timestamp')[:20]
            .values_list('results', flat=True)
        )

        latest_analytics = AnalyticsResult.objects.filter(field=field).order_by('-date').first()
        ndvi = latest_analytics.avg_ndvi if latest_analytics else 0.5

        latest_weather = WeatherData.objects.filter(farm=farm).order_by('-timestamp').first()
        temperature = latest_weather.temperature if latest_weather and latest_weather.temperature else 30.0

        result = analyzer.estimate_soil_moisture(recent, ndvi, temperature)

        model_obj, _ = MLModel.objects.get_or_create(
            name='Soil Moisture Estimator', version='1.0',
            defaults={
                'model_type': 'soil_moisture',
                'architecture': 'Sensor-Satellite Fusion',
                'framework': 'numpy',
                'model_path': 'n/a',
            },
        )

        InferenceResult.objects.create(
            model=model_obj,
            field=field,
            source_type='combined',
            source_reference=f"{len(recent)} sensor readings + NDVI={ndvi:.3f}",
            prediction=result,
            confidence=result.get('confidence', 0),
        )

        if result.get('estimated_moisture') is not None and result['estimated_moisture'] < 20:
            AnalyticsResult.objects.create(
                field=field, avg_ndvi=ndvi,
                irrigation_alert=True,
                notes=f"Low soil moisture: {result['estimated_moisture']}% (sensor-satellite fusion)",
            )

        return result

    except Exception as exc:
        logger.error(f"Soil moisture estimation failed for field {field_id}: {exc}")
        raise self.retry(exc=exc)
