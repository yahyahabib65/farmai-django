from django.db import models
from django.core.validators import MinValueValidator, MaxValueValidator


class MLModel(models.Model):
    """Registry of trained ML models available in the system."""
    MODEL_TYPES = [
        ('crop_classification', 'Crop Classification'),
        ('disease_detection', 'Disease Detection'),
        ('yield_prediction', 'Yield Prediction'),
        ('weed_detection', 'Weed Detection'),
        ('soil_moisture', 'Soil Moisture Estimation'),
        ('irrigation_classification', 'Irrigation Classification'),
        ('microclimate', 'Micro Climate Prediction'),
        ('ndvi_anomaly', 'NDVI Anomaly Detection'),
    ]

    name = models.CharField(max_length=200)
    model_type = models.CharField(max_length=50, choices=MODEL_TYPES)
    version = models.CharField(max_length=20, default='1.0')
    description = models.TextField(blank=True)

    # Model file location
    model_path = models.CharField(max_length=500, help_text="Path on disk or in MinIO bucket")

    # Architecture info
    architecture = models.CharField(max_length=100, blank=True, help_text="e.g., RandomForest, GBR, ResNet50")
    framework = models.CharField(max_length=50, default='scikit-learn')

    # Performance metrics from training
    accuracy = models.FloatField(null=True, blank=True, validators=[MinValueValidator(0), MaxValueValidator(1)])
    f1_score = models.FloatField(null=True, blank=True, validators=[MinValueValidator(0), MaxValueValidator(1)])
    rmse = models.FloatField(null=True, blank=True)
    r2_score = models.FloatField(null=True, blank=True)
    training_samples = models.IntegerField(null=True, blank=True)
    training_dataset = models.CharField(max_length=200, blank=True)

    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-created_at']
        unique_together = ['name', 'version']

    def __str__(self):
        return f"{self.name} v{self.version} ({self.get_model_type_display()})"


class TrainingJob(models.Model):
    """Tracks ML model training jobs triggered from the UI."""
    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('running', 'Running'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
    ]

    model_type = models.CharField(max_length=50, choices=MLModel.MODEL_TYPES)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')

    # Configuration
    farm = models.ForeignKey('core.Farm', on_delete=models.CASCADE, null=True, blank=True)
    field = models.ForeignKey('core.FieldBoundary', on_delete=models.SET_NULL, null=True, blank=True)
    config = models.JSONField(default=dict, blank=True, help_text="Training hyperparameters and options")

    # Results
    result_metrics = models.JSONField(default=dict, blank=True)
    trained_model = models.ForeignKey(MLModel, on_delete=models.SET_NULL, null=True, blank=True,
                                       related_name='training_jobs')
    celery_task_id = models.CharField(max_length=255, blank=True)

    # Logs
    log = models.TextField(blank=True)
    error_message = models.TextField(blank=True)

    started_at = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"Train {self.get_model_type_display()} — {self.status}"


class InferenceResult(models.Model):
    """Results from running ML inference on farm data."""
    model = models.ForeignKey(MLModel, on_delete=models.CASCADE, related_name='inference_results')
    field = models.ForeignKey('core.FieldBoundary', on_delete=models.CASCADE, related_name='inference_results')

    source_type = models.CharField(max_length=20, choices=[
        ('satellite', 'Satellite Image'),
        ('drone', 'Drone Image'),
        ('sensor', 'Sensor Data'),
        ('combined', 'Multi-Source Fusion'),
    ])
    source_reference = models.CharField(max_length=500, blank=True)

    # Results
    prediction = models.JSONField(help_text="Model output — probabilities, values, maps, etc.")
    confidence = models.FloatField(validators=[MinValueValidator(0), MaxValueValidator(1)])

    # Output artifacts
    result_image_path = models.CharField(max_length=500, blank=True)

    processed_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-processed_at']

    def __str__(self):
        return f"{self.model.name} on {self.field.name} — {self.confidence:.0%}"
