from django.contrib import admin
from .models import MLModel, TrainingJob, InferenceResult


@admin.register(MLModel)
class MLModelAdmin(admin.ModelAdmin):
    list_display = ['name', 'model_type', 'version', 'architecture', 'accuracy',
                     'r2_score', 'training_samples', 'is_active', 'created_at']
    list_filter = ['model_type', 'framework', 'is_active']
    search_fields = ['name', 'description']
    readonly_fields = ['created_at', 'updated_at']


@admin.register(TrainingJob)
class TrainingJobAdmin(admin.ModelAdmin):
    list_display = ['id', 'model_type', 'status', 'farm', 'field',
                     'started_at', 'completed_at']
    list_filter = ['model_type', 'status']
    readonly_fields = ['created_at', 'started_at', 'completed_at',
                        'celery_task_id', 'result_metrics']


@admin.register(InferenceResult)
class InferenceResultAdmin(admin.ModelAdmin):
    list_display = ['model', 'field', 'source_type', 'confidence', 'processed_at']
    list_filter = ['model__model_type', 'source_type']
    raw_id_fields = ['model', 'field']
    readonly_fields = ['processed_at']
