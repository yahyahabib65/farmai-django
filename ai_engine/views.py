"""
AI Engine views — callable from the dashboard UI.

Every workflow (training + inference) has a JSON API endpoint
and is wired to the AI Dashboard template buttons.
"""
import json
import logging

from django.http import JsonResponse
from django.shortcuts import render, get_object_or_404
from django.views.decorators.http import require_GET, require_POST, require_http_methods
from django.views.decorators.csrf import csrf_exempt
from django.utils import timezone

from core.models import Farm, FieldBoundary
from .models import MLModel, TrainingJob, InferenceResult

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
#  DASHBOARD PAGE
# ═══════════════════════════════════════════════════════════════

def ai_dashboard(request):
    """Render the full AI Engine dashboard."""
    farms = Farm.objects.all()
    fields = FieldBoundary.objects.all()
    models = MLModel.objects.filter(is_active=True)
    recent_jobs = TrainingJob.objects.all()[:10]
    recent_inferences = InferenceResult.objects.select_related('model', 'field')[:15]

    return render(request, 'ai_engine/ai_dashboard.html', {
        'farms': farms,
        'fields': fields,
        'ml_models': models,
        'training_jobs': recent_jobs,
        'recent_inferences': recent_inferences,
    })


# ═══════════════════════════════════════════════════════════════
#  TRAINING ENDPOINTS  (POST → queue Celery task)
# ═══════════════════════════════════════════════════════════════

@require_POST
def start_training(request):
    """
    POST /ai/train/
    Body: {"model_type": "irrigation_classification", "farm_id": 1}
    """
    try:
        body = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({'error': 'Invalid JSON'}, status=400)

    model_type = body.get('model_type')
    farm_id = body.get('farm_id')
    field_id = body.get('field_id')

    VALID = dict(MLModel.MODEL_TYPES)
    if model_type not in VALID:
        return JsonResponse({'error': f"Invalid model_type. Choose from: {list(VALID.keys())}"}, status=400)

    farm = Farm.objects.filter(id=farm_id).first() if farm_id else None
    field = FieldBoundary.objects.filter(id=field_id).first() if field_id else None

    job = TrainingJob.objects.create(
        model_type=model_type,
        farm=farm,
        field=field,
        config=body.get('config', {}),
    )

    # Dispatch to correct Celery task
    from . import tasks
    task_map = {
        'irrigation_classification': tasks.train_irrigation_model,
        'yield_prediction': tasks.train_yield_model,
        'microclimate': tasks.train_microclimate_model,
    }

    task_fn = task_map.get(model_type)
    if task_fn:
        result = task_fn.delay(job.id)
        job.celery_task_id = result.id
        job.save()
        return JsonResponse({
            'status': 'queued',
            'job_id': job.id,
            'task_id': result.id,
            'model_type': model_type,
        })
    else:
        # For types without dedicated training tasks, mark as unsupported training
        job.status = 'failed'
        job.error_message = f"Training not yet implemented for '{model_type}'. Use inference instead."
        job.save()
        return JsonResponse({
            'status': 'unsupported',
            'job_id': job.id,
            'message': f"Training for '{VALID[model_type]}' runs via inference workflows.",
        })


@require_GET
def training_status(request, job_id):
    """GET /ai/train/<job_id>/status/"""
    job = get_object_or_404(TrainingJob, id=job_id)
    return JsonResponse({
        'job_id': job.id,
        'model_type': job.model_type,
        'status': job.status,
        'log': job.log,
        'error': job.error_message,
        'metrics': job.result_metrics,
        'started_at': job.started_at.isoformat() if job.started_at else None,
        'completed_at': job.completed_at.isoformat() if job.completed_at else None,
        'model_id': job.trained_model_id,
    })


@require_GET
def list_training_jobs(request):
    """GET /ai/train/history/"""
    jobs = TrainingJob.objects.all()[:20]
    return JsonResponse({
        'jobs': [
            {
                'id': j.id,
                'model_type': j.model_type,
                'model_type_display': j.get_model_type_display(),
                'status': j.status,
                'metrics': j.result_metrics,
                'created_at': j.created_at.isoformat(),
                'completed_at': j.completed_at.isoformat() if j.completed_at else None,
            }
            for j in jobs
        ]
    })


# ═══════════════════════════════════════════════════════════════
#  INFERENCE ENDPOINTS  (POST → queue Celery task)
# ═══════════════════════════════════════════════════════════════

@require_POST
def run_inference(request):
    """
    POST /ai/infer/
    Body: {"workflow": "crop_classification", "field_id": 1}
    """
    try:
        body = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({'error': 'Invalid JSON'}, status=400)

    workflow = body.get('workflow')
    field_id = body.get('field_id')

    if not field_id:
        return JsonResponse({'error': 'field_id is required'}, status=400)

    field = get_object_or_404(FieldBoundary, id=field_id)

    from . import tasks

    WORKFLOWS = {
        'crop_classification': tasks.run_crop_classification,
        'ndvi_anomaly': tasks.run_ndvi_anomaly_detection,
        'soil_moisture': tasks.run_soil_moisture_estimation,
    }

    task_fn = WORKFLOWS.get(workflow)
    if not task_fn:
        return JsonResponse({'error': f"Unknown workflow. Choose from: {list(WORKFLOWS.keys())}"}, status=400)

    if workflow == 'crop_classification':
        result = task_fn.delay(field_id, body.get('image_path'))
    else:
        result = task_fn.delay(field_id)

    return JsonResponse({
        'status': 'queued',
        'task_id': result.id,
        'workflow': workflow,
        'field': field.name,
    })


@require_GET
def inference_results(request, field_id):
    """GET /ai/results/<field_id>/"""
    field = get_object_or_404(FieldBoundary, id=field_id)
    results = InferenceResult.objects.filter(field=field).select_related('model')[:20]

    return JsonResponse({
        'field': field.name,
        'field_id': field.id,
        'results': [
            {
                'id': r.id,
                'model': r.model.name,
                'model_type': r.model.get_model_type_display(),
                'source_type': r.source_type,
                'prediction': r.prediction,
                'confidence': r.confidence,
                'processed_at': r.processed_at.isoformat(),
            }
            for r in results
        ],
    })


# ═══════════════════════════════════════════════════════════════
#  MODEL REGISTRY
# ═══════════════════════════════════════════════════════════════

@require_GET
def list_models(request):
    """GET /ai/models/"""
    models = MLModel.objects.filter(is_active=True)
    return JsonResponse({
        'models': [
            {
                'id': m.id,
                'name': m.name,
                'type': m.get_model_type_display(),
                'model_type': m.model_type,
                'version': m.version,
                'architecture': m.architecture,
                'framework': m.framework,
                'accuracy': m.accuracy,
                'rmse': m.rmse,
                'r2_score': m.r2_score,
                'training_samples': m.training_samples,
                'created_at': m.created_at.isoformat(),
            }
            for m in models
        ]
    })


@require_GET
def model_detail(request, model_id):
    """GET /ai/models/<model_id>/"""
    m = get_object_or_404(MLModel, id=model_id)
    recent = InferenceResult.objects.filter(model=m).select_related('field')[:10]

    return JsonResponse({
        'id': m.id,
        'name': m.name,
        'type': m.get_model_type_display(),
        'version': m.version,
        'architecture': m.architecture,
        'accuracy': m.accuracy,
        'rmse': m.rmse,
        'r2_score': m.r2_score,
        'training_samples': m.training_samples,
        'description': m.description,
        'created_at': m.created_at.isoformat(),
        'recent_inferences': [
            {
                'field': r.field.name,
                'confidence': r.confidence,
                'processed_at': r.processed_at.isoformat(),
            }
            for r in recent
        ],
    })


# ═══════════════════════════════════════════════════════════════
#  QUICK STATS API (for dashboard cards)
# ═══════════════════════════════════════════════════════════════

@require_GET
def ai_stats(request):
    """GET /ai/stats/ — aggregate counts for dashboard."""
    return JsonResponse({
        'total_models': MLModel.objects.filter(is_active=True).count(),
        'total_inferences': InferenceResult.objects.count(),
        'total_training_jobs': TrainingJob.objects.count(),
        'completed_trainings': TrainingJob.objects.filter(status='completed').count(),
        'failed_trainings': TrainingJob.objects.filter(status='failed').count(),
        'running_trainings': TrainingJob.objects.filter(status='running').count(),
    })
