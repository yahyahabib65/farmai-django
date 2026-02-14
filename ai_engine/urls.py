from django.urls import path
from . import views

app_name = 'ai_engine'

urlpatterns = [
    # Dashboard page
    path('', views.ai_dashboard, name='ai-dashboard'),

    # Training
    path('train/', views.start_training, name='start-training'),
    path('train/<int:job_id>/status/', views.training_status, name='training-status'),
    path('train/history/', views.list_training_jobs, name='training-history'),

    # Inference
    path('infer/', views.run_inference, name='run-inference'),
    path('results/<int:field_id>/', views.inference_results, name='inference-results'),

    # Model registry
    path('models/', views.list_models, name='list-models'),
    path('models/<int:model_id>/', views.model_detail, name='model-detail'),

    # Stats
    path('stats/', views.ai_stats, name='ai-stats'),
]
