import os
from celery import Celery

# Set the default Django settings module
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'farm_system.settings')

app = Celery('farm_system')

# Read config from settings.py (variables starting with CELERY_)
app.config_from_object('django.conf:settings', namespace='CELERY')

# Auto-discover tasks in all installed apps (like imagery/tasks.py)
app.autodiscover_tasks()