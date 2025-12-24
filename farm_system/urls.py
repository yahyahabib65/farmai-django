"""
URL configuration for farm_system project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/6.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include
from rest_framework.routers import DefaultRouter
from imagery.views import DroneImageViewSet, DroneImageUploadView, AnalyticsView, SentinelImageUploadView, SatelliteImageListView, ImagePreviewView
from core.views import FarmViewSet, FieldBoundaryViewSet, DeviceViewSet, AssetViewSet, ManualImportView, farm_data_view, farm_filter_view
from iot.views import SensorReadingViewSet, ThingsBoardSyncView, ThingsBoardHistoricalSyncView, FarmSensorSummaryView, SensorTimeSeriesView, ThingsBoardTimeSeriesView, ThingsBoardKeysView
from django.shortcuts import redirect

router = DefaultRouter()
router.register(r'imagery', DroneImageViewSet, basename='droneimage')
router.register(r'farms', FarmViewSet, basename='farm')
router.register(r'fields', FieldBoundaryViewSet, basename='field')
router.register(r'devices', DeviceViewSet, basename='device')
router.register(r'assets', AssetViewSet, basename='asset')
router.register(r'readings', SensorReadingViewSet, basename='reading')

def home_view(request):
    return redirect('/analytics/dashboard/')

urlpatterns = [
    path('admin/', admin.site.urls),
    path('analytics/', include('analytics.urls')),
    path('api/manual-import/', ManualImportView.as_view(), name='manual-import'),
    path('api/drone-upload/', DroneImageUploadView.as_view(), name='drone-upload'),
    path('api/analytics/run/', AnalyticsView.as_view(), name='analytics-run'),
    path('api/sentinel-upload/', SentinelImageUploadView.as_view(), name='sentinel-upload'),
    path('api/satellite-images/', SatelliteImageListView.as_view(), name='satellite-images'),
    path('api/image-preview/', ImagePreviewView.as_view(), name='image-preview'),
    path('api/farms/<int:farm_id>/data/', farm_data_view, name='farm-data'),
    path('api/farms/filter/', farm_filter_view, name='farm-filter'),
    # ThingsBoard IoT sync endpoints
    path('api/iot/sync/', ThingsBoardSyncView.as_view(), name='thingsboard-sync'),
    path('api/iot/sync-historical/', ThingsBoardHistoricalSyncView.as_view(), name='thingsboard-sync-historical'),
    path('api/iot/summary/', FarmSensorSummaryView.as_view(), name='sensor-summary'),
    path('api/iot/summary/<int:farm_id>/', FarmSensorSummaryView.as_view(), name='sensor-summary-farm'),
    path('api/iot/timeseries/', SensorTimeSeriesView.as_view(), name='sensor-timeseries'),
    path('api/iot/timeseries/<int:farm_id>/', SensorTimeSeriesView.as_view(), name='sensor-timeseries-farm'),
    # ThingsBoard direct telemetry API
    path('api/iot/thingsboard-timeseries/', ThingsBoardTimeSeriesView.as_view(), name='thingsboard-timeseries'),
    path('api/iot/thingsboard-timeseries/<int:device_id>/', ThingsBoardTimeSeriesView.as_view(), name='thingsboard-timeseries-device'),
    path('api/iot/thingsboard-keys/', ThingsBoardKeysView.as_view(), name='thingsboard-keys'),
    # Imagery workflows
    path('api/imagery/', include('imagery.urls')),
    path('api/', include(router.urls)),
    path('', home_view, name='home'),
]
