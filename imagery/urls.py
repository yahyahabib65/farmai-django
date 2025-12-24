from django.urls import path
from .views import (
    DroneImageViewSet,
    DroneImageUploadView,
    SentinelImageUploadView,
    AnalyticsView,
    SatelliteImageListView,
    SAMSegmentationView,
    FieldBoundarySegmentationView,
    ForestMapsView,
    TimelapseView,
    SpectralFusionView
)

urlpatterns = [
    # Existing endpoints
    path('drone/upload/', DroneImageUploadView.as_view(), name='drone-upload'),
    path('sentinel/upload/', SentinelImageUploadView.as_view(), name='sentinel-upload'),
    path('analytics/', AnalyticsView.as_view(), name='analytics'),
    path('satellite-images/', SatelliteImageListView.as_view(), name='satellite-images'),
    
    # SAM Segmentation
    path('sam/segment/', SAMSegmentationView.as_view(), name='sam-segment'),
    path('sam/field/<int:field_id>/', SAMSegmentationView.as_view(), name='sam-field'),
    
    # Field Boundary Segmentation
    path('segment-fields/', FieldBoundarySegmentationView.as_view(), name='segment-fields'),
    
    # Forest Maps
    path('forest-maps/<int:farm_id>/', ForestMapsView.as_view(), name='forest-maps-farm'),
    path('forest-maps/', ForestMapsView.as_view(), name='forest-maps'),
    
    # Timelapse
    path('timelapse/<int:field_id>/', TimelapseView.as_view(), name='timelapse'),
    
    # Spectral Fusion
    path('spectral-fusion/', SpectralFusionView.as_view(), name='spectral-fusion'),
]
