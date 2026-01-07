from django.urls import path
from .views import (
    DashboardView, 
    dashboard_view,
    NDVITimeSeriesView,
    IrrigationClassificationView,
    WeedDetectionView,
    SensorPlacementView,
    GHGCalculatorView,
    MicroClimateView,
    YieldPredictionView
)

urlpatterns = [
    # Dashboard
    path('dashboard/<int:farm_id>/', DashboardView.as_view()),
    path('dashboard/', dashboard_view, name='dashboard'),
    
    # NDVI Time-Series Analysis
    path('timeseries/<int:field_id>/', NDVITimeSeriesView.as_view(), name='ndvi-timeseries'),
    
    # Irrigation Classification
    path('irrigation/<int:field_id>/', IrrigationClassificationView.as_view(), name='irrigation-classification'),
    
    # Weed Detection
    path('weed-detection/', WeedDetectionView.as_view(), name='weed-detection'),
    
    # Optimal Sensor Placement
    path('sensor-placement/<int:field_id>/', SensorPlacementView.as_view(), name='sensor-placement'),
    
    # GHG Calculator
    path('ghg/<int:field_id>/', GHGCalculatorView.as_view(), name='ghg-calculator'),
    
    # Micro Climate Prediction
    path('microclimate/', MicroClimateView.as_view(), name='microclimate'),
    
    # Yield Prediction (based on real satellite NDVI data)
    path('yield/<int:field_id>/', YieldPredictionView.as_view(), name='yield-prediction'),
]