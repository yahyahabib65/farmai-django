from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from core.models import FieldBoundary, Device, Farm
from iot.models import SensorReading
from analytics.models import AnalyticsResult
from rest_framework.parsers import MultiPartParser
from imagery.models import SentinelImage, DroneImage
# Lazy import geoai to avoid heavy torch/torchvision dependencies at startup
# from geoai import semantic_segmentation
import json
from django.shortcuts import render
from datetime import datetime, timedelta

class DashboardView(APIView):
    def get(self, request, farm_id):
        # --- PART 1: FIELD MAP & HEALTH ---
        fields = FieldBoundary.objects.filter(farm_id=farm_id)
        features = []
        alerts_count = 0
        
        for f in fields:
            # Get latest health check
            last_check = AnalyticsResult.objects.filter(field=f).order_by('-date').first()
            
            props = {
                "id": f.id,
                "name": f.name,
                "crop": f.crop_type,
                "area_ha": f.area_hectares,
                "ndvi": last_check.avg_ndvi if last_check else None,
                "status": "ALERT" if (last_check and last_check.irrigation_alert) else "OK"
            }
            
            if props["status"] == "ALERT":
                alerts_count += 1

            features.append({
                "type": "Feature",
                "geometry": json.loads(f.boundary.json),
                "properties": props
            })

        # --- PART 2: IOT SENSOR DATA ---
        # Get last 24 hours of data for charts
        sensors = SensorReading.objects.filter(device__farm_id=farm_id).order_by('-timestamp')[:50]
        chart_data = [
            {
                "time": s.timestamp.strftime("%H:%M"), 
                "moisture": s.moisture,
                "temp": s.temperature
            } 
            for s in sensors
        ]

        # --- PART 3: RESPONSE ---
        return Response({
            "farm_summary": {
                "total_fields": fields.count(),
                "active_alerts": alerts_count
            },
            "map_data": {
                "type": "FeatureCollection",
                "features": features
            },
            "sensor_history": chart_data
        })

class SentinelProcessingView(APIView):
    parser_classes = [MultiPartParser]

    def post(self, request):
        # Validate and save uploaded Sentinel image
        image_file = request.FILES.get('image')
        if not image_file:
            return Response({"error": "No image file provided."}, status=400)

        sentinel_image = SentinelImage.objects.create(image=image_file)

        # Process NDVI and NDWI
        try:
            ndvi_result = semantic_segmentation(sentinel_image.image.path, model='ndvi')
            ndwi_result = semantic_segmentation(sentinel_image.image.path, model='ndwi')

            # Save results to database
            sentinel_image.ndvi_result = ndvi_result
            sentinel_image.ndwi_result = ndwi_result
            sentinel_image.save()

            return Response({
                "message": "Processing complete.",
                "ndvi": ndvi_result,
                "ndwi": ndwi_result
            })
        except Exception as e:
            return Response({"error": str(e)}, status=500)

def dashboard_view(request):
    from django.conf import settings as django_settings
    
    farms = Farm.objects.all()
    fields_count = FieldBoundary.objects.count()
    devices_count = Device.objects.count()
    
    # Count processed images (handle missing processed field gracefully)
    try:
        drone_count = DroneImage.objects.filter(processed=True).count()
    except:
        drone_count = DroneImage.objects.count()
    
    try:
        sentinel_count = SentinelImage.objects.filter(processed=True).count()
    except:
        sentinel_count = SentinelImage.objects.count()
    
    images_count = drone_count + sentinel_count
    
    # Get configurable GIS settings
    context = {
        'farms': farms,
        'fields_count': fields_count,
        'devices_count': devices_count,
        'images_count': images_count,
        # FarmAI GIS configuration from settings
        'default_lat': getattr(django_settings, 'FARMAI_DEFAULT_LAT', 31.4697),
        'default_lon': getattr(django_settings, 'FARMAI_DEFAULT_LON', 74.4101),
        'default_zoom': getattr(django_settings, 'FARMAI_DEFAULT_ZOOM', 12),
        'max_distance_km': getattr(django_settings, 'FARMAI_MAX_FIELD_DISTANCE_KM', 50),
        'min_area_ha': getattr(django_settings, 'FARMAI_MIN_FIELD_AREA_HA', 0.00),
        'max_area_ha': getattr(django_settings, 'FARMAI_MAX_FIELD_AREA_HA', 10000),
    }
    return render(request, 'dashboard.html', context)


# ============== NEW WORKFLOW VIEWS ==============

class NDVITimeSeriesView(APIView):
    """
    Analyze NDVI time-series for a field.
    
    Endpoints:
    - GET /api/analytics/timeseries/<field_id>/
    - GET /api/analytics/timeseries/<field_id>/?analysis=crop_cycles
    - GET /api/analytics/timeseries/<field_id>/?analysis=harvest
    - GET /api/analytics/timeseries/<field_id>/?analysis=degradation
    """
    
    def get(self, request, field_id):
        from analytics.time_series import analyze_field_ndvi_timeseries, NDVITimeSeries
        
        analysis_type = request.query_params.get('analysis', 'all')
        start_date = request.query_params.get('start_date')
        end_date = request.query_params.get('end_date')
        
        # Parse dates
        start = datetime.fromisoformat(start_date) if start_date else None
        end = datetime.fromisoformat(end_date) if end_date else None
        
        result = analyze_field_ndvi_timeseries(field_id, start, end)
        
        if 'error' in result:
            return Response(result, status=status.HTTP_400_BAD_REQUEST)
        
        # Filter by analysis type if specified
        if analysis_type == 'crop_cycles':
            return Response({
                'field_id': result['field_id'],
                'field_name': result['field_name'],
                'crop_cycles': result['crop_cycles']
            })
        elif analysis_type == 'harvest':
            return Response({
                'field_id': result['field_id'],
                'field_name': result['field_name'],
                'harvest_germination': result['harvest_germination']
            })
        elif analysis_type == 'degradation':
            return Response({
                'field_id': result['field_id'],
                'field_name': result['field_name'],
                'land_degradation': result['land_degradation']
            })
        
        return Response(result)


class IrrigationClassificationView(APIView):
    """
    Classify irrigation status using ML.
    
    GET /api/analytics/irrigation/<field_id>/
    """
    
    def get(self, request, field_id):
        from analytics.ml_models import classify_field_irrigation
        
        result = classify_field_irrigation(field_id)
        
        if 'error' in result:
            return Response(result, status=status.HTTP_400_BAD_REQUEST)
        
        return Response(result)


class WeedDetectionView(APIView):
    """
    Detect weeds in field imagery using GMM.
    
    POST /api/analytics/weed-detection/
    """
    parser_classes = [MultiPartParser]
    
    def post(self, request):
        from analytics.ml_models import WeedDetector
        from PIL import Image
        import numpy as np
        
        image_file = request.FILES.get('image')
        if not image_file:
            return Response({'error': 'No image provided'}, status=status.HTTP_400_BAD_REQUEST)
        
        # Load image
        img = Image.open(image_file)
        img_array = np.array(img)
        
        if len(img_array.shape) != 3 or img_array.shape[2] < 3:
            return Response({'error': 'Image must be RGB'}, status=status.HTTP_400_BAD_REQUEST)
        
        # Detect weeds
        detector = WeedDetector(n_components=3)
        result = detector.detect_from_image(
            img_array[:, :, 0],
            img_array[:, :, 1],
            img_array[:, :, 2]
        )
        
        # Remove large arrays from response
        if 'cluster_labels' in result:
            del result['cluster_labels']
        if 'semantic_labels' in result:
            del result['semantic_labels']
        
        return Response(result)


class SensorPlacementView(APIView):
    """
    Find optimal sensor placement locations.
    
    GET /api/analytics/sensor-placement/<field_id>/
    GET /api/analytics/sensor-placement/<field_id>/?n_sensors=5&method=auto
    """
    
    def get(self, request, field_id):
        from analytics.sensor_placement import find_sensor_locations_for_field
        
        n_sensors = int(request.query_params.get('n_sensors', 5))
        method = request.query_params.get('method', 'auto')
        
        result = find_sensor_locations_for_field(field_id, n_sensors, method)
        
        if 'error' in result:
            return Response(result, status=status.HTTP_400_BAD_REQUEST)
        
        return Response(result)


class GHGCalculatorView(APIView):
    """
    Calculate greenhouse gas balance for a field.
    
    GET /api/analytics/ghg/<field_id>/
    POST /api/analytics/ghg/<field_id>/ (with custom inputs)
    """
    
    def get(self, request, field_id):
        from analytics.ghg_calculator import calculate_field_ghg
        
        result = calculate_field_ghg(field_id)
        
        if 'error' in result:
            return Response(result, status=status.HTTP_400_BAD_REQUEST)
        
        return Response(result)
    
    def post(self, request, field_id):
        from analytics.ghg_calculator import calculate_field_ghg
        
        synthetic_n = request.data.get('synthetic_n_kg')
        diesel = request.data.get('diesel_liters')
        
        result = calculate_field_ghg(
            field_id,
            synthetic_n_kg=float(synthetic_n) if synthetic_n else None,
            diesel_liters=float(diesel) if diesel else None
        )
        
        if 'error' in result:
            return Response(result, status=status.HTTP_400_BAD_REQUEST)
        
        return Response(result)


class MicroClimateView(APIView):
    """
    Predict micro climate for a location.
    
    POST /api/analytics/microclimate/
    """
    
    def post(self, request):
        from analytics.ml_models import MicroClimatePredictor
        
        predictor = MicroClimatePredictor()
        
        result = predictor.predict(
            global_temp=float(request.data.get('global_temp', 25)),
            global_humidity=float(request.data.get('global_humidity', 60)),
            global_pressure=float(request.data.get('global_pressure', 1013)),
            wind_speed=float(request.data.get('wind_speed', 5)),
            cloud_cover=float(request.data.get('cloud_cover', 0.5)),
            elevation=float(request.data.get('elevation', 100)),
            slope=float(request.data.get('slope', 2)),
            aspect=float(request.data.get('aspect', 180)),
            distance_to_water=float(request.data.get('distance_to_water', 1000)),
            ndvi=float(request.data.get('ndvi', 0.5)),
            hour_of_day=int(request.data.get('hour_of_day', 12)),
            day_of_year=int(request.data.get('day_of_year', 180))
        )
        
        return Response(result)


class YieldPredictionView(APIView):
    """
    Predict crop yield for a field based on REAL satellite imagery.
    Uses NDVI time series from ImageReading model - NO fake data.
    
    GET /api/analytics/yield/<field_id>/
    """
    
    def get(self, request, field_id):
        from analytics.ml_models import YieldPredictor
        from imagery.models import ImageReading
        from core.models import FieldBoundary
        
        try:
            field = FieldBoundary.objects.get(id=field_id)
        except FieldBoundary.DoesNotExist:
            return Response({'error': f'Field {field_id} not found'}, status=status.HTTP_404_NOT_FOUND)
        
        # Get REAL NDVI/NDWI time series from ImageReading (satellite data)
        readings = ImageReading.objects.filter(field=field).order_by('acquisition_date')
        
        if readings.count() < 3:
            return Response({
                'error': 'Insufficient satellite data for yield prediction',
                'message': f'Need at least 3 satellite readings, found {readings.count()}',
                'field_id': field_id,
                'field_name': field.name
            }, status=status.HTTP_400_BAD_REQUEST)
        
        # Extract real data
        ndvi_series = [r.ndvi_mean for r in readings if r.ndvi_mean is not None]
        ndwi_series = [r.ndwi_mean for r in readings if r.ndwi_mean is not None]
        dates = [r.acquisition_date.isoformat() for r in readings]
        
        # Get area
        area_ha = float(field.area_hectares) if field.area_hectares else 1.0
        
        # Get real temperature data if available
        from iot.models import WeatherData
        weather = WeatherData.objects.filter(farm=field.farm).order_by('-timestamp')[:30]
        temp_series = [w.temperature for w in weather if w.temperature is not None]
        precip_total = sum([w.precipitation or 0 for w in weather])
        
        # Predict using real data only
        predictor = YieldPredictor()
        result = predictor.predict(
            ndvi_series=ndvi_series,
            ndwi_series=ndwi_series,
            dates=dates,
            crop_type=field.crop_type,
            area_ha=area_ha,
            temperature_series=temp_series if temp_series else None,
            precipitation_total=precip_total if precip_total > 0 else None
        )
        
        result['field_id'] = field_id
        result['field_name'] = field.name
        result['satellite_readings'] = len(ndvi_series)
        result['date_range'] = {
            'first': dates[0] if dates else None,
            'last': dates[-1] if dates else None
        }
        
        return Response(result)