from rest_framework.views import APIView
from rest_framework.response import Response
from core.models import FieldBoundary, Device
from iot.models import SensorReading
from analytics.models import AnalyticsResult
from rest_framework.parsers import MultiPartParser
from imagery.models import SentinelImage, DroneImage
# Lazy import geoai to avoid heavy torch/torchvision dependencies at startup
# from geoai import semantic_segmentation
import json
from django.shortcuts import render
from core.models import Farm

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
    
    context = {
        'farms': farms,
        'fields_count': fields_count,
        'devices_count': devices_count,
        'images_count': images_count,
    }
    return render(request, 'dashboard.html', context)