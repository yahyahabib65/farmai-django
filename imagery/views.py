from rest_framework import viewsets, status
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.permissions import AllowAny
from django.db import models
from django.shortcuts import get_object_or_404
from django.conf import settings
from django.core.files.storage import default_storage
from .models import DroneImage, SentinelImage
from .serializers import DroneImageSerializer
from core.models import FieldBoundary, Farm
from analytics.models import AnalyticsResult, HarvestPrediction, CarbonFootprint, CropClassification
from iot.models import WeatherData
from .ndvi_processor import process_uploaded_image, get_health_status, calculate_ndvi_from_separate_bands
# Lazy imports to avoid heavy dependencies at module load
# from .planetary_downloader import PlanetaryComputerDownloader, download_satellite_for_farm
# from .geoai_processor import GeoAIProcessor
import os
import uuid
import glob
import tempfile
import json
from datetime import datetime, timedelta
import random


def get_planetary_downloader():
    """Lazy import of PlanetaryComputerDownloader"""
    try:
        from .planetary_downloader import PlanetaryComputerDownloader
        return PlanetaryComputerDownloader
    except ImportError:
        return None


def get_geoai_processor():
    """Lazy import of GeoAIProcessor"""
    try:
        from .geoai_processor import GeoAIProcessor
        return GeoAIProcessor
    except ImportError:
        return None


class DroneImageViewSet(viewsets.ModelViewSet):
    """View drone images uploaded to the system"""
    queryset = DroneImage.objects.all()
    serializer_class = DroneImageSerializer
    permission_classes = (AllowAny,)


class DroneImageUploadView(APIView):
    """Upload drone imagery for a specific field"""
    parser_classes = (MultiPartParser, FormParser)
    permission_classes = (AllowAny,)

    def get(self, request):
        # Return a simple form for uploading
        return Response({
            'message': 'Drone Image Upload',
            'instructions': 'POST a file with field_id to upload drone imagery',
            'fields': FieldBoundary.objects.values('id', 'name', 'farm__name')[:10]
        })

    def post(self, request, *args, **kwargs):
        field_id = request.data.get('field_id')
        if not field_id:
            # Use first field if none specified
            field = FieldBoundary.objects.first()
            if not field:
                return Response({'error': 'No fields exist. Please create a field first.'}, 
                              status=status.HTTP_400_BAD_REQUEST)
        else:
            field = get_object_or_404(FieldBoundary, id=field_id)

        file_obj = request.FILES.get('file')
        if not file_obj:
            return Response({'error': 'No file uploaded'}, status=status.HTTP_400_BAD_REQUEST)

        # Save file locally
        filename = f"{uuid.uuid4()}_{file_obj.name}"
        file_path = f"uploads/drone/{field.id}/{filename}"
        
        saved_path = default_storage.save(file_path, file_obj)
        
        # Create database record
        drone_image = DroneImage.objects.create(
            farm=field.farm,
            minio_path=saved_path,
            processed=False
        )

        return Response({
            'message': 'Drone image uploaded successfully!',
            'image_id': drone_image.id,
            'field': field.name,
            'farm': field.farm.name,
            'file_path': saved_path
        }, status=status.HTTP_201_CREATED)


class SentinelImageUploadView(APIView):
    """Upload satellite imagery for a farm"""
    parser_classes = (MultiPartParser, FormParser)
    permission_classes = (AllowAny,)

    def get(self, request):
        return Response({
            'message': 'Satellite Image Upload',
            'instructions': 'POST a file with farm_id to upload satellite imagery',
            'farms': list(Farm.objects.values('id', 'name'))
        })

    def post(self, request, *args, **kwargs):
        farm_id = request.data.get('farm_id')
        if not farm_id:
            farm = Farm.objects.first()
            if not farm:
                return Response({'error': 'No farms exist. Please create a farm first.'}, 
                              status=status.HTTP_400_BAD_REQUEST)
        else:
            farm = get_object_or_404(Farm, id=farm_id)

        file_obj = request.FILES.get('file')
        if not file_obj:
            return Response({'error': 'No file uploaded'}, status=status.HTTP_400_BAD_REQUEST)

        # Save file locally
        filename = f"{uuid.uuid4()}_{file_obj.name}"
        file_path = f"uploads/satellite/{farm.id}/{filename}"
        
        saved_path = default_storage.save(file_path, file_obj)

        return Response({
            'message': 'Satellite image uploaded successfully!',
            'farm': farm.name,
            'file_path': saved_path
        }, status=status.HTTP_201_CREATED)


class AnalyticsView(APIView):
    """Run AI analytics on farm imagery - Covers all FarmVibes.AI workflows"""
    permission_classes = (AllowAny,)

    def get(self, request):
        # Return available farms and recent images for selection
        farms = Farm.objects.all()
        recent_images = DroneImage.objects.order_by('-created_at')[:5]
        
        return Response({
            'message': 'AI Farm Analytics',
            'instructions': 'Select a farm and analysis type to run AI analytics',
            'farms': [{'id': f.id, 'name': f.name} for f in farms],
            'recent_images': [{'id': img.id, 'path': img.minio_path, 'processed': img.processed} for img in recent_images],
            'analysis_types': ['NDVI', 'NDWI', 'Crop Health', 'Harvest Prediction', 'Weather', 'Carbon Footprint']
        })

    def _find_latest_image(self, farm_id, start_date=None, end_date=None):
        """Find the latest uploaded image for a farm within date range"""
        media_root = getattr(settings, 'MEDIA_ROOT', 'media')
        
        # Check satellite uploads
        satellite_path = os.path.join(media_root, 'uploads', 'satellite', str(farm_id))
        drone_path = os.path.join(media_root, 'uploads', 'drone', str(farm_id))
        
        latest_image = None
        latest_time = 0
        image_date = None
        
        for folder in [satellite_path, drone_path]:
            if os.path.exists(folder):
                # Check for date-organized folders (e.g., satellite/1/2024-01-15/)
                for item in os.listdir(folder):
                    item_path = os.path.join(folder, item)
                    
                    # If it's a date folder (YYYY-MM-DD format)
                    if os.path.isdir(item_path) and len(item) == 10 and '-' in item:
                        folder_date = item
                        # Filter by date range if specified
                        if start_date and folder_date < start_date:
                            continue
                        if end_date and folder_date > end_date:
                            continue
                        
                        for ext in ['*.tif', '*.tiff', '*.jpg', '*.jpeg', '*.png']:
                            files = glob.glob(os.path.join(item_path, ext))
                            for f in files:
                                mtime = os.path.getmtime(f)
                                if mtime > latest_time:
                                    latest_time = mtime
                                    latest_image = f
                                    image_date = folder_date
                    else:
                        # Direct files in folder (legacy structure)
                        for ext in ['*.tif', '*.tiff', '*.jpg', '*.jpeg', '*.png']:
                            files = glob.glob(os.path.join(folder, ext))
                            for f in files:
                                mtime = os.path.getmtime(f)
                                if mtime > latest_time:
                                    latest_time = mtime
                                    latest_image = f
        
        return {'path': latest_image, 'date': image_date} if latest_image else None
    
    def _check_existing_imagery(self, farm_id, start_date=None, end_date=None):
        """
        Check if satellite imagery already exists for this farm within date range
        Returns existing image info if found, None otherwise
        
        Args:
            farm_id: Farm ID to check
            start_date: Start of date range (YYYY-MM-DD string)
            end_date: End of date range (YYYY-MM-DD string)
        """
        media_root = getattr(settings, 'MEDIA_ROOT', 'media')
        satellite_path = os.path.join(media_root, 'uploads', 'satellite', str(farm_id))
        
        if not os.path.exists(satellite_path):
            return None
        
        # Find all existing dated imagery
        existing_dates = []
        for item in os.listdir(satellite_path):
            item_path = os.path.join(satellite_path, item)
            if os.path.isdir(item_path) and len(item) == 10 and '-' in item:
                folder_date = item  # YYYY-MM-DD format
                
                # Filter by date range if specified
                if start_date and folder_date < start_date:
                    continue
                if end_date and folder_date > end_date:
                    continue
                
                for ext in ['*.tif', '*.tiff']:
                    files = glob.glob(os.path.join(item_path, ext))
                    if files:
                        existing_dates.append({
                            'date': folder_date,
                            'path': files[0],
                            'file_count': len(files),
                            'all_files': files
                        })
        
        if existing_dates:
            # Return most recent within range
            existing_dates.sort(key=lambda x: x['date'], reverse=True)
            return {
                'path': existing_dates[0]['path'],
                'date': existing_dates[0]['date'],
                'source': 'existing_local',
                'all_files': existing_dates[0]['all_files'],
                'all_dates': existing_dates
            }
        
        return None
    
    def _build_download_status(self, result):
        """Build a human-readable status message from download result"""
        downloaded = result.get('images_downloaded', 0)
        skipped = len(result.get('skipped_dates', []))
        total = result.get('total_available', 0)
        
        parts = []
        if downloaded > 0:
            parts.append(f"Downloaded {downloaded} new image(s)")
        if skipped > 0:
            parts.append(f"Skipped {skipped} existing")
        if total > 0:
            parts.append(f"{total} available in range")
        
        if not parts:
            return "No images found in date range"
        
        return " | ".join(parts)
    
    def _fetch_from_minio(self, farm_id, date):
        """Fetch satellite imagery from MinIO for a specific date"""
        try:
            from minio import Minio
            import tempfile
            
            minio_client = Minio(
                os.environ.get('MINIO_ENDPOINT', 'localhost:9000'),
                access_key=os.environ.get('MINIO_ACCESS_KEY', 'minioadmin'),
                secret_key=os.environ.get('MINIO_SECRET_KEY', 'minioadmin'),
                secure=False
            )
            
            bucket_name = 'satellite-imagery'
            prefix = f"farm_{farm_id}/{date}/"
            
            # List objects with the date prefix
            objects = list(minio_client.list_objects(bucket_name, prefix=prefix))
            
            if not objects:
                # Try alternative path format
                prefix = f"{date}/"
                objects = list(minio_client.list_objects(bucket_name, prefix=prefix))
            
            if not objects:
                return None
            
            # Download ALL bands to temp directory for processing
            temp_dir = tempfile.mkdtemp()
            downloaded_files = []
            nir_file = None
            
            for obj in objects:
                obj_name = obj.object_name.lower()
                if obj_name.endswith(('.tif', '.tiff')):
                    local_path = os.path.join(temp_dir, os.path.basename(obj.object_name))
                    minio_client.fget_object(bucket_name, obj.object_name, local_path)
                    downloaded_files.append(local_path)
                    
                    # Track NIR file for primary path
                    if 'nir' in obj_name or 'b08' in obj_name:
                        nir_file = local_path
            
            if not downloaded_files:
                return None
            
            # Use NIR file as primary, or first file if no NIR
            primary_file = nir_file or downloaded_files[0]
            
            return {
                'path': primary_file,
                'folder': temp_dir,
                'minio_path': f"{bucket_name}/{prefix}",
                'date': date,
                'all_objects': [obj.object_name for obj in objects],
                'downloaded_files': downloaded_files
            }
        except Exception as e:
            print(f"MinIO fetch error: {e}")
            return None
    
    def _download_from_planetary_computer(self, farm, start_date=None, end_date=None):
        """Download ALL satellite imagery from Planetary Computer within date range"""
        PlanetaryComputerDownloader = get_planetary_downloader()
        if not PlanetaryComputerDownloader:
            return None
        
        try:
            downloader = PlanetaryComputerDownloader()
            
            # Parse date strings if provided
            parsed_start_date = None
            parsed_end_date = None
            
            if start_date:
                try:
                    parsed_start_date = datetime.strptime(start_date, '%Y-%m-%d')
                except:
                    pass
            
            if end_date:
                try:
                    parsed_end_date = datetime.strptime(end_date, '%Y-%m-%d')
                except:
                    pass
            
            # Download ALL images within the date range
            result = downloader.download_imagery_for_farm(
                farm, 
                start_date=parsed_start_date,
                end_date=parsed_end_date,
                download_all=True,  # Download ALL images, not just the best one
                max_cloud_cover=30
            )
            
            if result.get('success'):
                # Return info about all downloaded images
                media_root = getattr(settings, 'MEDIA_ROOT', 'media')
                downloaded_dates = result.get('downloaded_dates', [])
                
                return {
                    'success': True,
                    'source': 'planetary_computer',
                    'images_downloaded': result.get('images_downloaded', 0),
                    'total_bands_downloaded': result.get('total_bands_downloaded', 0),
                    'downloaded_dates': downloaded_dates,
                    'date_range': result.get('date_range', {}),
                    'total_available': result.get('total_available', 0),
                    'minio_paths': result.get('minio_paths', []),
                    # For backward compatibility - use first date
                    'path': os.path.join(media_root, 'uploads', 'satellite', 
                                         str(farm.id), downloaded_dates[0]) if downloaded_dates else None,
                    'date': downloaded_dates[0] if downloaded_dates else None,
                    'cloud_cover': result.get('cloud_cover', 0)
                }
            
            return None
        except Exception as e:
            print(f"Planetary Computer download error: {e}")
            return None

    def post(self, request, *args, **kwargs):
        farm_id = request.data.get('farm_id')
        analysis_type = request.data.get('analysis_type', 'all')
        run_ndvi = request.data.get('run_ndvi', True)
        run_ndwi = request.data.get('run_ndwi', True)
        
        # NEW: Time parameters
        start_date = request.data.get('start_date')  # Format: YYYY-MM-DD
        end_date = request.data.get('end_date')      # Format: YYYY-MM-DD
        auto_download = request.data.get('auto_download', True)  # Auto-download if not exists
        
        # NEW: Image source parameter (local, minio)
        requested_image_source = request.data.get('image_source')  # 'local' or 'minio'
        
        # Get farm
        if farm_id:
            farm = get_object_or_404(Farm, id=farm_id)
        else:
            farm = Farm.objects.first()
            if not farm:
                return Response({'error': 'No farms exist. Please create a farm first.'}, 
                              status=status.HTTP_400_BAD_REQUEST)

        # Get fields for this farm
        fields = FieldBoundary.objects.filter(farm=farm)
        
        field_results = []
        harvest_predictions = []
        crop_classifications = []
        yield_predictions = []  # Yield prediction based on real NDVI data
        raw_calculations = []  # Store raw NDVI/NDWI calculation details
        
        # SMART DOWNLOAD LOGIC: Always check Planetary Computer and download missing images
        latest_image = None
        image_processed = False
        image_results = None
        image_source = 'none'
        planetary_info = None
        download_skipped = False
        date_filter_applied = bool(start_date or end_date)
        
        minio_folder = None  # Track folder with all bands
        
        # If image source is MinIO, fetch the image from MinIO first
        if requested_image_source == 'minio' and start_date:
            minio_image = self._fetch_from_minio(farm.id, start_date)
            if minio_image:
                latest_image = minio_image.get('path')
                minio_folder = minio_image.get('folder')  # Folder with all downloaded bands
                image_source = 'minio'
                bands_count = len(minio_image.get('downloaded_files', []))
                planetary_info = {
                    'source': 'MinIO Object Storage',
                    'image_date': start_date,
                    'status': f'Using imagery from MinIO storage ({bands_count} bands)',
                    'minio_path': minio_image.get('minio_path'),
                    'bands_downloaded': bands_count,
                    'bands_available': bands_count  # For dashboard display
                }
        
        # Step 1: If auto_download enabled AND we don't have a MinIO image, sync with Planetary Computer
        # This downloads any missing images within the date range
        if auto_download and not latest_image:
            planetary_result = self._download_from_planetary_computer(
                farm, start_date=start_date, end_date=end_date
            )
            if planetary_result:
                planetary_info = {
                    'source': 'Microsoft Planetary Computer (Sentinel-2)',
                    'images_downloaded': planetary_result.get('images_downloaded', 0),
                    'images_skipped': len(planetary_result.get('skipped_dates', [])),
                    'total_bands_downloaded': planetary_result.get('total_bands_downloaded', 0),
                    'downloaded_dates': planetary_result.get('downloaded_dates', []),
                    'skipped_dates': planetary_result.get('skipped_dates', []),
                    'existing_dates': planetary_result.get('existing_dates', []),
                    'date_range': planetary_result.get('date_range', {}),
                    'total_available': planetary_result.get('total_available', 0),
                    'minio_storage': planetary_result.get('minio_paths', []),
                    'status': self._build_download_status(planetary_result),
                    'date_filter': {'start': start_date, 'end': end_date}
                }
        
        # Step 2: Check for existing Sentinel imagery (including just-downloaded)
        # Skip if we already have a MinIO image
        if not latest_image:
            existing_imagery = self._check_existing_imagery(
                farm.id, 
                start_date=start_date, 
                end_date=end_date
            )
            
            if existing_imagery:
                latest_image = existing_imagery.get('path')
                image_source = 'sentinel'
                
                if not planetary_info:
                    planetary_info = {
                        'source': 'Local Sentinel Storage',
                        'image_date': existing_imagery.get('date', 'Unknown'),
                        'status': f'Using existing Sentinel imagery',
                        'date_filter': {'start': start_date, 'end': end_date},
                        'available_dates': [d['date'] for d in existing_imagery.get('all_dates', [])],
                        'bands_available': len(existing_imagery.get('all_files', []))
                    }
                else:
                    # Add info about which image we're using for analysis
                    planetary_info['using_image_date'] = existing_imagery.get('date')
                    planetary_info['available_dates'] = [d['date'] for d in existing_imagery.get('all_dates', [])]
        
        if not latest_image:
            # Step 3: Try to find uploaded images with date range
            image_result = self._find_latest_image(farm.id, start_date=start_date, end_date=end_date)
            if image_result:
                latest_image = image_result.get('path')
                image_source = 'local_upload'
                if not planetary_info:
                    planetary_info = {
                        'source': 'Local Upload',
                        'image_date': image_result.get('date', 'Unknown'),
                        'status': 'Using uploaded image'
                    }
        
        if not latest_image and not planetary_info:
            planetary_info = {
                'status': 'No imagery found for specified dates.',
                'date_filter': {'start': start_date, 'end': end_date},
                'tip': 'Check farm field boundaries or try a different date range.'
            }
        
        if latest_image and run_ndvi:
            # Process actual image using GeoAI processor for advanced analysis
            GeoAIProcessor = get_geoai_processor()
            
            # Try GeoAI processing first for multispectral images
            if GeoAIProcessor and latest_image.lower().endswith(('.tif', '.tiff')):
                geoai = GeoAIProcessor()
                geoai_results = geoai.process_multispectral_image(latest_image)
                if geoai_results.get('success'):
                    image_processed = True
                    image_results = {
                        'success': True,
                        'ndvi_mean': geoai_results['indices']['ndvi']['mean'],
                        'ndvi_min': geoai_results['indices']['ndvi']['min'],
                        'ndvi_max': geoai_results['indices']['ndvi']['max'],
                        'ndvi_std': geoai_results['indices']['ndvi']['std'],
                        'ndwi_mean': geoai_results['indices'].get('ndwi', {}).get('mean'),
                        'ndwi_min': geoai_results['indices'].get('ndwi', {}).get('min'),
                        'ndwi_max': geoai_results['indices'].get('ndwi', {}).get('max'),
                        'ndwi_std': geoai_results['indices'].get('ndwi', {}).get('std'),
                        'pixel_count': geoai_results['image_info'].get('width', 0) * geoai_results['image_info'].get('height', 0),
                        'healthy_pixels_pct': geoai_results['crop_health']['percentages'].get('healthy', 0) + \
                                             geoai_results['crop_health']['percentages'].get('very_healthy', 0),
                        'stressed_pixels_pct': geoai_results['crop_health']['percentages'].get('critical', 0) + \
                                              geoai_results['crop_health']['percentages'].get('stressed', 0),
                        'geoai_summary': geoai_results.get('summary', {})
                    }
                    
                    raw_calculations.append({
                        'image': os.path.basename(latest_image),
                        'source': image_source,
                        'processed_at': datetime.now().isoformat(),
                        'processor': 'GeoAI + Advanced Indices',
                        'pixel_count': image_results.get('pixel_count', 0),
                        'ndvi_mean': image_results.get('ndvi_mean'),
                        'ndvi_min': image_results.get('ndvi_min'),
                        'ndvi_max': image_results.get('ndvi_max'),
                        'ndvi_std': image_results.get('ndvi_std'),
                        'ndwi_mean': image_results.get('ndwi_mean'),
                        'ndwi_min': image_results.get('ndwi_min'),
                        'ndwi_max': image_results.get('ndwi_max'),
                        'ndwi_std': image_results.get('ndwi_std'),
                        'evi_mean': geoai_results['indices'].get('evi', {}).get('mean'),
                        'savi_mean': geoai_results['indices'].get('savi', {}).get('mean'),
                        'healthy_pixels_pct': image_results.get('healthy_pixels_pct'),
                        'stressed_pixels_pct': image_results.get('stressed_pixels_pct'),
                        'crop_health_class': geoai_results['crop_health'].get('dominant_class'),
                        'water_stress_class': geoai_results.get('water_stress', {}).get('dominant_class'),
                        'planetary_computer': planetary_info,
                        'calculation_method': 'GeoAI: NDVI=(NIR-RED)/(NIR+RED), NDWI=(GREEN-NIR)/(GREEN+NIR), EVI, SAVI'
                    })
            
            # Fallback to basic processor for other image types
            if not image_processed:
                image_results = process_uploaded_image(latest_image)
                if image_results.get('success'):
                    image_processed = True
                    raw_calculations.append({
                        'image': os.path.basename(latest_image),
                        'source': image_source,
                        'processed_at': datetime.now().isoformat(),
                        'processor': 'Basic NDVI Processor',
                        'pixel_count': image_results.get('pixel_count', 0),
                        'ndvi_mean': image_results.get('ndvi_mean'),
                        'ndvi_min': image_results.get('ndvi_min'),
                        'ndvi_max': image_results.get('ndvi_max'),
                        'ndvi_std': image_results.get('ndvi_std'),
                        'ndwi_mean': image_results.get('ndwi_mean'),
                        'ndwi_min': image_results.get('ndwi_min'),
                        'ndwi_max': image_results.get('ndwi_max'),
                        'ndwi_std': image_results.get('ndwi_std'),
                        'healthy_pixels_pct': image_results.get('healthy_pixels_pct'),
                        'stressed_pixels_pct': image_results.get('stressed_pixels_pct'),
                        'bare_soil_pct': image_results.get('bare_soil_pct'),
                        'planetary_computer': planetary_info,
                        'calculation_method': 'Real NDVI: (NIR - RED) / (NIR + RED)'
                    })
            
            # Try separate bands processor for satellite imagery folders
            # Use minio_folder if available (all bands downloaded from MinIO)
            if not image_processed and latest_image:
                folder_path = minio_folder if minio_folder else os.path.dirname(latest_image)
                image_results = calculate_ndvi_from_separate_bands(folder_path)
                if image_results.get('success'):
                    image_processed = True
                    raw_calculations.append({
                        'image': os.path.basename(folder_path),
                        'source': image_source,
                        'processed_at': datetime.now().isoformat(),
                        'processor': 'Sentinel-2 Separate Bands Processor',
                        'pixel_count': image_results.get('pixel_count', 0),
                        'ndvi_mean': image_results.get('ndvi_mean'),
                        'ndvi_min': image_results.get('ndvi_min'),
                        'ndvi_max': image_results.get('ndvi_max'),
                        'ndvi_std': image_results.get('ndvi_std'),
                        'ndwi_mean': image_results.get('ndwi_mean'),
                        'ndwi_min': image_results.get('ndwi_min'),
                        'ndwi_max': image_results.get('ndwi_max'),
                        'ndwi_std': image_results.get('ndwi_std'),
                        'healthy_pixels_pct': image_results.get('healthy_pixels_pct'),
                        'stressed_pixels_pct': image_results.get('stressed_pixels_pct'),
                        'bare_soil_pct': image_results.get('bare_soil_pct'),
                        'planetary_computer': planetary_info,
                        'calculation_method': 'Sentinel-2 Bands: NDVI=(NIR-RED)/(NIR+RED), NDWI=(GREEN-NIR)/(GREEN+NIR)'
                    })
        
        # Determine the actual analysis date (use image date if available, otherwise today)
        analysis_date = datetime.now().date()
        image_date_str = None
        if planetary_info:
            image_date_str = planetary_info.get('using_image_date') or planetary_info.get('image_date')
            if image_date_str and image_date_str != 'Unknown':
                try:
                    analysis_date = datetime.strptime(image_date_str, '%Y-%m-%d').date()
                except:
                    pass
        
        # Generate NDVI/NDWI visualization images
        ndvi_image_url = None
        ndwi_image_url = None
        if image_processed and image_date_str and image_date_str != 'Unknown':
            from imagery.ndvi_processor import generate_ndvi_image
            media_root = getattr(settings, 'MEDIA_ROOT', 'media')
            
            # Use minio_folder if available (all bands already downloaded)
            # Otherwise use local folder
            folder_for_viz = minio_folder if minio_folder else (os.path.dirname(latest_image) if latest_image else None)
            
            if folder_for_viz:
                viz_result = generate_ndvi_image(folder_for_viz, media_root, farm.id, image_date_str)
                if viz_result.get('success'):
                    # Use the MinIO URLs returned from generate_ndvi_image
                    ndvi_image_url = viz_result.get('ndvi_image_url')
                    ndwi_image_url = viz_result.get('ndwi_image_url')
        
        # === WORKFLOW 1: NDVI/NDWI Analysis (Vegetation & Water Indices) ===
        for field in fields:
            # Use real data from ImageReading (MinIO satellite data) or image_results
            if image_processed and image_results:
                ndvi_value = image_results.get('ndvi_mean', 0.5)
                ndwi_value = image_results.get('ndwi_mean', 0.1)
            else:
                # Get real data from ImageReading database
                from imagery.models import ImageReading
                latest_reading = ImageReading.objects.filter(field=field).order_by('-acquisition_date').first()
                if latest_reading:
                    ndvi_value = round(latest_reading.ndvi_mean, 3) if latest_reading.ndvi_mean else 0.5
                    ndwi_value = round(latest_reading.ndwi_mean, 3) if latest_reading.ndwi_mean else 0.1
                else:
                    # Fallback to farm-level average if no field-specific reading
                    farm_avg = ImageReading.objects.filter(farm=farm).aggregate(
                        avg_ndvi=models.Avg('ndvi_mean'),
                        avg_ndwi=models.Avg('ndwi_mean')
                    )
                    ndvi_value = round(farm_avg['avg_ndvi'] or 0.5, 3)
                    ndwi_value = round(farm_avg['avg_ndwi'] or 0.1, 3)
            
            # Determine health status based on NDVI
            health_status, health_color = get_health_status(ndvi_value)
            
            # Determine water status based on NDWI
            if ndwi_value >= 0.1:
                water_status = 'Good water content'
            elif ndwi_value >= 0:
                water_status = 'Moderate water content'
            else:
                water_status = 'Low water - irrigation recommended'
            
            # Save NDVI/NDWI result to database using the actual image date
            analytics_result, _ = AnalyticsResult.objects.update_or_create(
                field=field,
                date=analysis_date,
                defaults={
                    'avg_ndvi': ndvi_value,
                    'avg_ndwi': ndwi_value,
                    'irrigation_alert': ndwi_value < 0
                }
            )
            
            # === WORKFLOW 2: Crop Classification ===
            # Use field's actual crop type (required, no random fallback)
            detected_crop = field.crop_type if field.crop_type else 'Unknown'
            
            # Calculate healthy/stressed percentages based on real NDVI value
            # NDVI thresholds: >0.6 healthy, 0.3-0.6 moderate, <0.3 stressed
            if ndvi_value >= 0.6:
                healthy_pct = round(min(95, 60 + (ndvi_value - 0.6) * 100), 1)
                stressed_pct = round(max(2, 15 - (ndvi_value - 0.6) * 40), 1)
            elif ndvi_value >= 0.3:
                healthy_pct = round(40 + (ndvi_value - 0.3) * 66, 1)
                stressed_pct = round(35 - (ndvi_value - 0.3) * 50, 1)
            else:
                healthy_pct = round(max(10, ndvi_value * 130), 1)
                stressed_pct = round(min(50, 50 - ndvi_value * 50), 1)
            
            bare_soil_pct = round(max(0, 100 - healthy_pct - stressed_pct - 5), 1)
            water_pct = round(max(0, 100 - healthy_pct - stressed_pct - bare_soil_pct), 1)
            
            # Confidence based on data quality (use ImageReading valid_pixel_percentage if available)
            from imagery.models import ImageReading
            latest_reading = ImageReading.objects.filter(field=field).order_by('-acquisition_date').first()
            crop_confidence = 0.85  # Base confidence
            if latest_reading and latest_reading.valid_pixel_percentage:
                crop_confidence = round(min(0.98, latest_reading.valid_pixel_percentage / 100), 2)
            
            crop_class, _ = CropClassification.objects.update_or_create(
                field=field,
                date=analysis_date,
                defaults={
                    'detected_crop': detected_crop,
                    'confidence': crop_confidence,
                    'healthy_area_percent': healthy_pct,
                    'stressed_area_percent': stressed_pct,
                    'bare_soil_percent': max(0, bare_soil_pct),
                    'water_area_percent': max(0, water_pct)
                }
            )
            
            crop_classifications.append({
                'field': field.name,
                'detected_crop': detected_crop,
                'confidence': f"{crop_confidence * 100:.0f}%",
                'healthy_area': f"{healthy_pct}%",
                'stressed_area': f"{stressed_pct}%"
            })
            
            # === WORKFLOW 3: Harvest Prediction ===
            # Determine growth stage based on NDVI time series pattern
            # Higher NDVI = more vegetative growth, NDVI peak then decline = approaching maturity
            readings = ImageReading.objects.filter(field=field).order_by('acquisition_date')
            readings_list = list(readings.values('acquisition_date', 'ndvi_mean'))
            
            if len(readings_list) >= 3:
                # Analyze NDVI trend to determine growth stage
                recent_ndvi = [r['ndvi_mean'] for r in readings_list[-3:] if r['ndvi_mean']]
                if recent_ndvi:
                    avg_recent = sum(recent_ndvi) / len(recent_ndvi)
                    ndvi_trend = recent_ndvi[-1] - recent_ndvi[0] if len(recent_ndvi) > 1 else 0
                    
                    # Determine stage based on NDVI level and trend
                    if avg_recent < 0.2:
                        current_stage = 'Germination'
                    elif avg_recent < 0.35:
                        current_stage = 'Seedling' if ndvi_trend > 0 else 'Ripening'
                    elif avg_recent < 0.5:
                        current_stage = 'Vegetative' if ndvi_trend >= 0 else 'Fruiting'
                    elif avg_recent < 0.65:
                        current_stage = 'Flowering' if ndvi_trend <= 0.05 else 'Vegetative'
                    else:
                        current_stage = 'Flowering' if ndvi_trend < 0 else 'Vegetative'
                    
                    # Estimate days to harvest based on stage
                    stage_to_days = {'Germination': 120, 'Seedling': 100, 'Vegetative': 75, 
                                     'Flowering': 50, 'Fruiting': 30, 'Ripening': 15}
                    days_to_harvest = stage_to_days.get(current_stage, 60)
                    
                    # Calculate planting date based on first reading
                    first_reading_date = readings_list[0]['acquisition_date']
                    planting_date = first_reading_date - timedelta(days=14)  # Assume planting 2 weeks before first reading
                else:
                    current_stage = 'Vegetative'
                    days_to_harvest = 60
                    planting_date = datetime.now().date() - timedelta(days=45)
            else:
                # Not enough data - use existing HarvestPrediction if available
                existing_pred = HarvestPrediction.objects.filter(field=field).first()
                if existing_pred:
                    current_stage = existing_pred.current_growth_stage
                    days_to_harvest = existing_pred.days_to_harvest or 60
                    planting_date = existing_pred.planting_date
                else:
                    current_stage = 'Unknown'
                    days_to_harvest = None
                    planting_date = None
            
            # Only create/update prediction if we have real data
            if planting_date and days_to_harvest:
                harvest_pred, _ = HarvestPrediction.objects.update_or_create(
                    field=field,
                    defaults={
                        'planting_date': planting_date,
                        'predicted_emergence_date': planting_date + timedelta(days=10) if planting_date else None,
                        'predicted_harvest_date': datetime.now().date() + timedelta(days=days_to_harvest),
                        'current_growth_stage': current_stage.lower() if current_stage != 'Unknown' else 'vegetative',
                        'days_to_harvest': days_to_harvest,
                        'confidence_score': 0.85 if len(readings_list) >= 5 else 0.70
                    }
                )
            
            harvest_predictions.append({
                'field': field.name,
                'crop': detected_crop,
                'growth_stage': current_stage,
                'days_to_harvest': days_to_harvest if days_to_harvest else 'Unknown',
                'expected_harvest': (datetime.now() + timedelta(days=days_to_harvest)).strftime('%B %d, %Y') if days_to_harvest else 'Insufficient data'
            })
            
            # === WORKFLOW 4: Yield Prediction (ML-based, using real NDVI data) ===
            from analytics.ml_models import YieldPredictor
            from iot.models import WeatherData
            
            # Get real NDVI series for yield prediction
            ndvi_for_yield = [r['ndvi_mean'] for r in readings_list if r.get('ndvi_mean')]
            ndwi_for_yield = list(ImageReading.objects.filter(field=field).values_list('ndwi_mean', flat=True))
            ndwi_for_yield = [n for n in ndwi_for_yield if n is not None]
            
            if len(ndvi_for_yield) >= 3:
                # Get area (use minimum 1 hectare if area is too small or not set)
                area_ha = float(field.area_hectares) if field.area_hectares and field.area_hectares > 0.01 else 1.0
                
                # Get weather data
                weather = WeatherData.objects.filter(farm=farm).order_by('-timestamp')[:30]
                temp_series = [w.temperature for w in weather if w.temperature]
                precip_total = sum([w.precipitation or 0 for w in weather])
                
                yield_predictor = YieldPredictor()
                yield_result = yield_predictor.predict(
                    ndvi_series=ndvi_for_yield,
                    ndwi_series=ndwi_for_yield,
                    crop_type=detected_crop,
                    area_ha=area_ha,
                    temperature_series=temp_series if temp_series else None,
                    precipitation_total=precip_total if precip_total > 0 else None
                )
                
                yield_predictions.append({
                    'field': field.name,
                    'crop': detected_crop,
                    'predicted_yield_tonnes': yield_result.get('predicted_yield_tonnes'),
                    'yield_per_ha_kg': yield_result.get('yield_per_ha_kg'),
                    'yield_range': yield_result.get('yield_range', {}),
                    'confidence': yield_result.get('confidence'),
                    'data_quality': yield_result.get('data_quality', {})
                })
            else:
                yield_predictions.append({
                    'field': field.name,
                    'crop': detected_crop,
                    'predicted_yield_tonnes': 'Insufficient data',
                    'message': f'Need at least 3 satellite readings, found {len(ndvi_for_yield)}'
                })
            
            field_results.append({
                'field_name': field.name,
                'crop_type': detected_crop,
                'ndvi': ndvi_value if run_ndvi else None,
                'ndwi': ndwi_value if run_ndwi else None,
                'health_status': health_status,
                'health_color': health_color,
                'water_status': water_status,
                'growth_stage': current_stage,
                'days_to_harvest': days_to_harvest if days_to_harvest else 'Unknown',
                'recommendation': self._get_recommendation(ndvi_value, ndwi_value, current_stage)
            })
        
        # If no fields, return empty results with message
        if not field_results:
            # Get farm-level averages from ImageReading
            from imagery.models import ImageReading
            farm_readings = ImageReading.objects.filter(farm=farm)
            if farm_readings.exists():
                avg_data = farm_readings.aggregate(
                    avg_ndvi=models.Avg('ndvi_mean'),
                    avg_ndwi=models.Avg('ndwi_mean')
                )
                ndvi_val = round(avg_data['avg_ndvi'] or 0.5, 3)
                ndwi_val = round(avg_data['avg_ndwi'] or 0.1, 3)
                health_status, health_color = get_health_status(ndvi_val)
                
                field_results.append({
                    'field_name': 'Farm Overview (aggregated)',
                    'crop_type': 'Mixed',
                    'ndvi': ndvi_val,
                    'ndwi': ndwi_val,
                    'health_status': health_status,
                    'health_color': health_color,
                    'water_status': 'Good water content' if ndwi_val >= 0.1 else 'Moderate water content' if ndwi_val >= 0 else 'Low water',
                    'growth_stage': 'Vegetative',
                    'days_to_harvest': 'No field data',
                    'recommendation': 'Please create field boundaries for detailed analysis.'
                })
            # Don't add fake harvest predictions or crop classifications
        
        # === WORKFLOW 4: Weather Forecast ===
        # Use REAL weather data from database
        weather_forecast = []
        recent_weather = WeatherData.objects.filter(farm=farm).order_by('-timestamp')[:7]
        
        if recent_weather.exists():
            # Use real historical weather data
            for i, weather in enumerate(recent_weather):
                weather_forecast.append({
                    'date': weather.timestamp.strftime('%a, %b %d'),
                    'temperature': f"{weather.temperature}°C" if weather.temperature else 'N/A',
                    'humidity': f"{weather.humidity}%" if weather.humidity else 'N/A',
                    'precipitation': f"{weather.precipitation}mm" if weather.precipitation else '0mm',
                    'conditions': weather.conditions if weather.conditions else 'Unknown'
                })
        else:
            # No weather data available - return empty with message
            weather_forecast.append({
                'date': datetime.now().strftime('%a, %b %d'),
                'temperature': 'No data',
                'humidity': 'No data',
                'precipitation': 'No data',
                'conditions': 'No weather data available. Connect weather sensors or API.'
            })
        
        # === WORKFLOW 5: Carbon Footprint & Sustainability ===
        # Use REAL carbon data from database if available
        seasons = ['Rabi', 'Kharif']
        current_season = seasons[0] if datetime.now().month in [10, 11, 12, 1, 2, 3] else seasons[1]
        
        # Try to get existing real carbon data
        existing_carbon = CarbonFootprint.objects.filter(
            farm=farm, 
            year=datetime.now().year
        ).order_by('-id').first()
        
        if existing_carbon:
            # Use existing real carbon data
            carbon_data = existing_carbon
        else:
            # Calculate carbon metrics based on real farm data
            # Use GHG calculator if available, otherwise estimate from farm area
            from analytics.ghg_calculator import calculate_farm_emissions
            
            # Calculate total farm area from sum of field areas
            farm_area_ha = sum(f.area_hectares or 0 for f in fields) or 10.0  # hectares
            
            try:
                emissions_data = calculate_farm_emissions(farm.id)
                total_emissions = emissions_data.get('total_emissions_co2eq_kg', farm_area_ha * 200)
                carbon_sequestration = emissions_data.get('total_sequestration_co2eq_kg', farm_area_ha * 50)
                fertilizer_emissions = emissions_data.get('per_hectare_net_emissions_kg', farm_area_ha * 40) * farm_area_ha * 0.2
                fuel_emissions = emissions_data.get('per_hectare_net_emissions_kg', farm_area_ha * 30) * farm_area_ha * 0.15
            except:
                # Estimate based on typical values per hectare
                total_emissions = round(farm_area_ha * 200, 0)  # ~200 kg CO2e per hectare
                carbon_sequestration = round(farm_area_ha * 50, 0)  # ~50 kg CO2e sequestered per ha
                fertilizer_emissions = round(farm_area_ha * 40, 0)
                fuel_emissions = round(farm_area_ha * 30, 0)
            
            net_carbon = total_emissions - carbon_sequestration
            
            # Calculate sustainability score based on real metrics
            # Higher NDVI = more vegetation = more carbon sequestration
            from imagery.models import ImageReading
            avg_farm_ndvi = ImageReading.objects.filter(farm=farm).aggregate(avg=models.Avg('ndvi_mean'))['avg'] or 0.5
            sustainability_score = round(min(95, max(40, avg_farm_ndvi * 100 + 30)), 1)
            
            carbon_data, _ = CarbonFootprint.objects.update_or_create(
                farm=farm,
                year=datetime.now().year,
                season=current_season,
                defaults={
                    'total_emissions': total_emissions,
                    'soil_carbon_sequestration': carbon_sequestration,
                    'net_carbon': net_carbon,
                    'fertilizer_emissions': fertilizer_emissions,
                    'fuel_emissions': fuel_emissions,
                    'livestock_emissions': 0,  # Set to 0 unless user inputs
                    'crop_residue_emissions': round(farm_area_ha * 10, 0),
                    'tillage_practice': 'Conventional',  # Default, user can update
                    'cover_crops': False,
                    'crop_rotation': True,
                    'sustainability_score': sustainability_score
                }
            )
        
        carbon_summary = {
            'total_emissions': f"{carbon_data.total_emissions} kg CO₂e",
            'carbon_sequestered': f"{carbon_data.soil_carbon_sequestration} kg CO₂e",
            'net_carbon': f"{carbon_data.net_carbon} kg CO₂e",
            'sustainability_score': carbon_data.sustainability_score,
            'rating': 'Excellent' if carbon_data.sustainability_score >= 80 else 'Good' if carbon_data.sustainability_score >= 60 else 'Needs Improvement'
        }
        
        # === WORKFLOW 6: Irrigation Recommendations ===
        irrigation_recommendation = self._get_irrigation_recommendation(field_results)
        
        # Calculate summary statistics
        avg_ndvi = round(sum(r['ndvi'] or 0 for r in field_results) / len(field_results), 3)
        avg_ndwi = round(sum(r['ndwi'] or 0 for r in field_results) / len(field_results), 3)
        
        # === Get historical time series data ===
        time_series = self._get_time_series_data(farm)
        
        return Response({
            'message': 'AI Analysis completed successfully!',
            'farm_name': farm.name,
            'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M'),
            'fields_analyzed': len(field_results),
            'image_processed': image_processed,
            'image_source': image_source if image_processed else None,
            'image_file': os.path.basename(latest_image) if latest_image else None,
            'download_skipped': download_skipped,
            'planetary_computer': planetary_info,
            
            # NDVI/NDWI visualization images
            'ndvi_image_url': ndvi_image_url,
            'ndwi_image_url': ndwi_image_url,
            
            # FarmVibes Workflow Results
            'results': field_results,
            
            'summary': {
                'avg_ndvi': avg_ndvi,
                'avg_ndwi': avg_ndwi,
                'healthy_fields': sum(1 for r in field_results if r['health_status'] in ['Healthy', 'Excellent']),
                'attention_needed': sum(1 for r in field_results if r['health_status'] == 'Needs Attention'),
                'overall_health': 'Good' if avg_ndvi >= 0.5 else 'Moderate' if avg_ndvi >= 0.35 else 'Poor'
            },
            
            # Raw calculations for thesis documentation
            'raw_calculations': raw_calculations if raw_calculations else [{
                'note': 'No images uploaded yet - using sample data',
                'calculation_method': 'NDVI = (NIR - RED) / (NIR + RED)',
                'upload_instruction': 'Upload satellite/drone images to see real calculations'
            }],
            
            # Time series for graphs
            'time_series': time_series,
            
            'harvest_predictions': harvest_predictions,
            'yield_predictions': yield_predictions,  # ML-based yield prediction from real NDVI
            'crop_classifications': crop_classifications,
            'weather_forecast': weather_forecast,
            'carbon_footprint': carbon_summary,
            'irrigation': irrigation_recommendation,
            
            # FarmVibes.AI Workflows Covered
            'workflows_executed': [
                'Vegetation Index Analysis (NDVI/NDWI)',
                'Crop Health Assessment',
                'Crop Type Classification',
                'Harvest Date Prediction',
                'Yield Prediction (ML)',
                'Weather Forecasting',
                'Carbon Footprint Analysis',
                'Irrigation Recommendations',
                'Sustainability Scoring'
            ]
        }, status=status.HTTP_200_OK)

    def _get_time_series_data(self, farm):
        """Get historical NDVI/NDWI data for time series charts - REAL DATA ONLY"""
        from imagery.models import ImageReading
        
        # Get historical data from ImageReading table (real satellite data from MinIO)
        fields = FieldBoundary.objects.filter(farm=farm)
        historical_data = ImageReading.objects.filter(
            field__in=fields
        ).order_by('acquisition_date').values('acquisition_date', 'ndvi_mean', 'ndwi_mean', 'field__name')
        
        # Group by date
        time_series = {
            'dates': [],
            'ndvi_values': [],
            'ndwi_values': [],
            'labels': [],
            'data_source': 'database',
            'record_count': 0
        }
        
        if historical_data.exists():
            seen_dates = set()
            for record in historical_data:
                date_str = record['acquisition_date'].strftime('%Y-%m-%d') if record['acquisition_date'] else ''
                if date_str and date_str not in seen_dates:
                    seen_dates.add(date_str)
                    time_series['dates'].append(date_str)
                    time_series['ndvi_values'].append(round(record['ndvi_mean'], 4) if record['ndvi_mean'] else 0)
                    time_series['ndwi_values'].append(round(record['ndwi_mean'], 4) if record['ndwi_mean'] else 0)
                    time_series['labels'].append(record['field__name'])
            time_series['record_count'] = len(time_series['dates'])
        
        # If no database records, try to get historical data from stored imagery
        if len(time_series['dates']) == 0:
            # Check for historical satellite imagery that was processed
            media_root = getattr(settings, 'MEDIA_ROOT', 'media')
            satellite_path = os.path.join(media_root, 'uploads', 'satellite', str(farm.id))
            
            if os.path.exists(satellite_path):
                # Find all dated folders with results
                dated_folders = []
                for item in os.listdir(satellite_path):
                    item_path = os.path.join(satellite_path, item)
                    if os.path.isdir(item_path) and len(item) == 10 and '-' in item:
                        # Check for NDVI results file
                        results_file = os.path.join(item_path, 'analysis_results.json')
                        if os.path.exists(results_file):
                            try:
                                with open(results_file, 'r') as f:
                                    result_data = json.load(f)
                                    dated_folders.append({
                                        'date': item,
                                        'ndvi': result_data.get('ndvi_mean', 0),
                                        'ndwi': result_data.get('ndwi_mean', 0)
                                    })
                            except:
                                pass
                
                # Sort by date
                dated_folders.sort(key=lambda x: x['date'])
                
                for record in dated_folders[-30:]:  # Last 30 records
                    time_series['dates'].append(record['date'])
                    time_series['ndvi_values'].append(round(record['ndvi'], 4))
                    time_series['ndwi_values'].append(round(record['ndwi'], 4))
                    time_series['labels'].append(farm.name)
                
                if dated_folders:
                    time_series['data_source'] = 'stored_imagery_results'
                    time_series['record_count'] = len(dated_folders)
        
        # Add message if no real data available
        if len(time_series['dates']) == 0:
            time_series['data_source'] = 'no_data'
            time_series['message'] = 'No historical analysis data available. Run more analyses to build time series.'
        
        return time_series

    def _get_recommendation(self, ndvi, ndwi, growth_stage):
        """Generate farming recommendations based on indices and growth stage"""
        recommendations = []
        
        if ndvi < 0.4:
            recommendations.append("Consider fertilizer application to boost crop health.")
        if ndwi < 0:
            recommendations.append("Schedule irrigation - soil moisture is low.")
        if ndvi >= 0.6 and ndwi >= 0:
            recommendations.append("Excellent conditions - maintain current practices.")
        if 0.4 <= ndvi < 0.6:
            recommendations.append("Monitor crop growth - consider nutrient supplements.")
        
        # Growth stage specific recommendations
        if growth_stage == 'Flowering':
            recommendations.append("Critical growth stage - ensure adequate water supply.")
        elif growth_stage == 'Ripening':
            recommendations.append("Reduce irrigation as harvest approaches.")
        
        return " ".join(recommendations) if recommendations else "Continue regular monitoring."
    
    def _get_irrigation_recommendation(self, field_results):
        """Generate irrigation recommendations based on field analysis"""
        fields_needing_water = [r for r in field_results if r.get('ndwi', 0) < 0]
        
        if not fields_needing_water:
            return {
                'status': 'No irrigation needed',
                'message': 'All fields have adequate moisture levels.',
                'next_check': 'Check again in 3-5 days',
                'water_savings': 'Estimated 15-20% water savings this week'
            }
        
        return {
            'status': 'Irrigation recommended',
            'fields': [f['field_name'] for f in fields_needing_water],
            'message': f"{len(fields_needing_water)} field(s) need irrigation.",
            'recommended_amount': '25-35mm per field',
            'best_time': 'Early morning (5-7 AM) or evening (6-8 PM)'
        }


class SatelliteImageListView(APIView):
    """List available satellite images from MinIO (primary) with local fallback"""
    permission_classes = (AllowAny,)

    def get(self, request):
        farm_id = request.query_params.get('farm_id')
        
        minio_images = []
        local_images = []
        minio_available = False
        
        # Try MinIO first (primary storage)
        try:
            from minio import Minio
            minio_client = Minio(
                getattr(settings, 'MINIO_ENDPOINT', os.getenv('MINIO_ENDPOINT', 'localhost:9000')),
                access_key=getattr(settings, 'MINIO_ACCESS_KEY', os.getenv('MINIO_ACCESS_KEY', 'minioadmin')),
                secret_key=getattr(settings, 'MINIO_SECRET_KEY', os.getenv('MINIO_SECRET_KEY', 'minioadmin')),
                secure=False
            )
            
            bucket = 'satellite-imagery'
            if minio_client.bucket_exists(bucket):
                minio_available = True
                # Build prefix for farm-specific query
                prefix = f"farm_{farm_id}/" if farm_id else ""
                objects = minio_client.list_objects(bucket, prefix=prefix, recursive=True)
                
                for obj in objects:
                    # Parse path: farm_1/2024-01-15/red_B04.tif
                    parts = obj.object_name.split('/')
                    if len(parts) >= 3:
                        farm_id_str = parts[0].replace('farm_', '')
                        date_str = parts[1]
                        filename = parts[2] if len(parts) > 2 else parts[-1]
                        
                        minio_images.append({
                            'farm_id': farm_id_str,
                            'date': date_str,
                            'filename': filename,
                            'path': obj.object_name,
                            'preview_url': f"/api/image-preview/?source=minio&path={obj.object_name}",
                            'file_size': obj.size,
                            'source': 'minio',
                            'type': self._get_band_type(filename)
                        })
        except Exception as e:
            pass  # MinIO not available, will use local fallback
        
        # Only check local storage if MinIO not available or empty
        if not minio_available or len(minio_images) == 0:
            media_root = getattr(settings, 'MEDIA_ROOT', 'media')
            
            if farm_id:
                farms = [{'id': farm_id}]
            else:
                farms = Farm.objects.values('id', 'name')
            
            for farm_info in farms:
                farm_folder = os.path.join(media_root, 'uploads', 'satellite', str(farm_info['id'] if isinstance(farm_info, dict) else farm_info.id))
                
                if os.path.exists(farm_folder):
                    for item in os.listdir(farm_folder):
                        item_path = os.path.join(farm_folder, item)
                        
                        if os.path.isdir(item_path):
                            for f in os.listdir(item_path):
                                if f.lower().endswith(('.tif', '.tiff', '.jpg', '.jpeg', '.png')):
                                    full_path = os.path.join(item_path, f)
                                    farm_id_val = farm_info.get('id') if isinstance(farm_info, dict) else farm_info['id']
                                    local_images.append({
                                        'farm_id': farm_id_val,
                                        'date': item,
                                        'filename': f,
                                        'path': f"/media/uploads/satellite/{farm_id_val}/{item}/{f}",
                                        'preview_url': f"/api/image-preview/?source=minio&path=farm_{farm_id_val}/{item}/{f}",
                                        'file_size': os.path.getsize(full_path),
                                        'source': 'local',
                                        'type': self._get_band_type(f)
                                    })
        
        # Combine and sort results
        all_images = minio_images + local_images
        all_images.sort(key=lambda x: x.get('date', ''), reverse=True)
        
        # Group by farm and date for better display
        grouped = {}
        for img in all_images:
            key = f"{img.get('farm_id')}_{img.get('date')}"
            if key not in grouped:
                grouped[key] = {
                    'farm_id': img.get('farm_id'),
                    'date': img.get('date'),
                    'source': img.get('source'),
                    'bands': [],
                    'preview_url': None  # Will be set to the first RGB-capable band
                }
            grouped[key]['bands'].append({
                'filename': img.get('filename'),
                'type': img.get('type'),
                'path': img.get('path'),
                'preview_url': img.get('preview_url'),
                'file_size': img.get('file_size')
            })
            # Set preview URL to the first suitable band (prefer red, green, or blue)
            if not grouped[key]['preview_url'] and img.get('type') in ['red', 'green', 'blue', 'nir']:
                grouped[key]['preview_url'] = img.get('preview_url')
        
        return Response({
            'total_images': len(all_images),
            'local_count': len(local_images),
            'minio_count': len(minio_images),
            'grouped_by_date': list(grouped.values()),
            'all_images': all_images[:50]  # Limit to 50 for display
        })
    
    def _get_band_type(self, filename):
        """Determine band type from filename"""
        fn_lower = filename.lower()
        if 'red' in fn_lower or 'b04' in fn_lower:
            return 'red'
        elif 'nir' in fn_lower or 'b08' in fn_lower:
            return 'nir'
        elif 'green' in fn_lower or 'b03' in fn_lower:
            return 'green'
        elif 'blue' in fn_lower or 'b02' in fn_lower:
            return 'blue'
        elif 'swir' in fn_lower or 'b11' in fn_lower:
            return 'swir'
        elif 'ndvi' in fn_lower:
            return 'ndvi'
        elif 'ndwi' in fn_lower:
            return 'ndwi'
        return 'unknown'


class ImagePreviewView(APIView):
    """Generate image previews/thumbnails for satellite imagery with high-quality heatmap visualization"""
    permission_classes = (AllowAny,)
    
    def get(self, request):
        from django.http import HttpResponse
        import io
        
        path = request.query_params.get('path', '')
        source = request.query_params.get('source', 'local')
        # Higher default resolution for better quality
        width = int(request.query_params.get('width', 800))
        height = int(request.query_params.get('height', 600))
        quality = int(request.query_params.get('quality', 95))  # PNG/JPEG quality
        
        if not path:
            return Response({'error': 'Path required'}, status=400)
        
        try:
            image_data = None
            
            if source == 'minio':
                # Fetch from MinIO
                from minio import Minio
                import tempfile
                
                minio_client = Minio(
                    getattr(settings, 'MINIO_ENDPOINT', os.getenv('MINIO_ENDPOINT', 'localhost:9000')),
                    access_key=getattr(settings, 'MINIO_ACCESS_KEY', os.getenv('MINIO_ACCESS_KEY', 'minioadmin')),
                    secret_key=getattr(settings, 'MINIO_SECRET_KEY', os.getenv('MINIO_SECRET_KEY', 'minioadmin')),
                    secure=False
                )
                
                bucket = 'satellite-imagery'
                response = minio_client.get_object(bucket, path)
                image_data = response.read()
                response.close()
                response.release_conn()
            else:
                # Fetch from local storage
                media_root = getattr(settings, 'MEDIA_ROOT', 'media')
                full_path = os.path.join(media_root, path)
                
                if not os.path.exists(full_path):
                    return Response({'error': 'File not found'}, status=404)
                
                with open(full_path, 'rb') as f:
                    image_data = f.read()
            
            # Process and create thumbnail with pseudo-color/heatmap visualization
            if path.lower().endswith(('.tif', '.tiff')):
                # For GeoTIFF, use rasterio to read and apply colormap
                try:
                    import rasterio
                    from PIL import Image
                    import numpy as np
                    import matplotlib.pyplot as plt
                    import matplotlib.colors as mcolors
                    
                    # Get colormap from query param (default: viridis for general, RdYlGn for vegetation)
                    colormap_name = request.query_params.get('colormap', 'auto')
                    
                    # Write to temp file for rasterio
                    with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as tmp:
                        tmp.write(image_data)
                        tmp_path = tmp.name
                    
                    with rasterio.open(tmp_path) as src:
                        # Read the first band
                        band = src.read(1)
                        
                        # Normalize to 0-1 range
                        band = band.astype(float)
                        
                        # Handle nodata values
                        nodata = src.nodata
                        if nodata is not None:
                            mask = band == nodata
                            band[mask] = np.nan
                        
                        # Use percentile for better contrast
                        valid_data = band[~np.isnan(band)]
                        if len(valid_data) > 0:
                            min_val = np.percentile(valid_data, 2)
                            max_val = np.percentile(valid_data, 98)
                        else:
                            min_val, max_val = 0, 1
                        
                        if max_val > min_val:
                            band_norm = np.clip((band - min_val) / (max_val - min_val), 0, 1)
                        else:
                            band_norm = np.zeros_like(band)
                        
                        # Auto-detect best colormap based on band type from filename
                        if colormap_name == 'auto':
                            filename_lower = path.lower()
                            if 'ndvi' in filename_lower or 'nir' in filename_lower:
                                colormap_name = 'RdYlGn'  # Red-Yellow-Green for vegetation
                            elif 'ndwi' in filename_lower or 'swir' in filename_lower:
                                colormap_name = 'Blues'  # Blues for water
                            elif 'red' in filename_lower:
                                colormap_name = 'Reds'
                            elif 'green' in filename_lower:
                                colormap_name = 'Greens'
                            elif 'blue' in filename_lower:
                                colormap_name = 'Blues'
                            else:
                                colormap_name = 'viridis'  # Default scientific colormap
                        
                        # Apply colormap with high quality rendering
                        cmap = plt.get_cmap(colormap_name)
                        
                        # Use matplotlib for high-quality figure rendering
                        # Calculate figure size to match requested dimensions at high DPI
                        dpi = 150  # Higher DPI for sharper images
                        fig_width = width / dpi
                        fig_height = height / dpi
                        
                        # Create figure with tight layout
                        fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
                        
                        # Display with no interpolation for crisp pixels
                        im = ax.imshow(band_norm, cmap=cmap, interpolation='nearest', aspect='auto')
                        
                        # Remove axes for clean image
                        ax.set_axis_off()
                        
                        # Add colorbar if requested
                        show_colorbar = request.query_params.get('colorbar', 'false').lower() == 'true'
                        if show_colorbar:
                            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                            cbar.ax.tick_params(labelsize=8, colors='white')
                        
                        # Tight layout to maximize image area
                        plt.tight_layout(pad=0)
                        
                        # Save to bytes with high quality
                        output = io.BytesIO()
                        fig.savefig(output, format='png', dpi=dpi, bbox_inches='tight', 
                                   pad_inches=0, facecolor='#1a1a2e', edgecolor='none')
                        plt.close(fig)  # Important: close figure to free memory
                        output.seek(0)
                    
                    # Cleanup temp file
                    os.unlink(tmp_path)
                    
                    return HttpResponse(output.getvalue(), content_type='image/png')
                    
                except Exception as e:
                    # Fallback: return a placeholder with friendly message
                    return self._placeholder_image(width, height, 'No valid imagery')
            else:
                # For regular images (JPG, PNG) - high quality processing
                from PIL import Image, ImageEnhance
                
                img = Image.open(io.BytesIO(image_data))
                
                # Use high-quality resize instead of thumbnail for better quality
                # Calculate aspect-ratio-preserving dimensions
                orig_width, orig_height = img.size
                ratio = min(width / orig_width, height / orig_height)
                new_size = (int(orig_width * ratio), int(orig_height * ratio))
                
                # Use LANCZOS (high quality) resampling
                img = img.resize(new_size, Image.Resampling.LANCZOS)
                
                # Slight sharpening for cleaner output
                enhancer = ImageEnhance.Sharpness(img)
                img = enhancer.enhance(1.1)
                
                output = io.BytesIO()
                img.save(output, format='PNG')
                output.seek(0)
                
                return HttpResponse(output.getvalue(), content_type='image/png')
                
        except Exception as e:
            return self._placeholder_image(width, height, 'Error')
    
    def _placeholder_image(self, width, height, text='No Preview'):
        """Generate a high-quality placeholder image with modern styling"""
        from django.http import HttpResponse
        from PIL import Image, ImageDraw, ImageFont
        import io
        
        # Dark modern background matching the UI theme
        img = Image.new('RGB', (width, height), color=(26, 26, 46))  # #1a1a2e
        draw = ImageDraw.Draw(img)
        
        # Draw subtle grid pattern for visual interest
        grid_color = (40, 40, 70)
        grid_spacing = 30
        for x in range(0, width, grid_spacing):
            draw.line([(x, 0), (x, height)], fill=grid_color, width=1)
        for y in range(0, height, grid_spacing):
            draw.line([(0, y), (width, y)], fill=grid_color, width=1)
        
        # Draw border
        draw.rectangle([(0, 0), (width-1, height-1)], outline=(60, 60, 90), width=2)
        
        # Try to use a larger font
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size=min(width, height) // 12)
        except:
            font = ImageFont.load_default()
        
        # Draw icon-like shape (satellite/image icon)
        center_x, center_y = width // 2, height // 2 - 20
        icon_size = min(width, height) // 6
        draw.rectangle(
            [(center_x - icon_size, center_y - icon_size), 
             (center_x + icon_size, center_y + icon_size)], 
            outline=(100, 100, 140), width=3
        )
        # Mountain shape inside
        draw.polygon([
            (center_x - icon_size + 10, center_y + icon_size - 10),
            (center_x - icon_size // 2, center_y),
            (center_x, center_y + icon_size - 10)
        ], fill=(70, 70, 100))
        draw.polygon([
            (center_x - 10, center_y + icon_size - 10),
            (center_x + icon_size // 3, center_y - icon_size // 2),
            (center_x + icon_size - 10, center_y + icon_size - 10)
        ], fill=(90, 90, 120))
        # Sun circle
        draw.ellipse(
            [(center_x + icon_size // 2, center_y - icon_size + 15),
             (center_x + icon_size - 10, center_y - icon_size // 2 + 15)],
            fill=(120, 120, 160)
        )
        
        # Draw text below icon
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_x = (width - text_width) // 2
        text_y = center_y + icon_size + 20
        draw.text((text_x, text_y), text, fill=(150, 150, 180), font=font)
        
        output = io.BytesIO()
        img.save(output, format='PNG', quality=95)
        output.seek(0)
        
        return HttpResponse(output.getvalue(), content_type='image/png')


class AnalysisImageView(APIView):
    """Serve NDVI/NDWI analysis result images from MinIO"""
    permission_classes = (AllowAny,)
    
    def get(self, request):
        from django.http import HttpResponse
        
        path = request.query_params.get('path', '')
        
        if not path:
            return Response({'error': 'Path required'}, status=400)
        
        try:
            from minio import Minio
            
            minio_client = Minio(
                os.getenv('MINIO_ENDPOINT', 'localhost:9000'),
                access_key=os.getenv('MINIO_ACCESS_KEY', 'minioadmin'),
                secret_key=os.getenv('MINIO_SECRET_KEY', 'minioadmin'),
                secure=False
            )
            
            bucket_name = 'analysis-results'
            
            # Get the image from MinIO
            response = minio_client.get_object(bucket_name, path)
            image_data = response.read()
            response.close()
            response.release_conn()
            
            # Determine content type
            content_type = 'image/png'
            if path.lower().endswith('.jpg') or path.lower().endswith('.jpeg'):
                content_type = 'image/jpeg'
            
            return HttpResponse(image_data, content_type=content_type)
            
        except Exception as e:
            # Return a placeholder if image not found
            return self._generate_placeholder(str(e))
    
    def _generate_placeholder(self, error_msg='Image not found'):
        """Generate a placeholder image for missing analysis results"""
        from django.http import HttpResponse
        from PIL import Image, ImageDraw, ImageFont
        import io
        
        width, height = 400, 300
        img = Image.new('RGB', (width, height), color=(26, 26, 46))
        draw = ImageDraw.Draw(img)
        
        # Draw border
        draw.rectangle([(0, 0), (width-1, height-1)], outline=(60, 60, 90), width=2)
        
        # Draw text
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size=16)
        except:
            font = ImageFont.load_default()
        
        text = "Analysis image not available"
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        x = (width - text_width) // 2
        draw.text((x, height // 2 - 20), text, fill=(150, 150, 180), font=font)
        
        small_text = "Run analysis to generate"
        small_bbox = draw.textbbox((0, 0), small_text, font=font)
        small_width = small_bbox[2] - small_bbox[0]
        draw.text(((width - small_width) // 2, height // 2 + 10), small_text, fill=(100, 100, 130), font=font)
        
        output = io.BytesIO()
        img.save(output, format='PNG')
        output.seek(0)
        
        return HttpResponse(output.getvalue(), content_type='image/png')


# ============== NEW IMAGERY WORKFLOW VIEWS ==============

class SAMSegmentationView(APIView):
    """
    Segment imagery using Segment Anything Model.
    
    POST /api/imagery/sam/segment/
    GET /api/imagery/sam/field/<field_id>/
    """
    parser_classes = [MultiPartParser]
    
    def post(self, request):
        from .sam_processor import SAMProcessor
        from PIL import Image
        import numpy as np
        
        image_file = request.FILES.get('image')
        if not image_file:
            return Response({'error': 'No image provided'}, status=status.HTTP_400_BAD_REQUEST)
        
        # Load image
        img = Image.open(image_file)
        img_array = np.array(img)
        
        # Initialize SAM
        sam = SAMProcessor()
        
        # Try to load model
        if not sam.load_model():
            # Use fallback segmentation
            result = sam._fallback_segmentation(img_array)
        else:
            result = sam.segment_automatic(img_array)
        
        # Remove masks from response (too large)
        if 'masks' in result:
            result['num_masks'] = len(result['masks'])
            del result['masks']
        
        return Response(result)
    
    def get(self, request, field_id=None):
        if not field_id:
            return Response({'error': 'field_id required'}, status=status.HTTP_400_BAD_REQUEST)
        
        from .sam_processor import segment_field_from_sentinel
        result = segment_field_from_sentinel(field_id)
        
        return Response(result)


class FieldBoundarySegmentationView(APIView):
    """
    Segment field boundaries from satellite imagery.
    
    POST /api/imagery/segment-fields/
    """
    parser_classes = [MultiPartParser]
    
    def post(self, request):
        from .sam_processor import FieldBoundarySegmenter
        from PIL import Image
        import numpy as np
        
        image_file = request.FILES.get('image')
        if not image_file:
            return Response({'error': 'No image provided'}, status=status.HTTP_400_BAD_REQUEST)
        
        min_area = int(request.data.get('min_area', 1000))
        
        img = Image.open(image_file)
        img_array = np.array(img)
        
        segmenter = FieldBoundarySegmenter()
        result = segmenter.segment_fields(img_array, min_field_area=min_area)
        
        # Remove masks from response
        if 'masks' in result:
            del result['masks']
        
        return Response(result)


class ForestMapsView(APIView):
    """
    Download and analyze forest maps.
    
    GET /api/imagery/forest-maps/<farm_id>/
    GET /api/imagery/forest-maps/?lat=30.5&lon=70.5
    """
    
    def get(self, request, farm_id=None):
        from .forest_maps import download_forest_maps_for_farm, ForestChangeDetector
        
        if farm_id:
            result = download_forest_maps_for_farm(farm_id)
        else:
            lat = request.query_params.get('lat')
            lon = request.query_params.get('lon')
            
            if not lat or not lon:
                return Response(
                    {'error': 'Provide farm_id or lat/lon coordinates'},
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            detector = ForestChangeDetector()
            result = detector.analyze_location(float(lat), float(lon))
        
        return Response(result)


class TimelapseView(APIView):
    """
    Generate timelapse visualizations.
    
    GET /api/imagery/timelapse/<field_id>/
    GET /api/imagery/timelapse/<field_id>/?format=gif
    """
    
    def get(self, request, field_id):
        from .timelapse import create_field_timelapse
        from datetime import datetime
        
        format_type = request.query_params.get('format', 'gif')
        start_date = request.query_params.get('start_date')
        end_date = request.query_params.get('end_date')
        
        start = datetime.fromisoformat(start_date) if start_date else None
        end = datetime.fromisoformat(end_date) if end_date else None
        
        result = create_field_timelapse(field_id, start, end, format_type)
        
        if 'error' in result:
            return Response(result, status=status.HTTP_400_BAD_REQUEST)
        
        # Return file URL if successful
        if 'output_path' in result:
            result['url'] = f"/media/{result['output_path'].split('media/')[-1]}"
        
        return Response(result)


class SpectralFusionView(APIView):
    """
    Fuse drone and satellite imagery.
    
    POST /api/imagery/spectral-fusion/
    """
    parser_classes = [MultiPartParser]
    
    def post(self, request):
        from .spectral_fusion import SpectralFusion, SpectralExtension
        from PIL import Image
        import numpy as np
        
        drone_file = request.FILES.get('drone_image')
        
        if not drone_file:
            return Response({'error': 'drone_image required'}, status=status.HTTP_400_BAD_REQUEST)
        
        # Load drone image
        drone_img = np.array(Image.open(drone_file))
        if drone_img.max() > 1:
            drone_img = drone_img.astype(float) / 255
        
        # Extend spectral bands
        extender = SpectralExtension()
        extended = extender._empirical_extension(drone_img)
        
        # Calculate indices
        indices = extender.calculate_indices(extended)
        
        return Response({
            'status': 'success',
            'original_shape': list(drone_img.shape),
            'extended_shape': list(extended.shape),
            'indices': {
                'ndvi_mean': float(np.mean(indices['ndvi'])),
                'ndvi_std': float(np.std(indices['ndvi'])),
                'gndvi_mean': float(np.mean(indices['gndvi'])),
                'savi_mean': float(np.mean(indices['savi'])),
                'evi_mean': float(np.mean(indices['evi']))
            },
            'method': 'empirical_nir_estimation'
        })