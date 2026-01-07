"""
Management command to ingest satellite imagery from MinIO and create ImageReading records.
This processes REAL data only - NO synthetic/fake data generation.

Usage:
    python manage.py ingest_satellite_imagery
    python manage.py ingest_satellite_imagery --farm-id 1
    python manage.py ingest_satellite_imagery --date 2025-12-02
"""

import os
import tempfile
import numpy as np
from datetime import datetime, date

from django.core.management.base import BaseCommand
from django.db import transaction

from core.models import Farm, FieldBoundary
from imagery.models import ImageReading

try:
    from minio import Minio
    HAS_MINIO = True
except ImportError:
    HAS_MINIO = False

try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False


class Command(BaseCommand):
    help = 'Ingest satellite imagery from MinIO and create ImageReading records using REAL data only'

    def add_arguments(self, parser):
        parser.add_argument(
            '--farm-id',
            type=int,
            help='Process only a specific farm ID',
        )
        parser.add_argument(
            '--date',
            type=str,
            help='Process only a specific date (YYYY-MM-DD format)',
        )
        parser.add_argument(
            '--force',
            action='store_true',
            help='Force re-processing even if ImageReading already exists',
        )

    def handle(self, *args, **options):
        if not HAS_MINIO:
            self.stderr.write(self.style.ERROR('minio package not installed. Run: pip install minio'))
            return
        
        if not HAS_RASTERIO:
            self.stderr.write(self.style.ERROR('rasterio package not installed. Run: pip install rasterio'))
            return

        # Connect to MinIO
        try:
            minio_client = Minio(
                os.getenv('MINIO_ENDPOINT', 'localhost:9000'),
                access_key=os.getenv('MINIO_ACCESS_KEY', 'minioadmin'),
                secret_key=os.getenv('MINIO_SECRET_KEY', 'minioadmin'),
                secure=False
            )
            self.stdout.write(self.style.SUCCESS('Connected to MinIO'))
        except Exception as e:
            self.stderr.write(self.style.ERROR(f'Failed to connect to MinIO: {e}'))
            return

        bucket = 'satellite-imagery'
        
        # Check if bucket exists
        if not minio_client.bucket_exists(bucket):
            self.stderr.write(self.style.ERROR(f'Bucket {bucket} does not exist'))
            return

        # List all objects to find available farm/date combinations
        self.stdout.write('Scanning MinIO for satellite imagery...')
        
        farm_dates = self._discover_farm_dates(minio_client, bucket, options)
        
        if not farm_dates:
            self.stderr.write(self.style.WARNING('No satellite imagery found in MinIO'))
            return
        
        self.stdout.write(f'Found {len(farm_dates)} farm/date combinations to process')
        
        # Process each farm/date combination
        total_created = 0
        total_skipped = 0
        total_errors = 0
        
        for farm_id, image_date in farm_dates:
            try:
                created = self._process_farm_date(
                    minio_client, bucket, farm_id, image_date, 
                    force=options.get('force', False)
                )
                if created:
                    total_created += 1
                else:
                    total_skipped += 1
            except Exception as e:
                self.stderr.write(self.style.ERROR(f'Error processing farm_{farm_id}/{image_date}: {e}'))
                total_errors += 1
        
        self.stdout.write(self.style.SUCCESS(
            f'\nIngestion complete: {total_created} created, {total_skipped} skipped, {total_errors} errors'
        ))

    def _discover_farm_dates(self, minio_client, bucket, options):
        """Discover all farm/date combinations in MinIO bucket"""
        farm_dates = set()
        
        target_farm_id = options.get('farm_id')
        target_date = options.get('date')
        
        # List all objects
        objects = minio_client.list_objects(bucket, recursive=True)
        
        for obj in objects:
            # Parse path like "farm_1/2025-12-02/B04.tif"
            parts = obj.object_name.split('/')
            if len(parts) >= 2:
                farm_part = parts[0]  # "farm_1"
                date_part = parts[1]  # "2025-12-02"
                
                # Extract farm ID
                if farm_part.startswith('farm_'):
                    try:
                        farm_id = int(farm_part.split('_')[1])
                    except (ValueError, IndexError):
                        continue
                    
                    # Parse date
                    try:
                        image_date = datetime.strptime(date_part, '%Y-%m-%d').date()
                    except ValueError:
                        continue
                    
                    # Apply filters
                    if target_farm_id and farm_id != target_farm_id:
                        continue
                    if target_date and date_part != target_date:
                        continue
                    
                    farm_dates.add((farm_id, image_date))
        
        return sorted(farm_dates)

    def _process_farm_date(self, minio_client, bucket, farm_id, image_date, force=False):
        """Process a single farm/date combination and create ImageReading records"""
        
        self.stdout.write(f'Processing farm_{farm_id}/{image_date}...')
        
        # Get farm and fields
        try:
            farm = Farm.objects.get(id=farm_id)
        except Farm.DoesNotExist:
            self.stderr.write(self.style.WARNING(f'  Farm {farm_id} not found in database, skipping'))
            return False
        
        fields = FieldBoundary.objects.filter(farm=farm)
        if not fields.exists():
            self.stderr.write(self.style.WARNING(f'  No fields found for farm {farm_id}, skipping'))
            return False
        
        # Check if already processed (for any field)
        if not force:
            existing = ImageReading.objects.filter(farm=farm, acquisition_date=image_date).exists()
            if existing:
                self.stdout.write(f'  Already processed, skipping (use --force to re-process)')
                return False
        
        # Download bands from MinIO
        prefix = f"farm_{farm_id}/{image_date}/"
        objects = list(minio_client.list_objects(bucket, prefix=prefix))
        
        if not objects:
            self.stderr.write(self.style.WARNING(f'  No files found at {prefix}'))
            return False
        
        # Organize bands
        band_files = {}
        for obj in objects:
            filename = os.path.basename(obj.object_name).upper()
            
            # Detect band from filename
            for band_name in ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B10', 'B11', 'B12', 'SCL']:
                if band_name in filename:
                    band_files[band_name] = obj.object_name
                    break
        
        self.stdout.write(f'  Found bands: {", ".join(sorted(band_files.keys()))}')
        
        # Required bands for NDVI (B04=Red, B08=NIR) and NDWI (B03=Green, B08=NIR)
        if 'B04' not in band_files or 'B08' not in band_files:
            self.stderr.write(self.style.WARNING(f'  Missing required bands B04/B08, skipping'))
            return False
        
        # Download and process bands
        band_data = {}
        band_stats = {}
        
        with tempfile.TemporaryDirectory() as tmpdir:
            for band_name, obj_path in band_files.items():
                local_path = os.path.join(tmpdir, f'{band_name}.tif')
                
                try:
                    minio_client.fget_object(bucket, obj_path, local_path)
                    
                    # Read band data with rasterio
                    with rasterio.open(local_path) as src:
                        data = src.read(1).astype(float)
                        
                        # Store raw data for B03, B04, B08 (needed for indices)
                        if band_name in ['B03', 'B04', 'B08']:
                            band_data[band_name] = data
                        
                        # Calculate statistics for all bands
                        valid_mask = ~np.isnan(data) & (data != 0)
                        if np.any(valid_mask):
                            valid_data = data[valid_mask]
                            band_stats[band_name] = {
                                'mean': float(np.mean(valid_data)),
                                'std': float(np.std(valid_data)),
                                'min': float(np.min(valid_data)),
                                'max': float(np.max(valid_data)),
                                'valid_pixels': int(np.sum(valid_mask)),
                                'total_pixels': int(data.size),
                            }
                        else:
                            band_stats[band_name] = None
                            
                except Exception as e:
                    self.stderr.write(self.style.WARNING(f'  Error reading {band_name}: {e}'))
                    continue
            
            # Calculate NDVI from REAL band data
            nir = band_data.get('B08')
            red = band_data.get('B04')
            green = band_data.get('B03')
            
            if nir is None or red is None:
                self.stderr.write(self.style.WARNING(f'  Could not read required bands'))
                return False
            
            # NDVI = (NIR - RED) / (NIR + RED)
            denominator = nir + red
            denominator[denominator == 0] = 0.0001
            ndvi = (nir - red) / denominator
            ndvi = np.clip(ndvi, -1, 1)
            
            # NDVI statistics
            valid_mask = ~np.isnan(ndvi) & np.isfinite(ndvi)
            ndvi_valid = ndvi[valid_mask]
            
            ndvi_mean = float(np.mean(ndvi_valid)) if len(ndvi_valid) > 0 else None
            ndvi_std = float(np.std(ndvi_valid)) if len(ndvi_valid) > 0 else None
            ndvi_min = float(np.min(ndvi_valid)) if len(ndvi_valid) > 0 else None
            ndvi_max = float(np.max(ndvi_valid)) if len(ndvi_valid) > 0 else None
            
            # NDWI = (GREEN - NIR) / (GREEN + NIR)
            ndwi_mean = ndwi_std = ndwi_min = ndwi_max = None
            if green is not None:
                denominator = green + nir
                denominator[denominator == 0] = 0.0001
                ndwi = (green - nir) / denominator
                ndwi = np.clip(ndwi, -1, 1)
                
                ndwi_valid = ndwi[valid_mask]
                ndwi_mean = float(np.mean(ndwi_valid)) if len(ndwi_valid) > 0 else None
                ndwi_std = float(np.std(ndwi_valid)) if len(ndwi_valid) > 0 else None
                ndwi_min = float(np.min(ndwi_valid)) if len(ndwi_valid) > 0 else None
                ndwi_max = float(np.max(ndwi_valid)) if len(ndwi_valid) > 0 else None
            
            # Calculate cloud coverage from SCL band if available
            cloud_coverage = None
            valid_pixel_pct = None
            
            if 'SCL' in band_files:
                try:
                    scl_path = os.path.join(tmpdir, 'SCL.tif')
                    with rasterio.open(scl_path) as src:
                        scl = src.read(1)
                        # SCL classes 8, 9, 10 are clouds
                        cloud_pixels = np.isin(scl, [8, 9, 10]).sum()
                        total_pixels = scl.size
                        cloud_coverage = float(cloud_pixels / total_pixels * 100)
                        
                        # Valid pixels (vegetation, soil, water)
                        valid_pixels = np.isin(scl, [4, 5, 6]).sum()
                        valid_pixel_pct = float(valid_pixels / total_pixels * 100)
                except Exception:
                    pass
            
            # Create ImageReading for each field in this farm
            created_count = 0
            
            with transaction.atomic():
                for field in fields:
                    # Delete existing if force
                    if force:
                        ImageReading.objects.filter(field=field, acquisition_date=image_date).delete()
                    
                    # Create new ImageReading with REAL data
                    reading = ImageReading.objects.create(
                        field=field,
                        farm=farm,
                        acquisition_date=image_date,
                        minio_folder=prefix,
                        ndvi_mean=ndvi_mean,
                        ndvi_std=ndvi_std,
                        ndvi_min=ndvi_min,
                        ndvi_max=ndvi_max,
                        ndwi_mean=ndwi_mean,
                        ndwi_std=ndwi_std,
                        ndwi_min=ndwi_min,
                        ndwi_max=ndwi_max,
                        band_data=band_stats,
                        cloud_coverage=cloud_coverage,
                        valid_pixel_percentage=valid_pixel_pct,
                        processing_notes=f'Ingested from MinIO on {datetime.now().isoformat()}. Bands: {", ".join(sorted(band_files.keys()))}'
                    )
                    created_count += 1
                    
                    ndwi_str = f'{ndwi_mean:.4f}' if ndwi_mean is not None else 'N/A'
                    self.stdout.write(self.style.SUCCESS(
                        f'  Created ImageReading for {field.name}: NDVI={ndvi_mean:.4f}, NDWI={ndwi_str}'
                    ))
        
        return created_count > 0
