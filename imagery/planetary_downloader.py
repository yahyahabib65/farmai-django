"""
Planetary Computer Satellite Image Downloader
Downloads Sentinel-2 imagery based on farm boundaries (AOI)
Stores images in MinIO organized by date
"""

import os
import io
import json
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple
import requests

try:
    import planetary_computer as pc
    import pystac_client
    from shapely.geometry import shape, box, mapping
    from shapely.ops import transform
    import pyproj
    HAS_PLANETARY = True
except ImportError:
    HAS_PLANETARY = False

try:
    from minio import Minio
    from minio.error import S3Error
    HAS_MINIO = True
except ImportError:
    HAS_MINIO = False

try:
    import rasterio
    from rasterio.io import MemoryFile
    from rasterio.mask import mask
    from rasterio.warp import calculate_default_transform, reproject, Resampling
    import numpy as np
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

from django.conf import settings


class PlanetaryComputerDownloader:
    """
    Downloads satellite imagery from Microsoft Planetary Computer
    using farm boundaries as Area of Interest (AOI)
    """
    
    STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
    
    # Sentinel-2 bands for NDVI calculation
    BANDS = {
        'red': 'B04',      # Red band (665nm)
        'green': 'B03',    # Green band (560nm)
        'blue': 'B02',     # Blue band (490nm)
        'nir': 'B08',      # NIR band (842nm)
        'swir': 'B11',     # SWIR band for NDWI
    }
    
    def __init__(self, minio_endpoint: str = None, minio_access_key: str = None, 
                 minio_secret_key: str = None):
        """Initialize with MinIO configuration"""
        self.minio_endpoint = minio_endpoint or os.getenv('MINIO_ENDPOINT', 'localhost:9000')
        self.minio_access_key = minio_access_key or os.getenv('MINIO_ACCESS_KEY', 'minioadmin')
        self.minio_secret_key = minio_secret_key or os.getenv('MINIO_SECRET_KEY', 'minioadmin')
        
        self.minio_client = None
        self.bucket_name = 'satellite-imagery'
        
        if HAS_MINIO:
            try:
                self.minio_client = Minio(
                    self.minio_endpoint,
                    access_key=self.minio_access_key,
                    secret_key=self.minio_secret_key,
                    secure=False
                )
                # Ensure bucket exists
                if not self.minio_client.bucket_exists(self.bucket_name):
                    self.minio_client.make_bucket(self.bucket_name)
            except Exception as e:
                print(f"MinIO connection failed: {e}")
                self.minio_client = None
    
    def get_aoi_from_boundary(self, boundary_geom) -> Dict:
        """
        Convert Django GIS boundary to GeoJSON AOI
        Simplifies geometry if needed to reduce download area
        """
        if hasattr(boundary_geom, 'geojson'):
            geojson = json.loads(boundary_geom.geojson)
        elif hasattr(boundary_geom, 'json'):
            geojson = json.loads(boundary_geom.json)
        else:
            # Assume it's already a dict
            geojson = boundary_geom
        
        return geojson
    
    def get_bounding_box(self, boundary_geom) -> Tuple[float, float, float, float]:
        """Get bounding box from geometry (minx, miny, maxx, maxy)"""
        if hasattr(boundary_geom, 'extent'):
            return boundary_geom.extent
        
        geojson = self.get_aoi_from_boundary(boundary_geom)
        geom = shape(geojson)
        return geom.bounds
    
    def search_sentinel2_images(self, boundary_geom, start_date: datetime = None, 
                                 end_date: datetime = None, max_cloud_cover: int = 30,
                                 limit: int = 5) -> List[Dict]:
        """
        Search for Sentinel-2 images covering the farm boundary
        
        Args:
            boundary_geom: Farm boundary geometry
            start_date: Start of search period (default: 30 days ago)
            end_date: End of search period (default: today)
            max_cloud_cover: Maximum cloud cover percentage
            limit: Maximum number of results
        
        Returns:
            List of matching STAC items
        """
        if not HAS_PLANETARY:
            return []
        
        # Default date range: last 30 days
        if end_date is None:
            end_date = datetime.now()
        if start_date is None:
            start_date = end_date - timedelta(days=30)
        
        bbox = self.get_bounding_box(boundary_geom)
        
        try:
            catalog = pystac_client.Client.open(self.STAC_URL)
            
            search = catalog.search(
                collections=["sentinel-2-l2a"],
                bbox=bbox,
                datetime=f"{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}",
                query={
                    "eo:cloud_cover": {"lt": max_cloud_cover}
                },
                max_items=limit
            )
            
            items = list(search.items())
            
            results = []
            for item in items:
                # Sign the item for access
                signed_item = pc.sign(item)
                results.append({
                    'id': item.id,
                    'datetime': item.datetime.isoformat() if item.datetime else None,
                    'cloud_cover': item.properties.get('eo:cloud_cover', 0),
                    'assets': {k: v.href for k, v in signed_item.assets.items()},
                    'geometry': item.geometry,
                    'bbox': item.bbox
                })
            
            return results
            
        except Exception as e:
            print(f"Planetary Computer search error: {e}")
            return []
    
    def download_and_clip_band(self, band_url: str, boundary_geom, band_name: str) -> Optional[bytes]:
        """
        Download a single band and clip to AOI
        
        Args:
            band_url: URL to the band asset
            boundary_geom: Farm boundary for clipping
            band_name: Name of the band (for logging)
        
        Returns:
            Clipped raster as bytes (GeoTIFF format)
        """
        if not HAS_RASTERIO:
            return None
        
        try:
            aoi = self.get_aoi_from_boundary(boundary_geom)
            aoi_shape = shape(aoi)
            
            with rasterio.open(band_url) as src:
                # Transform AOI to match raster CRS if different
                raster_crs = src.crs
                
                # AOI is typically in WGS84 (EPSG:4326)
                aoi_crs = pyproj.CRS.from_epsg(4326)
                
                if raster_crs and str(raster_crs) != str(aoi_crs):
                    # Need to reproject AOI to match raster CRS
                    project = pyproj.Transformer.from_crs(
                        aoi_crs, 
                        raster_crs, 
                        always_xy=True
                    ).transform
                    aoi_shape_transformed = transform(project, aoi_shape)
                else:
                    aoi_shape_transformed = aoi_shape
                
                # Clip to AOI
                out_image, out_transform = mask(src, [aoi_shape_transformed], crop=True)
                out_meta = src.meta.copy()
                
                out_meta.update({
                    "driver": "GTiff",
                    "height": out_image.shape[1],
                    "width": out_image.shape[2],
                    "transform": out_transform,
                    "compress": "lzw"
                })
                
                # Write to memory buffer
                with MemoryFile() as memfile:
                    with memfile.open(**out_meta) as dst:
                        dst.write(out_image)
                    
                    return memfile.read()
                    
        except Exception as e:
            print(f"Error downloading/clipping band {band_name}: {e}")
            return None
    
    def _get_existing_dates(self, farm_id: int) -> set:
        """
        Get set of dates that already have satellite imagery downloaded.
        Checks both local storage and MinIO.
        
        Args:
            farm_id: Farm ID to check
            
        Returns:
            Set of date strings (YYYY-MM-DD format)
        """
        existing_dates = set()
        
        # Check local storage
        local_path = os.path.join(settings.MEDIA_ROOT, 'uploads', 'satellite', str(farm_id))
        if os.path.exists(local_path):
            for item in os.listdir(local_path):
                item_path = os.path.join(local_path, item)
                # Check if it's a date folder (YYYY-MM-DD format)
                if os.path.isdir(item_path) and len(item) == 10 and '-' in item:
                    # Verify it has band files
                    tif_files = [f for f in os.listdir(item_path) if f.endswith('.tif')]
                    if tif_files:
                        existing_dates.add(item)
        
        # Check MinIO
        if self.minio_client:
            try:
                prefix = f"farm_{farm_id}/"
                objects = self.minio_client.list_objects(self.bucket_name, prefix=prefix, recursive=False)
                for obj in objects:
                    # Extract date from path like "farm_1/2025-11-27/"
                    parts = obj.object_name.split('/')
                    if len(parts) >= 2:
                        date_str = parts[1]
                        if len(date_str) == 10 and '-' in date_str:
                            existing_dates.add(date_str)
            except Exception as e:
                print(f"Error checking MinIO for existing dates: {e}")
        
        return existing_dates
    
    def download_imagery_for_farm(self, farm, field_boundary=None, 
                                   target_date: datetime = None,
                                   start_date: datetime = None,
                                   end_date: datetime = None,
                                   download_all: bool = True,
                                   max_cloud_cover: int = 30) -> Dict:
        """
        Download satellite imagery for a farm, clipped to boundary.
        Downloads ALL images within the date range when download_all=True.
        
        Args:
            farm: Farm model instance
            field_boundary: Optional specific field boundary
            target_date: Target date for imagery (default: latest available)
            start_date: Start of date range to download
            end_date: End of date range to download
            download_all: If True, download ALL images in range (not just best one)
            max_cloud_cover: Maximum cloud cover percentage (default: 30)
        
        Returns:
            Dict with download results and MinIO paths
        """
        result = {
            'success': False,
            'farm_id': farm.id,
            'farm_name': farm.name,
            'images_downloaded': 0,
            'total_bands_downloaded': 0,
            'minio_paths': [],
            'downloaded_dates': [],
            'error': None
        }
        
        # Get boundary
        if field_boundary:
            boundary = field_boundary.boundary
        else:
            # Get all field boundaries for this farm
            from core.models import FieldBoundary
            fields = FieldBoundary.objects.filter(farm=farm)
            if not fields.exists():
                result['error'] = 'No field boundaries defined for this farm'
                return result
            # Use first field boundary or union of all
            boundary = fields.first().boundary
        
        # Determine date range
        if end_date is None:
            end_date = target_date or datetime.now()
        if start_date is None:
            start_date = end_date - timedelta(days=30)
        
        # Search for ALL images in the date range (increase limit for full range)
        # Calculate days in range to estimate limit
        days_in_range = (end_date - start_date).days
        estimated_limit = max(10, days_in_range // 5)  # Sentinel-2 revisit time is ~5 days
        
        images = self.search_sentinel2_images(
            boundary,
            start_date=start_date,
            end_date=end_date,
            max_cloud_cover=max_cloud_cover,
            limit=estimated_limit
        )
        
        if not images:
            result['error'] = f'No satellite images found for the specified area and date range ({start_date.strftime("%Y-%m-%d")} to {end_date.strftime("%Y-%m-%d")})'
            return result
        
        # Sort by date (ascending) to process in chronological order
        images = sorted(images, key=lambda x: x.get('datetime', ''))
        
        # Get list of dates already downloaded locally
        existing_dates = self._get_existing_dates(farm.id)
        skipped_dates = []
        
        if download_all:
            # Download ALL images in the date range (skip existing)
            for image in images:
                image_datetime = image.get('datetime', '')
                if image_datetime:
                    try:
                        dt = datetime.fromisoformat(image_datetime.replace('Z', '+00:00'))
                        date_str = dt.strftime('%Y-%m-%d')
                        
                        # Skip if already exists locally
                        if date_str in existing_dates:
                            skipped_dates.append(date_str)
                            continue
                    except:
                        pass
                
                download_result = self._download_single_image(farm, boundary, image)
                if download_result.get('success'):
                    result['images_downloaded'] += 1
                    result['total_bands_downloaded'] += download_result.get('bands_downloaded', 0)
                    result['minio_paths'].extend(download_result.get('minio_paths', []))
                    result['downloaded_dates'].append(download_result.get('date'))
        else:
            # Download only the best (lowest cloud cover) image that doesn't exist
            for image in sorted(images, key=lambda x: x.get('cloud_cover', 100)):
                image_datetime = image.get('datetime', '')
                if image_datetime:
                    try:
                        dt = datetime.fromisoformat(image_datetime.replace('Z', '+00:00'))
                        date_str = dt.strftime('%Y-%m-%d')
                        
                        # Skip if already exists locally
                        if date_str in existing_dates:
                            skipped_dates.append(date_str)
                            continue
                    except:
                        pass
                
                download_result = self._download_single_image(farm, boundary, image)
                if download_result.get('success'):
                    result['images_downloaded'] = 1
                    result['total_bands_downloaded'] = download_result.get('bands_downloaded', 0)
                    result['minio_paths'] = download_result.get('minio_paths', [])
                    result['downloaded_dates'] = [download_result.get('date')]
                    result['cloud_cover'] = image.get('cloud_cover', 0)
                    result['image_id'] = image['id']
                    break
        
        result['success'] = result['images_downloaded'] > 0 or len(skipped_dates) > 0
        result['skipped_dates'] = skipped_dates
        result['existing_dates'] = list(existing_dates)
        result['date_range'] = {
            'start': start_date.strftime('%Y-%m-%d'),
            'end': end_date.strftime('%Y-%m-%d')
        }
        result['total_available'] = len(images)
        
        # Set image_date for backward compatibility (use first downloaded date)
        if result['downloaded_dates']:
            result['image_date'] = result['downloaded_dates'][0]
        
        return result
    
    def _download_single_image(self, farm, boundary, image: Dict) -> Dict:
        """
        Download a single Sentinel-2 image with all bands.
        
        Args:
            farm: Farm model instance
            boundary: Field boundary geometry
            image: Image metadata from STAC search
        
        Returns:
            Dict with download result for this single image
        """
        result = {
            'success': False,
            'bands_downloaded': 0,
            'minio_paths': [],
            'date': None,
            'error': None
        }
        
        try:
            image_date = datetime.fromisoformat(image['datetime'].replace('Z', '+00:00'))
            date_str = image_date.strftime('%Y-%m-%d')
            result['date'] = date_str
            
            downloaded_bands = {}
            
            # Download required bands
            for band_key, band_id in self.BANDS.items():
                if band_id in image['assets']:
                    band_url = image['assets'][band_id]
                    band_data = self.download_and_clip_band(band_url, boundary, band_key)
                    
                    if band_data:
                        downloaded_bands[band_key] = band_data
                        
                        # Store in MinIO
                        if self.minio_client:
                            minio_path = f"farm_{farm.id}/{date_str}/{band_key}_{band_id}.tif"
                            try:
                                self.minio_client.put_object(
                                    self.bucket_name,
                                    minio_path,
                                    io.BytesIO(band_data),
                                    len(band_data),
                                    content_type='image/tiff'
                                )
                                result['minio_paths'].append(minio_path)
                            except Exception as e:
                                print(f"MinIO upload error for {minio_path}: {e}")
                        
                        # Also save locally as backup
                        local_path = os.path.join(
                            settings.MEDIA_ROOT, 'uploads', 'satellite', 
                            str(farm.id), date_str
                        )
                        os.makedirs(local_path, exist_ok=True)
                        with open(os.path.join(local_path, f"{band_key}_{band_id}.tif"), 'wb') as f:
                            f.write(band_data)
            
            result['success'] = len(downloaded_bands) > 0
            result['bands_downloaded'] = len(downloaded_bands)
            result['cloud_cover'] = image.get('cloud_cover', 0)
            result['image_id'] = image.get('id')
            
        except Exception as e:
            result['error'] = str(e)
            print(f"Error downloading image {image.get('id')}: {e}")
        
        return result
    
    def get_available_dates(self, farm, days_back: int = 90) -> List[str]:
        """
        Get list of available image dates for a farm
        """
        from core.models import FieldBoundary
        
        fields = FieldBoundary.objects.filter(farm=farm)
        if not fields.exists():
            return []
        
        boundary = fields.first().boundary
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        images = self.search_sentinel2_images(
            boundary,
            start_date=start_date,
            end_date=end_date,
            max_cloud_cover=50,
            limit=20
        )
        
        dates = []
        for img in images:
            if img.get('datetime'):
                dt = datetime.fromisoformat(img['datetime'].replace('Z', '+00:00'))
                dates.append({
                    'date': dt.strftime('%Y-%m-%d'),
                    'cloud_cover': img.get('cloud_cover', 0),
                    'id': img.get('id')
                })
        
        return sorted(dates, key=lambda x: x['date'], reverse=True)


def download_satellite_for_farm(farm_id: int, target_date: datetime = None) -> Dict:
    """
    Convenience function to download satellite imagery for a farm
    """
    from core.models import Farm
    
    try:
        farm = Farm.objects.get(id=farm_id)
    except Farm.DoesNotExist:
        return {'success': False, 'error': f'Farm {farm_id} not found'}
    
    downloader = PlanetaryComputerDownloader()
    return downloader.download_imagery_for_farm(farm, target_date=target_date)
