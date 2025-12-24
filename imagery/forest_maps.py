"""
Forest Maps Downloader

Implements downloads for:
- JAXA ALOS Forest/Non-Forest Maps
- University of Maryland GLAD Forest Cover
- Hansen Global Forest Change (GFC)
"""

import os
import requests
import numpy as np
from typing import Dict, Tuple, Optional, List
from datetime import datetime
import logging
import tempfile

logger = logging.getLogger(__name__)


class ForestMapDownloader:
    """
    Base class for forest map downloaders.
    """
    
    def __init__(self, output_dir: str = 'media/forest_maps'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def _download_file(self, url: str, output_path: str) -> bool:
        """
        Download file from URL.
        """
        try:
            response = requests.get(url, stream=True, timeout=300)
            response.raise_for_status()
            
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            logger.info(f"Downloaded: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Download failed: {e}")
            return False


class ALOSForestDownloader(ForestMapDownloader):
    """
    Download JAXA ALOS PALSAR Forest/Non-Forest Maps.
    
    Available years: 2007-2020
    Resolution: 25m
    Source: https://www.eorc.jaxa.jp/ALOS/en/palsar_fnf/fnf_index.htm
    """
    
    BASE_URL = "https://www.eorc.jaxa.jp/ALOS/en/palsar_fnf/data"
    
    def __init__(self, output_dir: str = 'media/forest_maps/alos'):
        super().__init__(output_dir)
    
    def get_tile_name(self, lat: float, lon: float) -> str:
        """
        Get ALOS tile name for coordinates.
        
        Tiles are 1x1 degree.
        """
        # Determine hemisphere
        lat_hem = 'N' if lat >= 0 else 'S'
        lon_hem = 'E' if lon >= 0 else 'W'
        
        # Get tile corner
        lat_tile = int(abs(lat))
        lon_tile = int(abs(lon))
        
        return f"{lat_hem}{lat_tile:02d}{lon_hem}{lon_tile:03d}"
    
    def download_for_location(self, 
                              lat: float, 
                              lon: float, 
                              year: int = 2020) -> Dict:
        """
        Download ALOS forest map for a location.
        
        Args:
            lat: Latitude
            lon: Longitude
            year: Year of data (2007-2020)
            
        Returns:
            Download result with file path
        """
        if year < 2007 or year > 2020:
            return {'error': f'Year must be between 2007-2020, got {year}'}
        
        tile_name = self.get_tile_name(lat, lon)
        
        # Construct URL
        # Note: Actual URL structure may vary - this is a template
        url = f"{self.BASE_URL}/{year}/{tile_name}_F02DAR.tar.gz"
        
        output_file = os.path.join(
            self.output_dir, 
            f"alos_fnf_{tile_name}_{year}.tar.gz"
        )
        
        # Check if already downloaded
        if os.path.exists(output_file):
            return {
                'status': 'exists',
                'file_path': output_file,
                'tile': tile_name,
                'year': year
            }
        
        # Note: JAXA requires registration for download
        # This would need authentication in production
        logger.warning("ALOS download requires JAXA registration")
        
        return {
            'status': 'requires_authentication',
            'tile': tile_name,
            'year': year,
            'url': url,
            'note': 'Register at https://www.eorc.jaxa.jp/ALOS/en/palsar_fnf/fnf_index.htm'
        }
    
    def get_available_years(self) -> List[int]:
        """Get list of available years."""
        return list(range(2007, 2021))


class GLADForestDownloader(ForestMapDownloader):
    """
    Download University of Maryland GLAD Forest Cover.
    
    Source: https://glad.umd.edu/dataset/global-forest-cover-2020
    Resolution: 30m
    """
    
    BASE_URL = "https://storage.googleapis.com/earthenginepartners-hansen"
    
    def __init__(self, output_dir: str = 'media/forest_maps/glad'):
        super().__init__(output_dir)
    
    def get_tile_name(self, lat: float, lon: float) -> str:
        """
        Get GLAD tile name for coordinates.
        
        Tiles are 10x10 degrees.
        """
        # Calculate tile boundaries
        lat_tile = int(np.floor(lat / 10) * 10)
        lon_tile = int(np.floor(lon / 10) * 10)
        
        lat_str = f"{abs(lat_tile):02d}{'N' if lat_tile >= 0 else 'S'}"
        lon_str = f"{abs(lon_tile):03d}{'E' if lon_tile >= 0 else 'W'}"
        
        return f"{lat_str}_{lon_str}"
    
    def download_for_location(self,
                              lat: float,
                              lon: float,
                              layer: str = 'treecover2000') -> Dict:
        """
        Download GLAD forest cover for a location.
        
        Args:
            lat: Latitude
            lon: Longitude
            layer: One of 'treecover2000', 'gain', 'lossyear', 'datamask'
            
        Returns:
            Download result
        """
        valid_layers = ['treecover2000', 'gain', 'lossyear', 'datamask']
        if layer not in valid_layers:
            return {'error': f'Invalid layer. Must be one of: {valid_layers}'}
        
        tile_name = self.get_tile_name(lat, lon)
        
        # Construct filename
        # Format: Hansen_GFC-2020-v1.8_{layer}_{tile}.tif
        filename = f"Hansen_GFC-2020-v1.8_{layer}_{tile_name}.tif"
        url = f"{self.BASE_URL}/GFC-2020-v1.8/{filename}"
        
        output_file = os.path.join(self.output_dir, filename)
        
        if os.path.exists(output_file):
            return {
                'status': 'exists',
                'file_path': output_file,
                'tile': tile_name,
                'layer': layer
            }
        
        # Download
        success = self._download_file(url, output_file)
        
        if success:
            return {
                'status': 'downloaded',
                'file_path': output_file,
                'tile': tile_name,
                'layer': layer
            }
        else:
            return {
                'status': 'failed',
                'url': url,
                'note': 'Download failed - check URL and network'
            }
    
    def get_available_layers(self) -> Dict:
        """Get available data layers."""
        return {
            'treecover2000': 'Tree canopy cover for year 2000 (0-100%)',
            'gain': 'Forest gain 2000-2020 (binary)',
            'lossyear': 'Year of forest loss (1-20, representing 2001-2020)',
            'datamask': 'Data mask (1=land, 2=water)'
        }


class HansenGFCDownloader(ForestMapDownloader):
    """
    Download Hansen Global Forest Change maps.
    
    This is essentially the same as GLAD but provides additional utilities.
    Source: https://earthenginepartners.appspot.com/science-2013-global-forest
    """
    
    BASE_URL = "https://storage.googleapis.com/earthenginepartners-hansen"
    VERSION = "GFC-2023-v1.11"  # Latest version
    
    def __init__(self, output_dir: str = 'media/forest_maps/hansen'):
        super().__init__(output_dir)
    
    def get_tile_id(self, lat: float, lon: float) -> str:
        """
        Get Hansen tile ID for coordinates.
        """
        # Calculate 10x10 degree tile
        lat_tile = int(np.floor(lat / 10) * 10)
        lon_tile = int(np.floor(lon / 10) * 10)
        
        lat_str = f"{abs(lat_tile):02d}{'N' if lat_tile >= 0 else 'S'}"
        lon_str = f"{abs(lon_tile):03d}{'E' if lon_tile >= 0 else 'W'}"
        
        return f"{lat_str}_{lon_str}"
    
    def download_layer(self,
                       lat: float,
                       lon: float,
                       layer: str = 'treecover2000') -> Dict:
        """
        Download a specific Hansen GFC layer.
        
        Args:
            lat: Latitude
            lon: Longitude
            layer: Layer name
            
        Returns:
            Download result
        """
        tile_id = self.get_tile_id(lat, lon)
        
        filename = f"Hansen_{self.VERSION}_{layer}_{tile_id}.tif"
        url = f"{self.BASE_URL}/{self.VERSION}/{filename}"
        
        output_file = os.path.join(self.output_dir, filename)
        
        if os.path.exists(output_file):
            return {
                'status': 'exists',
                'file_path': output_file,
                'tile': tile_id,
                'layer': layer
            }
        
        success = self._download_file(url, output_file)
        
        return {
            'status': 'downloaded' if success else 'failed',
            'file_path': output_file if success else None,
            'tile': tile_id,
            'layer': layer,
            'url': url
        }
    
    def download_all_layers(self, lat: float, lon: float) -> Dict:
        """
        Download all available layers for a location.
        """
        layers = ['treecover2000', 'gain', 'lossyear', 'datamask', 'first', 'last']
        results = {}
        
        for layer in layers:
            results[layer] = self.download_layer(lat, lon, layer)
        
        return {
            'location': {'lat': lat, 'lon': lon},
            'tile': self.get_tile_id(lat, lon),
            'layers': results
        }
    
    def calculate_forest_change(self, 
                                treecover: np.ndarray,
                                lossyear: np.ndarray,
                                gain: np.ndarray,
                                start_year: int = 2000,
                                end_year: int = 2023,
                                threshold: int = 30) -> Dict:
        """
        Calculate forest change statistics.
        
        Args:
            treecover: Tree cover 2000 (0-100)
            lossyear: Year of loss (1-23)
            gain: Forest gain (0/1)
            start_year: Analysis start year
            end_year: Analysis end year
            threshold: Tree cover threshold for forest (%)
            
        Returns:
            Forest change statistics
        """
        # Forest in 2000
        forest_2000 = treecover >= threshold
        
        # Loss during period
        loss_years = lossyear > 0
        period_loss = loss_years & (lossyear >= (start_year - 2000 + 1)) & (lossyear <= (end_year - 2000 + 1))
        
        # Current forest estimate
        forest_current = (forest_2000 | (gain > 0)) & ~period_loss
        
        # Calculate areas (assuming 30m resolution)
        pixel_area_ha = 0.09  # 30m x 30m = 900 m² = 0.09 ha
        
        forest_2000_area = np.sum(forest_2000) * pixel_area_ha
        forest_current_area = np.sum(forest_current) * pixel_area_ha
        loss_area = np.sum(period_loss & forest_2000) * pixel_area_ha
        gain_area = np.sum(gain > 0) * pixel_area_ha
        
        return {
            'forest_2000_ha': float(forest_2000_area),
            'forest_current_ha': float(forest_current_area),
            'gross_loss_ha': float(loss_area),
            'gross_gain_ha': float(gain_area),
            'net_change_ha': float(forest_current_area - forest_2000_area),
            'net_change_percent': float((forest_current_area - forest_2000_area) / forest_2000_area * 100) if forest_2000_area > 0 else 0,
            'period': f'{start_year}-{end_year}',
            'threshold_percent': threshold
        }


class ForestChangeDetector:
    """
    Detect forest changes using downloaded forest maps.
    """
    
    def __init__(self):
        self.hansen = HansenGFCDownloader()
        self.glad = GLADForestDownloader()
    
    def analyze_location(self, 
                         lat: float, 
                         lon: float,
                         buffer_km: float = 10) -> Dict:
        """
        Analyze forest change for a location.
        
        Args:
            lat: Center latitude
            lon: Center longitude
            buffer_km: Analysis radius in km
            
        Returns:
            Forest change analysis
        """
        # Download data
        hansen_result = self.hansen.download_all_layers(lat, lon)
        
        # Check if all layers downloaded
        layers_available = all(
            r.get('status') in ['downloaded', 'exists']
            for r in hansen_result['layers'].values()
        )
        
        if not layers_available:
            return {
                'status': 'data_unavailable',
                'location': {'lat': lat, 'lon': lon},
                'note': 'Some forest map layers could not be downloaded'
            }
        
        # Would load and analyze rasters here
        # For now, return structure
        
        return {
            'status': 'ready_for_analysis',
            'location': {'lat': lat, 'lon': lon},
            'buffer_km': buffer_km,
            'data': hansen_result,
            'note': 'Load rasters and call HansenGFCDownloader.calculate_forest_change()'
        }
    
    def detect_deforestation_alerts(self,
                                    lat: float,
                                    lon: float,
                                    threshold_ha: float = 1.0) -> Dict:
        """
        Detect recent deforestation alerts.
        
        Args:
            lat: Latitude
            lon: Longitude
            threshold_ha: Minimum area for alert
            
        Returns:
            Deforestation alerts
        """
        # This would integrate with GLAD alerts or similar
        # For now, return structure
        
        return {
            'location': {'lat': lat, 'lon': lon},
            'threshold_ha': threshold_ha,
            'alerts': [],
            'note': 'Integrate with GLAD Forest Alerts API for real-time monitoring'
        }


def download_forest_maps_for_farm(farm_id: int) -> Dict:
    """
    Download forest maps for a farm location.
    """
    from core.models import Farm
    
    try:
        farm = Farm.objects.get(id=farm_id)
    except Farm.DoesNotExist:
        return {'error': f'Farm {farm_id} not found'}
    
    # Get farm coordinates
    if hasattr(farm, 'location') and farm.location:
        lat = farm.location.y
        lon = farm.location.x
    else:
        return {'error': 'Farm has no location set'}
    
    # Download from all sources
    results = {
        'farm_id': farm_id,
        'farm_name': farm.name,
        'location': {'lat': lat, 'lon': lon}
    }
    
    # Hansen/GLAD
    hansen = HansenGFCDownloader()
    results['hansen'] = hansen.download_all_layers(lat, lon)
    
    # ALOS
    alos = ALOSForestDownloader()
    results['alos'] = alos.download_for_location(lat, lon, year=2020)
    
    return results
