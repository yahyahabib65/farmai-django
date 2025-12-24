"""
GeoAI Image Processor
Uses geoai library for geospatial image analysis
Provides advanced NDVI/NDWI calculation and crop classification
"""

import os
import io
from datetime import datetime
from typing import Optional, Dict, List, Tuple, Any
import json

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    import rasterio
    from rasterio.io import MemoryFile
    from rasterio.mask import mask
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

try:
    from shapely.geometry import shape, box, mapping
    from shapely.ops import transform
    import pyproj
    HAS_SHAPELY = True
except ImportError:
    HAS_SHAPELY = False

# Note: GeoAI is available but we use our own implementation
# to avoid heavy dependencies like matplotlib during import
HAS_GEOAI = False  # Disabled to avoid matplotlib import issues


class GeoAIProcessor:
    """
    Advanced geospatial image processing using GeoAI
    Provides NDVI, NDWI, and crop classification
    """
    
    # Crop health thresholds
    NDVI_THRESHOLDS = {
        'critical': 0.0,
        'stressed': 0.2,
        'moderate': 0.4,
        'healthy': 0.6,
        'very_healthy': 0.8
    }
    
    # Water stress thresholds
    NDWI_THRESHOLDS = {
        'severe_drought': -0.3,
        'drought': -0.1,
        'normal': 0.0,
        'wet': 0.2,
        'very_wet': 0.4
    }
    
    def __init__(self):
        """Initialize GeoAI processor"""
        self.available = all([HAS_NUMPY, HAS_RASTERIO, HAS_SHAPELY])
    
    def calculate_ndvi(self, red_band: np.ndarray, nir_band: np.ndarray) -> np.ndarray:
        """
        Calculate NDVI from red and NIR bands
        NDVI = (NIR - Red) / (NIR + Red)
        
        Args:
            red_band: Red band array
            nir_band: NIR band array
        
        Returns:
            NDVI array with values -1 to 1
        """
        if not HAS_NUMPY:
            return None
        
        # Avoid division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            ndvi = (nir_band.astype(float) - red_band.astype(float)) / \
                   (nir_band.astype(float) + red_band.astype(float))
            ndvi = np.where(np.isnan(ndvi), 0, ndvi)
            ndvi = np.clip(ndvi, -1, 1)
        
        return ndvi
    
    def calculate_ndwi(self, green_band: np.ndarray, nir_band: np.ndarray) -> np.ndarray:
        """
        Calculate NDWI (Normalized Difference Water Index)
        NDWI = (Green - NIR) / (Green + NIR)
        
        Args:
            green_band: Green band array
            nir_band: NIR band array
        
        Returns:
            NDWI array with values -1 to 1
        """
        if not HAS_NUMPY:
            return None
        
        with np.errstate(divide='ignore', invalid='ignore'):
            ndwi = (green_band.astype(float) - nir_band.astype(float)) / \
                   (green_band.astype(float) + nir_band.astype(float))
            ndwi = np.where(np.isnan(ndwi), 0, ndwi)
            ndwi = np.clip(ndwi, -1, 1)
        
        return ndwi
    
    def calculate_evi(self, red_band: np.ndarray, nir_band: np.ndarray, 
                      blue_band: np.ndarray, gain: float = 2.5,
                      c1: float = 6.0, c2: float = 7.5, l: float = 1.0) -> np.ndarray:
        """
        Calculate Enhanced Vegetation Index (EVI)
        EVI = G * (NIR - Red) / (NIR + C1*Red - C2*Blue + L)
        
        More sensitive than NDVI in high biomass regions
        
        Args:
            red_band: Red band array
            nir_band: NIR band array
            blue_band: Blue band array
            gain: Gain factor (default 2.5)
            c1, c2: Atmospheric correction coefficients
            l: Canopy background adjustment
        
        Returns:
            EVI array
        """
        if not HAS_NUMPY:
            return None
        
        with np.errstate(divide='ignore', invalid='ignore'):
            evi = gain * (nir_band.astype(float) - red_band.astype(float)) / \
                  (nir_band.astype(float) + c1 * red_band.astype(float) - \
                   c2 * blue_band.astype(float) + l)
            evi = np.where(np.isnan(evi), 0, evi)
            evi = np.clip(evi, -1, 1)
        
        return evi
    
    def calculate_savi(self, red_band: np.ndarray, nir_band: np.ndarray, 
                       l: float = 0.5) -> np.ndarray:
        """
        Calculate Soil-Adjusted Vegetation Index (SAVI)
        SAVI = ((NIR - Red) / (NIR + Red + L)) * (1 + L)
        
        Better for areas with sparse vegetation
        
        Args:
            red_band: Red band array
            nir_band: NIR band array
            l: Soil brightness correction factor (0.5 for intermediate vegetation)
        
        Returns:
            SAVI array
        """
        if not HAS_NUMPY:
            return None
        
        with np.errstate(divide='ignore', invalid='ignore'):
            savi = ((nir_band.astype(float) - red_band.astype(float)) / \
                   (nir_band.astype(float) + red_band.astype(float) + l)) * (1 + l)
            savi = np.where(np.isnan(savi), 0, savi)
            savi = np.clip(savi, -1, 1)
        
        return savi
    
    def classify_crop_health(self, ndvi_array: np.ndarray) -> Dict:
        """
        Classify crop health based on NDVI values
        
        Args:
            ndvi_array: NDVI values array
        
        Returns:
            Classification results with percentages
        """
        if not HAS_NUMPY:
            return {}
        
        total_pixels = ndvi_array.size
        valid_pixels = np.sum(~np.isnan(ndvi_array))
        
        classifications = {
            'critical': np.sum(ndvi_array < self.NDVI_THRESHOLDS['stressed']),
            'stressed': np.sum((ndvi_array >= self.NDVI_THRESHOLDS['stressed']) & 
                              (ndvi_array < self.NDVI_THRESHOLDS['moderate'])),
            'moderate': np.sum((ndvi_array >= self.NDVI_THRESHOLDS['moderate']) & 
                              (ndvi_array < self.NDVI_THRESHOLDS['healthy'])),
            'healthy': np.sum((ndvi_array >= self.NDVI_THRESHOLDS['healthy']) & 
                             (ndvi_array < self.NDVI_THRESHOLDS['very_healthy'])),
            'very_healthy': np.sum(ndvi_array >= self.NDVI_THRESHOLDS['very_healthy'])
        }
        
        percentages = {k: round((v / valid_pixels) * 100, 2) if valid_pixels > 0 else 0 
                       for k, v in classifications.items()}
        
        return {
            'total_pixels': int(total_pixels),
            'valid_pixels': int(valid_pixels),
            'classifications': classifications,
            'percentages': percentages,
            'dominant_class': max(percentages, key=percentages.get)
        }
    
    def classify_water_stress(self, ndwi_array: np.ndarray) -> Dict:
        """
        Classify water stress based on NDWI values
        
        Args:
            ndwi_array: NDWI values array
        
        Returns:
            Water stress classification
        """
        if not HAS_NUMPY:
            return {}
        
        total_pixels = ndwi_array.size
        valid_pixels = np.sum(~np.isnan(ndwi_array))
        
        classifications = {
            'severe_drought': np.sum(ndwi_array < self.NDWI_THRESHOLDS['drought']),
            'drought': np.sum((ndwi_array >= self.NDWI_THRESHOLDS['drought']) & 
                             (ndwi_array < self.NDWI_THRESHOLDS['normal'])),
            'normal': np.sum((ndwi_array >= self.NDWI_THRESHOLDS['normal']) & 
                            (ndwi_array < self.NDWI_THRESHOLDS['wet'])),
            'wet': np.sum((ndwi_array >= self.NDWI_THRESHOLDS['wet']) & 
                         (ndwi_array < self.NDWI_THRESHOLDS['very_wet'])),
            'very_wet': np.sum(ndwi_array >= self.NDWI_THRESHOLDS['very_wet'])
        }
        
        percentages = {k: round((v / valid_pixels) * 100, 2) if valid_pixels > 0 else 0 
                       for k, v in classifications.items()}
        
        return {
            'total_pixels': int(total_pixels),
            'valid_pixels': int(valid_pixels),
            'classifications': classifications,
            'percentages': percentages,
            'dominant_class': max(percentages, key=percentages.get)
        }
    
    def process_multispectral_image(self, image_path: str, boundary_geom=None) -> Dict:
        """
        Process a multispectral image and calculate all vegetation indices
        
        Args:
            image_path: Path to multispectral GeoTIFF
            boundary_geom: Optional boundary to clip to
        
        Returns:
            Comprehensive analysis results
        """
        if not self.available:
            return {
                'success': False,
                'error': 'Required libraries not available (numpy, rasterio, shapely)'
            }
        
        try:
            with rasterio.open(image_path) as src:
                # Read bands (assuming standard Sentinel-2 band order)
                # Adjust based on actual band configuration
                if src.count >= 4:
                    blue = src.read(1).astype(float)
                    green = src.read(2).astype(float)
                    red = src.read(3).astype(float)
                    nir = src.read(4).astype(float)
                else:
                    return {
                        'success': False,
                        'error': f'Image has only {src.count} bands, need at least 4'
                    }
                
                # Clip to boundary if provided
                if boundary_geom and HAS_SHAPELY:
                    try:
                        if hasattr(boundary_geom, 'geojson'):
                            geom = json.loads(boundary_geom.geojson)
                        else:
                            geom = boundary_geom
                        
                        geom_shape = shape(geom)
                        out_image, out_transform = mask(src, [geom_shape], crop=True)
                        
                        if out_image.shape[0] >= 4:
                            blue = out_image[0].astype(float)
                            green = out_image[1].astype(float)
                            red = out_image[2].astype(float)
                            nir = out_image[3].astype(float)
                    except Exception as e:
                        print(f"Clipping error: {e}, using full image")
                
                # Calculate indices
                ndvi = self.calculate_ndvi(red, nir)
                ndwi = self.calculate_ndwi(green, nir)
                evi = self.calculate_evi(red, nir, blue)
                savi = self.calculate_savi(red, nir)
                
                # Get classifications
                health = self.classify_crop_health(ndvi)
                water_stress = self.classify_water_stress(ndwi)
                
                # Generate statistics
                results = {
                    'success': True,
                    'image_info': {
                        'path': image_path,
                        'width': src.width,
                        'height': src.height,
                        'crs': str(src.crs) if src.crs else None,
                        'bounds': list(src.bounds)
                    },
                    'indices': {
                        'ndvi': {
                            'min': float(np.nanmin(ndvi)),
                            'max': float(np.nanmax(ndvi)),
                            'mean': float(np.nanmean(ndvi)),
                            'std': float(np.nanstd(ndvi)),
                            'median': float(np.nanmedian(ndvi))
                        },
                        'ndwi': {
                            'min': float(np.nanmin(ndwi)),
                            'max': float(np.nanmax(ndwi)),
                            'mean': float(np.nanmean(ndwi)),
                            'std': float(np.nanstd(ndwi)),
                            'median': float(np.nanmedian(ndwi))
                        },
                        'evi': {
                            'min': float(np.nanmin(evi)),
                            'max': float(np.nanmax(evi)),
                            'mean': float(np.nanmean(evi)),
                            'std': float(np.nanstd(evi)),
                            'median': float(np.nanmedian(evi))
                        },
                        'savi': {
                            'min': float(np.nanmin(savi)),
                            'max': float(np.nanmax(savi)),
                            'mean': float(np.nanmean(savi)),
                            'std': float(np.nanstd(savi)),
                            'median': float(np.nanmedian(savi))
                        }
                    },
                    'crop_health': health,
                    'water_stress': water_stress,
                    'analysis_timestamp': datetime.now().isoformat()
                }
                
                # Generate farmer-friendly summary
                results['summary'] = self._generate_summary(results)
                
                return results
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def _generate_summary(self, results: Dict) -> Dict:
        """Generate farmer-friendly summary of analysis"""
        ndvi_mean = results['indices']['ndvi']['mean']
        health_class = results['crop_health'].get('dominant_class', 'unknown')
        water_class = results['water_stress'].get('dominant_class', 'unknown')
        
        # Overall health assessment
        if ndvi_mean >= 0.6:
            overall_health = 'Excellent'
            recommendation = 'Crops are healthy. Continue current practices.'
        elif ndvi_mean >= 0.4:
            overall_health = 'Good'
            recommendation = 'Crops are doing well. Monitor for any changes.'
        elif ndvi_mean >= 0.2:
            overall_health = 'Fair'
            recommendation = 'Some stress detected. Check soil moisture and nutrients.'
        else:
            overall_health = 'Poor'
            recommendation = 'Significant stress detected. Immediate attention needed - check water and fertilizer.'
        
        # Water stress advice
        if 'drought' in water_class.lower():
            water_advice = 'Water stress detected. Consider irrigation.'
        elif 'wet' in water_class.lower():
            water_advice = 'Excess moisture. Monitor for waterlogging.'
        else:
            water_advice = 'Water levels appear normal.'
        
        return {
            'overall_health': overall_health,
            'health_score': round(ndvi_mean * 100),  # 0-100 scale
            'recommendation': recommendation,
            'water_status': water_class.replace('_', ' ').title(),
            'water_advice': water_advice,
            'healthy_area_percent': results['crop_health']['percentages'].get('healthy', 0) + \
                                    results['crop_health']['percentages'].get('very_healthy', 0),
            'stressed_area_percent': results['crop_health']['percentages'].get('critical', 0) + \
                                     results['crop_health']['percentages'].get('stressed', 0)
        }
    
    def process_band_files(self, red_path: str, nir_path: str, 
                          green_path: str = None, blue_path: str = None) -> Dict:
        """
        Process separate band files (common for satellite imagery)
        
        Args:
            red_path: Path to red band file
            nir_path: Path to NIR band file
            green_path: Optional path to green band file
            blue_path: Optional path to blue band file
        
        Returns:
            Analysis results
        """
        if not self.available:
            return {'success': False, 'error': 'Required libraries not available'}
        
        try:
            with rasterio.open(red_path) as red_src:
                red = red_src.read(1).astype(float)
            
            with rasterio.open(nir_path) as nir_src:
                nir = nir_src.read(1).astype(float)
            
            # Calculate NDVI
            ndvi = self.calculate_ndvi(red, nir)
            health = self.classify_crop_health(ndvi)
            
            results = {
                'success': True,
                'indices': {
                    'ndvi': {
                        'min': float(np.nanmin(ndvi)),
                        'max': float(np.nanmax(ndvi)),
                        'mean': float(np.nanmean(ndvi)),
                        'std': float(np.nanstd(ndvi))
                    }
                },
                'crop_health': health
            }
            
            # Add NDWI if green band available
            if green_path:
                with rasterio.open(green_path) as green_src:
                    green = green_src.read(1).astype(float)
                
                ndwi = self.calculate_ndwi(green, nir)
                water_stress = self.classify_water_stress(ndwi)
                
                results['indices']['ndwi'] = {
                    'min': float(np.nanmin(ndwi)),
                    'max': float(np.nanmax(ndwi)),
                    'mean': float(np.nanmean(ndwi)),
                    'std': float(np.nanstd(ndwi))
                }
                results['water_stress'] = water_stress
            
            # Add EVI if blue band available
            if blue_path and green_path:
                with rasterio.open(blue_path) as blue_src:
                    blue = blue_src.read(1).astype(float)
                
                evi = self.calculate_evi(red, nir, blue)
                results['indices']['evi'] = {
                    'min': float(np.nanmin(evi)),
                    'max': float(np.nanmax(evi)),
                    'mean': float(np.nanmean(evi)),
                    'std': float(np.nanstd(evi))
                }
            
            return results
            
        except Exception as e:
            return {'success': False, 'error': str(e)}


# Singleton instance for easy use
geoai_processor = GeoAIProcessor()
