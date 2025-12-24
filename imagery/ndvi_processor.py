"""
Real NDVI/NDWI Calculation Module
Processes satellite/drone imagery to calculate vegetation and water indices
"""

import os
import numpy as np
from PIL import Image
from datetime import datetime

try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False


def calculate_ndvi_from_bands(nir_band, red_band):
    """
    Calculate NDVI from NIR and RED bands
    NDVI = (NIR - RED) / (NIR + RED)
    
    Returns values between -1 and 1
    """
    # Avoid division by zero
    denominator = nir_band.astype(float) + red_band.astype(float)
    denominator[denominator == 0] = 0.0001
    
    ndvi = (nir_band.astype(float) - red_band.astype(float)) / denominator
    
    # Clip to valid range
    ndvi = np.clip(ndvi, -1, 1)
    
    return ndvi


def calculate_ndwi_from_bands(green_band, nir_band):
    """
    Calculate NDWI (Normalized Difference Water Index)
    NDWI = (GREEN - NIR) / (GREEN + NIR)
    
    Returns values between -1 and 1
    """
    denominator = green_band.astype(float) + nir_band.astype(float)
    denominator[denominator == 0] = 0.0001
    
    ndwi = (green_band.astype(float) - nir_band.astype(float)) / denominator
    ndwi = np.clip(ndwi, -1, 1)
    
    return ndwi


def process_tiff_image(file_path):
    """
    Process a GeoTIFF or TIFF image and calculate NDVI/NDWI
    
    For Sentinel-2 imagery:
    - Band 4 = RED (665nm)
    - Band 8 = NIR (842nm)
    - Band 3 = GREEN (560nm)
    
    For regular RGB images, we simulate using:
    - NIR ≈ inverted Blue channel (approximation)
    - RED = Red channel
    - GREEN = Green channel
    """
    results = {
        'success': False,
        'ndvi_mean': None,
        'ndwi_mean': None,
        'ndvi_min': None,
        'ndvi_max': None,
        'ndwi_min': None,
        'ndwi_max': None,
        'ndvi_array': None,
        'ndwi_array': None,
        'pixel_count': 0,
        'healthy_pixels_pct': 0,
        'stressed_pixels_pct': 0,
        'error': None
    }
    
    try:
        if HAS_RASTERIO and file_path.lower().endswith(('.tif', '.tiff')):
            # Use rasterio for GeoTIFF
            with rasterio.open(file_path) as src:
                band_count = src.count
                
                if band_count >= 4:
                    # Multi-band satellite image (Sentinel-2 style)
                    # Assuming Band 3=Green, Band 4=Red, Band 5 or 8=NIR
                    red_band = src.read(3 if band_count > 4 else 1).astype(float)
                    green_band = src.read(2 if band_count > 3 else 1).astype(float)
                    
                    # Try to get NIR band
                    if band_count >= 8:
                        nir_band = src.read(8).astype(float)
                    elif band_count >= 5:
                        nir_band = src.read(5).astype(float)
                    else:
                        nir_band = src.read(4).astype(float)
                    
                elif band_count >= 3:
                    # RGB image - use approximation
                    red_band = src.read(1).astype(float)
                    green_band = src.read(2).astype(float)
                    blue_band = src.read(3).astype(float)
                    # Approximate NIR using inverse of blue (rough approximation)
                    nir_band = (255 - blue_band) * 1.2  # Scale factor for better NDVI range
                    
                else:
                    # Single band - can't calculate NDVI
                    results['error'] = 'Image has insufficient bands for NDVI calculation'
                    return results
                    
        else:
            # Use PIL for regular images
            img = Image.open(file_path)
            img_array = np.array(img)
            
            if len(img_array.shape) < 3:
                results['error'] = 'Grayscale image - cannot calculate NDVI'
                return results
            
            red_band = img_array[:, :, 0].astype(float)
            green_band = img_array[:, :, 1].astype(float)
            blue_band = img_array[:, :, 2].astype(float)
            
            # Approximate NIR from blue channel inversion
            nir_band = (255 - blue_band) * 1.2
        
        # Calculate indices
        ndvi = calculate_ndvi_from_bands(nir_band, red_band)
        ndwi = calculate_ndwi_from_bands(green_band, nir_band)
        
        # Calculate statistics
        valid_ndvi = ndvi[~np.isnan(ndvi)]
        valid_ndwi = ndwi[~np.isnan(ndwi)]
        
        results['success'] = True
        results['ndvi_mean'] = round(float(np.mean(valid_ndvi)), 4)
        results['ndvi_min'] = round(float(np.min(valid_ndvi)), 4)
        results['ndvi_max'] = round(float(np.max(valid_ndvi)), 4)
        results['ndvi_std'] = round(float(np.std(valid_ndvi)), 4)
        
        results['ndwi_mean'] = round(float(np.mean(valid_ndwi)), 4)
        results['ndwi_min'] = round(float(np.min(valid_ndwi)), 4)
        results['ndwi_max'] = round(float(np.max(valid_ndwi)), 4)
        results['ndwi_std'] = round(float(np.std(valid_ndwi)), 4)
        
        results['pixel_count'] = len(valid_ndvi.flatten())
        
        # Calculate health percentages based on NDVI thresholds
        total_pixels = len(valid_ndvi.flatten())
        if total_pixels > 0:
            healthy_pixels = np.sum(valid_ndvi >= 0.4)
            stressed_pixels = np.sum((valid_ndvi >= 0.2) & (valid_ndvi < 0.4))
            bare_pixels = np.sum(valid_ndvi < 0.2)
            
            results['healthy_pixels_pct'] = round((healthy_pixels / total_pixels) * 100, 1)
            results['stressed_pixels_pct'] = round((stressed_pixels / total_pixels) * 100, 1)
            results['bare_soil_pct'] = round((bare_pixels / total_pixels) * 100, 1)
        
        # Store arrays for visualization (downsampled to save memory)
        max_size = 100
        step_y = max(1, ndvi.shape[0] // max_size)
        step_x = max(1, ndvi.shape[1] // max_size)
        results['ndvi_array'] = ndvi[::step_y, ::step_x].tolist()
        results['ndwi_array'] = ndwi[::step_y, ::step_x].tolist()
        
    except Exception as e:
        results['error'] = str(e)
    
    return results


def process_uploaded_image(file_path, save_results=True):
    """
    Process an uploaded image and return NDVI/NDWI results
    """
    if not os.path.exists(file_path):
        return {'success': False, 'error': f'File not found: {file_path}'}
    
    return process_tiff_image(file_path)


def get_ndvi_color(ndvi_value):
    """
    Return a color code based on NDVI value
    """
    if ndvi_value >= 0.6:
        return '#228B22'  # Forest Green - Very healthy
    elif ndvi_value >= 0.4:
        return '#32CD32'  # Lime Green - Healthy
    elif ndvi_value >= 0.2:
        return '#FFD700'  # Gold - Moderate stress
    elif ndvi_value >= 0:
        return '#FFA500'  # Orange - Stressed
    else:
        return '#8B4513'  # Brown - Bare soil/water


def get_health_status(ndvi_value):
    """
    Return health status based on NDVI
    """
    if ndvi_value >= 0.6:
        return 'Excellent', 'green'
    elif ndvi_value >= 0.4:
        return 'Healthy', 'green'
    elif ndvi_value >= 0.25:
        return 'Moderate', 'yellow'
    else:
        return 'Needs Attention', 'red'
