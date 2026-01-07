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


def calculate_ndvi_from_separate_bands(folder_path: str) -> dict:
    """
    Calculate NDVI/NDWI from separate band files in a folder.
    
    Expected files in folder:
    - nir_B08.tif (NIR band)
    - red_B04.tif (RED band)
    - green_B03.tif (GREEN band)
    
    Args:
        folder_path: Path to folder containing band files
    
    Returns:
        Dict with NDVI/NDWI results
    """
    results = {
        'success': False,
        'ndvi_mean': None,
        'ndwi_mean': None,
        'ndvi_min': None,
        'ndvi_max': None,
        'ndvi_std': None,
        'ndwi_min': None,
        'ndwi_max': None,
        'ndwi_std': None,
        'pixel_count': 0,
        'healthy_pixels_pct': 0,
        'stressed_pixels_pct': 0,
        'bare_soil_pct': 0,
        'error': None
    }
    
    if not os.path.exists(folder_path):
        results['error'] = f'Folder not found: {folder_path}'
        return results
    
    # Find band files
    nir_file = None
    red_file = None
    green_file = None
    
    for f in os.listdir(folder_path):
        f_lower = f.lower()
        if 'nir' in f_lower or 'b08' in f_lower or 'b8' in f_lower:
            nir_file = os.path.join(folder_path, f)
        elif 'red' in f_lower or 'b04' in f_lower or 'b4' in f_lower:
            red_file = os.path.join(folder_path, f)
        elif 'green' in f_lower or 'b03' in f_lower or 'b3' in f_lower:
            green_file = os.path.join(folder_path, f)
    
    if not nir_file or not red_file:
        results['error'] = f'Required bands not found. Need NIR and RED. Found: nir={nir_file}, red={red_file}'
        return results
    
    try:
        if HAS_RASTERIO:
            import rasterio
            
            # Read NIR and RED bands
            with rasterio.open(nir_file) as nir_src:
                nir_band = nir_src.read(1).astype(float)
            
            with rasterio.open(red_file) as red_src:
                red_band = red_src.read(1).astype(float)
            
            # Calculate NDVI
            ndvi = calculate_ndvi_from_bands(nir_band, red_band)
            
            # Calculate NDWI if green band exists
            ndwi = None
            if green_file:
                with rasterio.open(green_file) as green_src:
                    green_band = green_src.read(1).astype(float)
                ndwi = calculate_ndwi_from_bands(green_band, nir_band)
            
            # Calculate statistics
            valid_ndvi = ndvi[~np.isnan(ndvi)]
            
            if len(valid_ndvi) > 0:
                results['success'] = True
                results['ndvi_mean'] = round(float(np.mean(valid_ndvi)), 4)
                results['ndvi_min'] = round(float(np.min(valid_ndvi)), 4)
                results['ndvi_max'] = round(float(np.max(valid_ndvi)), 4)
                results['ndvi_std'] = round(float(np.std(valid_ndvi)), 4)
                results['pixel_count'] = len(valid_ndvi)
                
                # Calculate health percentages
                healthy = np.sum(valid_ndvi >= 0.4) / len(valid_ndvi) * 100
                stressed = np.sum((valid_ndvi >= 0.1) & (valid_ndvi < 0.4)) / len(valid_ndvi) * 100
                bare_soil = np.sum(valid_ndvi < 0.1) / len(valid_ndvi) * 100
                
                results['healthy_pixels_pct'] = round(healthy, 2)
                results['stressed_pixels_pct'] = round(stressed, 2)
                results['bare_soil_pct'] = round(bare_soil, 2)
            
            if ndwi is not None:
                valid_ndwi = ndwi[~np.isnan(ndwi)]
                if len(valid_ndwi) > 0:
                    results['ndwi_mean'] = round(float(np.mean(valid_ndwi)), 4)
                    results['ndwi_min'] = round(float(np.min(valid_ndwi)), 4)
                    results['ndwi_max'] = round(float(np.max(valid_ndwi)), 4)
                    results['ndwi_std'] = round(float(np.std(valid_ndwi)), 4)
        else:
            results['error'] = 'rasterio not available'
            
    except Exception as e:
        results['error'] = str(e)
    
    return results


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


def generate_ndvi_image(folder_path: str, output_path: str, farm_id: int, image_date: str) -> dict:
    """
    Generate NDVI visualization image from separate band files.
    Creates a colorized heatmap PNG and stores directly in MinIO.
    
    Args:
        folder_path: Path to folder containing band files (NIR, RED)
        output_path: Base path (used for MinIO bucket path structure)
        farm_id: Farm ID for naming
        image_date: Image date string (YYYY-MM-DD)
    
    Returns:
        Dict with MinIO URLs to generated images
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import io
    
    results = {
        'success': False,
        'ndvi_image_path': None,
        'ndwi_image_path': None,
        'ndvi_image_url': None,
        'ndwi_image_url': None,
        'ndvi_mean': None,
        'ndwi_mean': None,
        'error': None
    }
    
    # Find band files
    nir_file = None
    red_file = None
    green_file = None
    
    for f in os.listdir(folder_path):
        f_lower = f.lower()
        if 'nir' in f_lower or 'b08' in f_lower:
            nir_file = os.path.join(folder_path, f)
        elif 'red' in f_lower or 'b04' in f_lower:
            red_file = os.path.join(folder_path, f)
        elif 'green' in f_lower or 'b03' in f_lower:
            green_file = os.path.join(folder_path, f)
    
    if not nir_file or not red_file:
        results['error'] = 'Required NIR and RED bands not found'
        return results
    
    try:
        if not HAS_RASTERIO:
            results['error'] = 'rasterio not available'
            return results
        
        import rasterio
        
        # Initialize MinIO client
        try:
            from minio import Minio
            minio_client = Minio(
                os.getenv('MINIO_ENDPOINT', 'localhost:9000'),
                access_key=os.getenv('MINIO_ACCESS_KEY', 'minioadmin'),
                secret_key=os.getenv('MINIO_SECRET_KEY', 'minioadmin'),
                secure=False
            )
            bucket_name = 'analysis-results'
            # Ensure bucket exists
            if not minio_client.bucket_exists(bucket_name):
                minio_client.make_bucket(bucket_name)
        except Exception as e:
            results['error'] = f'MinIO connection failed: {e}'
            return results
        
        # Read bands
        with rasterio.open(nir_file) as nir_src:
            nir_band = nir_src.read(1).astype(float)
        
        with rasterio.open(red_file) as red_src:
            red_band = red_src.read(1).astype(float)
        
        # Calculate NDVI
        ndvi = calculate_ndvi_from_bands(nir_band, red_band)
        
        # Generate NDVI image with RdYlGn colormap (red-yellow-green)
        fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
        cmap = plt.cm.RdYlGn
        
        # Normalize NDVI to 0-1 for colormap (NDVI ranges from -1 to 1)
        ndvi_norm = (ndvi + 1) / 2  # Map -1,1 to 0,1
        
        im = ax.imshow(ndvi_norm, cmap=cmap, vmin=0, vmax=1, interpolation='nearest', aspect='auto')
        ax.set_axis_off()
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, orientation='vertical')
        cbar.set_label('NDVI', fontsize=12, fontweight='bold')
        cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
        cbar.set_ticklabels(['-1.0', '-0.5', '0.0', '0.5', '1.0'])
        
        ax.set_title(f'NDVI - Farm {farm_id} - {image_date}', fontsize=14, fontweight='bold', pad=10)
        plt.tight_layout()
        
        # Save to bytes buffer and upload to MinIO
        ndvi_buffer = io.BytesIO()
        fig.savefig(ndvi_buffer, format='png', dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        ndvi_buffer.seek(0)
        
        ndvi_object_name = f"ndvi/farm_{farm_id}_{image_date}_ndvi.png"
        minio_client.put_object(
            bucket_name,
            ndvi_object_name,
            ndvi_buffer,
            length=ndvi_buffer.getbuffer().nbytes,
            content_type='image/png'
        )
        
        results['ndvi_image_path'] = f"{bucket_name}/{ndvi_object_name}"
        results['ndvi_image_url'] = f"/api/analysis-image/?path={ndvi_object_name}"
        results['ndvi_mean'] = round(float(np.nanmean(ndvi)), 4)
        
        # Generate NDWI image if green band exists
        if green_file:
            with rasterio.open(green_file) as green_src:
                green_band = green_src.read(1).astype(float)
            
            ndwi = calculate_ndwi_from_bands(green_band, nir_band)
            
            fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
            cmap = plt.cm.Blues_r  # Reversed so high water = blue
            ndwi_norm = (ndwi + 1) / 2
            
            im = ax.imshow(ndwi_norm, cmap=cmap, vmin=0, vmax=1, interpolation='nearest', aspect='auto')
            ax.set_axis_off()
            
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, orientation='vertical')
            cbar.set_label('NDWI', fontsize=12, fontweight='bold')
            cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
            cbar.set_ticklabels(['-1.0', '-0.5', '0.0', '0.5', '1.0'])
            
            ax.set_title(f'NDWI - Farm {farm_id} - {image_date}', fontsize=14, fontweight='bold', pad=10)
            plt.tight_layout()
            
            # Save to bytes buffer and upload to MinIO
            ndwi_buffer = io.BytesIO()
            fig.savefig(ndwi_buffer, format='png', dpi=150, bbox_inches='tight', facecolor='white')
            plt.close(fig)
            ndwi_buffer.seek(0)
            
            ndwi_object_name = f"ndwi/farm_{farm_id}_{image_date}_ndwi.png"
            minio_client.put_object(
                bucket_name,
                ndwi_object_name,
                ndwi_buffer,
                length=ndwi_buffer.getbuffer().nbytes,
                content_type='image/png'
            )
            
            results['ndwi_image_path'] = f"{bucket_name}/{ndwi_object_name}"
            results['ndwi_image_url'] = f"/api/analysis-image/?path={ndwi_object_name}"
            results['ndwi_mean'] = round(float(np.nanmean(ndwi)), 4)
        
        results['success'] = True
        
    except Exception as e:
        results['error'] = str(e)
    
    return results


def generate_ndvi_from_minio(farm_id: int, image_date: str, output_path: str) -> dict:
    """
    Generate NDVI/NDWI visualization images from MinIO-stored bands.
    
    Args:
        farm_id: Farm ID
        image_date: Image date (YYYY-MM-DD)
        output_path: Base media path for output
    
    Returns:
        Dict with paths to generated images
    """
    import matplotlib.pyplot as plt
    import tempfile
    
    try:
        from minio import Minio
    except ImportError:
        return {'success': False, 'error': 'minio package not installed'}
    
    results = {
        'success': False,
        'ndvi_image_path': None,
        'ndwi_image_path': None,
        'ndvi_image_url': None,
        'ndwi_image_url': None,
        'error': None
    }
    
    try:
        minio_client = Minio(
            os.getenv('MINIO_ENDPOINT', 'localhost:9000'),
            access_key=os.getenv('MINIO_ACCESS_KEY', 'minioadmin'),
            secret_key=os.getenv('MINIO_SECRET_KEY', 'minioadmin'),
            secure=False
        )
        
        bucket = 'satellite-imagery'
        prefix = f"farm_{farm_id}/{image_date}/"
        
        # List and download required bands
        objects = list(minio_client.list_objects(bucket, prefix=prefix))
        
        nir_data = None
        red_data = None
        green_data = None
        
        for obj in objects:
            obj_name = obj.object_name.lower()
            if 'nir' in obj_name or 'b08' in obj_name:
                response = minio_client.get_object(bucket, obj.object_name)
                nir_data = response.read()
                response.close()
                response.release_conn()
            elif 'red' in obj_name or 'b04' in obj_name:
                response = minio_client.get_object(bucket, obj.object_name)
                red_data = response.read()
                response.close()
                response.release_conn()
            elif 'green' in obj_name or 'b03' in obj_name:
                response = minio_client.get_object(bucket, obj.object_name)
                green_data = response.read()
                response.close()
                response.release_conn()
        
        if not nir_data or not red_data:
            results['error'] = 'Required NIR and RED bands not found in MinIO'
            return results
        
        # Write to temp files and process
        import rasterio
        
        with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as nir_tmp:
            nir_tmp.write(nir_data)
            nir_path = nir_tmp.name
        
        with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as red_tmp:
            red_tmp.write(red_data)
            red_path = red_tmp.name
        
        green_path = None
        if green_data:
            with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as green_tmp:
                green_tmp.write(green_data)
                green_path = green_tmp.name
        
        # Read bands
        with rasterio.open(nir_path) as nir_src:
            nir_band = nir_src.read(1).astype(float)
        
        with rasterio.open(red_path) as red_src:
            red_band = red_src.read(1).astype(float)
        
        # Calculate NDVI
        ndvi = calculate_ndvi_from_bands(nir_band, red_band)
        
        # Create output directories
        ndvi_output_dir = os.path.join(output_path, 'results', 'ndvi')
        ndwi_output_dir = os.path.join(output_path, 'results', 'ndwi')
        os.makedirs(ndvi_output_dir, exist_ok=True)
        os.makedirs(ndwi_output_dir, exist_ok=True)
        
        # Generate NDVI image
        fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
        cmap = plt.cm.RdYlGn
        ndvi_norm = (ndvi + 1) / 2
        
        im = ax.imshow(ndvi_norm, cmap=cmap, vmin=0, vmax=1, interpolation='nearest', aspect='auto')
        ax.set_axis_off()
        
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('NDVI', fontsize=12, fontweight='bold')
        cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
        cbar.set_ticklabels(['-1.0', '-0.5', '0.0', '0.5', '1.0'])
        
        ax.set_title(f'NDVI - Farm {farm_id} - {image_date}', fontsize=14, fontweight='bold', pad=10)
        plt.tight_layout()
        
        ndvi_filename = f"farm_{farm_id}_{image_date}_ndvi.png"
        ndvi_full_path = os.path.join(ndvi_output_dir, ndvi_filename)
        fig.savefig(ndvi_full_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        
        results['ndvi_image_path'] = ndvi_full_path
        results['ndvi_image_url'] = f"/media/results/ndvi/{ndvi_filename}"
        
        # Generate NDWI image if green available
        if green_path:
            with rasterio.open(green_path) as green_src:
                green_band = green_src.read(1).astype(float)
            
            ndwi = calculate_ndwi_from_bands(green_band, nir_band)
            
            fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
            cmap = plt.cm.Blues_r
            ndwi_norm = (ndwi + 1) / 2
            
            im = ax.imshow(ndwi_norm, cmap=cmap, vmin=0, vmax=1, interpolation='nearest', aspect='auto')
            ax.set_axis_off()
            
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('NDWI', fontsize=12, fontweight='bold')
            cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
            cbar.set_ticklabels(['-1.0', '-0.5', '0.0', '0.5', '1.0'])
            
            ax.set_title(f'NDWI - Farm {farm_id} - {image_date}', fontsize=14, fontweight='bold', pad=10)
            plt.tight_layout()
            
            ndwi_filename = f"farm_{farm_id}_{image_date}_ndwi.png"
            ndwi_full_path = os.path.join(ndwi_output_dir, ndwi_filename)
            fig.savefig(ndwi_full_path, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close(fig)
            
            results['ndwi_image_path'] = ndwi_full_path
            results['ndwi_image_url'] = f"/media/results/ndwi/{ndwi_filename}"
            
            os.unlink(green_path)
        
        # Cleanup temp files
        os.unlink(nir_path)
        os.unlink(red_path)
        
        results['success'] = True
        
    except Exception as e:
        results['error'] = str(e)
    
    return results
