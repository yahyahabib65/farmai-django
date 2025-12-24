"""
Spectral Fusion / Extension

Implements:
- Pan-sharpening (high-res drone + multispectral Sentinel)
- Spectral extension (RGB drone to multispectral)
- Resolution enhancement
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy.ndimage import zoom, gaussian_filter
from sklearn.linear_model import LinearRegression
import logging

logger = logging.getLogger(__name__)


class SpectralFusion:
    """
    Fuse high-resolution drone imagery with lower-resolution 
    multispectral satellite imagery.
    
    Methods:
    1. Brovey Transform
    2. IHS (Intensity-Hue-Saturation)
    3. Wavelet-based fusion
    4. Regression-based spectral extension
    """
    
    def __init__(self):
        pass
    
    def brovey_transform(self,
                         high_res_rgb: np.ndarray,
                         low_res_ms: np.ndarray) -> np.ndarray:
        """
        Brovey Transform for pan-sharpening.
        
        Fuses high-resolution RGB with low-resolution multispectral
        to create high-resolution multispectral.
        
        Args:
            high_res_rgb: High-resolution RGB image (H, W, 3)
            low_res_ms: Low-resolution multispectral (h, w, bands)
            
        Returns:
            Fused high-resolution multispectral image
        """
        # Upsample low-res multispectral to match high-res
        scale_y = high_res_rgb.shape[0] / low_res_ms.shape[0]
        scale_x = high_res_rgb.shape[1] / low_res_ms.shape[1]
        
        upsampled_ms = np.zeros((
            high_res_rgb.shape[0],
            high_res_rgb.shape[1],
            low_res_ms.shape[2]
        ))
        
        for i in range(low_res_ms.shape[2]):
            upsampled_ms[:, :, i] = zoom(
                low_res_ms[:, :, i],
                (scale_y, scale_x),
                order=1  # Bilinear interpolation
            )
        
        # Calculate pan band from RGB (intensity)
        pan_high = np.mean(high_res_rgb, axis=2)
        pan_low = np.mean(upsampled_ms[:, :, :3], axis=2)  # Assume first 3 are RGB
        
        # Avoid division by zero
        pan_low = np.maximum(pan_low, 1e-10)
        
        # Apply Brovey transform
        ratio = pan_high / pan_low
        
        fused = np.zeros_like(upsampled_ms)
        for i in range(upsampled_ms.shape[2]):
            fused[:, :, i] = upsampled_ms[:, :, i] * ratio
        
        return np.clip(fused, 0, 1)
    
    def ihs_fusion(self,
                   high_res_rgb: np.ndarray,
                   low_res_ms: np.ndarray) -> np.ndarray:
        """
        IHS (Intensity-Hue-Saturation) fusion.
        
        Args:
            high_res_rgb: High-resolution RGB (H, W, 3)
            low_res_ms: Low-resolution multispectral (h, w, bands)
            
        Returns:
            Fused image
        """
        # Upsample low-res
        scale_y = high_res_rgb.shape[0] / low_res_ms.shape[0]
        scale_x = high_res_rgb.shape[1] / low_res_ms.shape[1]
        
        upsampled = np.zeros((
            high_res_rgb.shape[0],
            high_res_rgb.shape[1],
            low_res_ms.shape[2]
        ))
        
        for i in range(low_res_ms.shape[2]):
            upsampled[:, :, i] = zoom(
                low_res_ms[:, :, i],
                (scale_y, scale_x),
                order=1
            )
        
        # Convert upsampled RGB to IHS
        r, g, b = upsampled[:, :, 0], upsampled[:, :, 1], upsampled[:, :, 2]
        
        # IHS transformation
        intensity = (r + g + b) / 3
        
        # Calculate v1, v2 (color components)
        v1 = (-r - g + 2*b) / np.sqrt(6)
        v2 = (r - g) / np.sqrt(2)
        
        # Replace intensity with high-res intensity
        new_intensity = np.mean(high_res_rgb.astype(float) / 255, axis=2)
        
        # Match histogram
        intensity_flat = intensity.flatten()
        new_intensity_flat = new_intensity.flatten()
        
        if np.std(intensity_flat) > 0:
            new_intensity = (
                (new_intensity - np.mean(new_intensity_flat)) * 
                (np.std(intensity_flat) / np.std(new_intensity_flat)) +
                np.mean(intensity_flat)
            )
        
        # Inverse IHS transformation
        fused = np.zeros_like(upsampled)
        fused[:, :, 0] = new_intensity - v1/np.sqrt(6) + v2/np.sqrt(2)  # R
        fused[:, :, 1] = new_intensity - v1/np.sqrt(6) - v2/np.sqrt(2)  # G
        fused[:, :, 2] = new_intensity + 2*v1/np.sqrt(6)  # B
        
        # Process additional bands using intensity ratio
        if upsampled.shape[2] > 3:
            ratio = new_intensity / np.maximum(intensity, 1e-10)
            for i in range(3, upsampled.shape[2]):
                fused[:, :, i] = upsampled[:, :, i] * ratio
        
        return np.clip(fused, 0, 1)
    
    def wavelet_fusion(self,
                       high_res: np.ndarray,
                       low_res: np.ndarray,
                       levels: int = 3) -> np.ndarray:
        """
        Wavelet-based fusion.
        
        Uses high-frequency details from high-res image with
        spectral content from low-res image.
        
        Args:
            high_res: High-resolution image (grayscale or single band)
            low_res: Low-resolution image (same band)
            levels: Number of decomposition levels
            
        Returns:
            Fused image
        """
        try:
            import pywt
        except ImportError:
            logger.warning("PyWavelets not installed, using simple fusion")
            return self._simple_fusion(high_res, low_res)
        
        # Upsample low-res to match high-res
        scale_y = high_res.shape[0] / low_res.shape[0]
        scale_x = high_res.shape[1] / low_res.shape[1]
        upsampled = zoom(low_res, (scale_y, scale_x), order=1)
        
        # Wavelet decomposition
        coeffs_high = pywt.wavedec2(high_res, 'haar', level=levels)
        coeffs_low = pywt.wavedec2(upsampled, 'haar', level=levels)
        
        # Fuse: use high-frequency from high-res, low-frequency from low-res
        fused_coeffs = [coeffs_low[0]]  # Use low-res approximation
        
        for i in range(1, len(coeffs_high)):
            # Use high-res details
            fused_coeffs.append(coeffs_high[i])
        
        # Reconstruct
        fused = pywt.waverec2(fused_coeffs, 'haar')
        
        # Match size
        fused = fused[:high_res.shape[0], :high_res.shape[1]]
        
        return np.clip(fused, 0, 1)
    
    def _simple_fusion(self, 
                       high_res: np.ndarray,
                       low_res: np.ndarray) -> np.ndarray:
        """Simple fusion fallback when wavelets not available."""
        scale_y = high_res.shape[0] / low_res.shape[0]
        scale_x = high_res.shape[1] / low_res.shape[1]
        upsampled = zoom(low_res, (scale_y, scale_x), order=1)
        
        # High-pass from high-res
        high_pass = high_res - gaussian_filter(high_res, sigma=5)
        
        # Add to upsampled low-res
        fused = upsampled + high_pass * 0.5
        
        return np.clip(fused, 0, 1)


class SpectralExtension:
    """
    Extend RGB imagery to estimate additional spectral bands.
    
    Uses regression models trained on paired RGB-multispectral data
    to predict NIR, Red Edge, etc. from RGB.
    """
    
    def __init__(self):
        self.models = {}
        self.fitted = False
    
    def fit(self,
            rgb_images: List[np.ndarray],
            ms_images: List[np.ndarray],
            band_names: List[str] = None) -> Dict:
        """
        Train regression models to predict spectral bands from RGB.
        
        Args:
            rgb_images: List of RGB images (H, W, 3)
            ms_images: List of corresponding multispectral images (H, W, bands)
            band_names: Names for additional bands beyond RGB
            
        Returns:
            Training metrics
        """
        if not rgb_images or not ms_images:
            return {'error': 'No training data provided'}
        
        if len(rgb_images) != len(ms_images):
            return {'error': 'RGB and MS image counts must match'}
        
        n_bands = ms_images[0].shape[2]
        
        if band_names is None:
            band_names = [f'band_{i}' for i in range(n_bands)]
        
        # Collect training samples
        X_all = []
        Y_all = {i: [] for i in range(n_bands)}
        
        for rgb, ms in zip(rgb_images, ms_images):
            # Ensure same size
            if rgb.shape[:2] != ms.shape[:2]:
                # Resize ms to match rgb
                scale_y = rgb.shape[0] / ms.shape[0]
                scale_x = rgb.shape[1] / ms.shape[1]
                ms_resized = np.zeros((rgb.shape[0], rgb.shape[1], n_bands))
                for i in range(n_bands):
                    ms_resized[:, :, i] = zoom(ms[:, :, i], (scale_y, scale_x), order=1)
                ms = ms_resized
            
            # Flatten for regression
            rgb_flat = rgb.reshape(-1, 3)
            X_all.append(rgb_flat)
            
            for i in range(n_bands):
                Y_all[i].append(ms[:, :, i].flatten())
        
        X = np.vstack(X_all)
        
        # Subsample if too large
        max_samples = 100000
        if len(X) > max_samples:
            indices = np.random.choice(len(X), max_samples, replace=False)
            X = X[indices]
            for i in range(n_bands):
                Y_all[i] = [np.concatenate(Y_all[i])[indices]]
        
        # Train models for each band
        metrics = {}
        
        for i in range(n_bands):
            y = np.concatenate(Y_all[i]) if isinstance(Y_all[i], list) else Y_all[i]
            
            if i < 3:
                # RGB bands - skip (we have these)
                self.models[i] = None
                continue
            
            model = LinearRegression()
            model.fit(X, y)
            
            score = model.score(X, y)
            
            self.models[i] = model
            metrics[band_names[i]] = {
                'r2_score': float(score),
                'coefficients': model.coef_.tolist(),
                'intercept': float(model.intercept_)
            }
        
        self.fitted = True
        self.band_names = band_names
        
        return {
            'status': 'fitted',
            'n_bands': n_bands,
            'n_samples': len(X),
            'band_metrics': metrics
        }
    
    def extend(self, rgb_image: np.ndarray) -> np.ndarray:
        """
        Extend RGB image to multispectral.
        
        Args:
            rgb_image: RGB image (H, W, 3)
            
        Returns:
            Extended multispectral image (H, W, n_bands)
        """
        if not self.fitted:
            # Use empirical relationships for NIR estimation
            return self._empirical_extension(rgb_image)
        
        h, w = rgb_image.shape[:2]
        rgb_flat = rgb_image.reshape(-1, 3)
        
        n_bands = len(self.models)
        result = np.zeros((h, w, n_bands))
        
        # Copy RGB bands
        result[:, :, :3] = rgb_image[:, :, :3]
        
        # Predict additional bands
        for i in range(3, n_bands):
            if self.models[i] is not None:
                predicted = self.models[i].predict(rgb_flat)
                result[:, :, i] = predicted.reshape(h, w)
        
        return np.clip(result, 0, 1)
    
    def _empirical_extension(self, rgb_image: np.ndarray) -> np.ndarray:
        """
        Empirical NIR estimation from RGB.
        
        Uses vegetation indices and empirical relationships.
        """
        h, w = rgb_image.shape[:2]
        
        # Normalize to 0-1
        if rgb_image.max() > 1:
            rgb = rgb_image.astype(float) / 255
        else:
            rgb = rgb_image.astype(float)
        
        r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
        
        # Estimate NIR using empirical relationships
        # Based on vegetation reflectance patterns
        
        # Excess Green Index
        egi = 2 * g - r - b
        
        # Estimated NIR (simplified)
        # Vegetation: high NIR, high green, low red
        # Non-vegetation: NIR similar to visible
        
        nir_veg = 0.5 + 0.3 * egi  # Vegetation NIR
        nir_nonveg = (r + g + b) / 3  # Non-vegetation NIR
        
        # Blend based on greenness
        veg_mask = egi > 0
        nir = np.where(veg_mask, nir_veg, nir_nonveg)
        
        # Estimate Red Edge (between Red and NIR)
        red_edge = (r + nir) / 2
        
        # Create extended image
        extended = np.zeros((h, w, 5))
        extended[:, :, 0] = r  # Red
        extended[:, :, 1] = g  # Green
        extended[:, :, 2] = b  # Blue
        extended[:, :, 3] = red_edge  # Red Edge (estimated)
        extended[:, :, 4] = nir  # NIR (estimated)
        
        return np.clip(extended, 0, 1)
    
    def calculate_indices(self, extended_image: np.ndarray) -> Dict:
        """
        Calculate vegetation indices from extended image.
        
        Args:
            extended_image: Extended multispectral image (H, W, 5+)
            
        Returns:
            Dictionary of vegetation indices
        """
        r = extended_image[:, :, 0]
        g = extended_image[:, :, 1]
        b = extended_image[:, :, 2]
        
        if extended_image.shape[2] >= 5:
            re = extended_image[:, :, 3]  # Red Edge
            nir = extended_image[:, :, 4]  # NIR
        else:
            # Estimate if not available
            nir = np.mean(extended_image, axis=2) + 0.2
            re = (r + nir) / 2
        
        # Avoid division by zero
        eps = 1e-10
        
        # NDVI
        ndvi = (nir - r) / (nir + r + eps)
        
        # NDRE (Red Edge NDVI)
        ndre = (nir - re) / (nir + re + eps)
        
        # GNDVI (Green NDVI)
        gndvi = (nir - g) / (nir + g + eps)
        
        # SAVI (Soil Adjusted)
        L = 0.5
        savi = ((nir - r) / (nir + r + L + eps)) * (1 + L)
        
        # EVI (Enhanced)
        evi = 2.5 * ((nir - r) / (nir + 6*r - 7.5*b + 1 + eps))
        
        return {
            'ndvi': np.clip(ndvi, -1, 1),
            'ndre': np.clip(ndre, -1, 1),
            'gndvi': np.clip(gndvi, -1, 1),
            'savi': np.clip(savi, -1, 1),
            'evi': np.clip(evi, -1, 1)
        }


def fuse_drone_sentinel(drone_image_path: str,
                        sentinel_image_path: str,
                        method: str = 'brovey') -> Dict:
    """
    Fuse high-res drone with Sentinel-2 imagery.
    
    Args:
        drone_image_path: Path to drone RGB image
        sentinel_image_path: Path to Sentinel multispectral
        method: 'brovey', 'ihs', or 'wavelet'
        
    Returns:
        Fused image result
    """
    from PIL import Image
    import rasterio
    
    # Load drone image
    try:
        drone = np.array(Image.open(drone_image_path))
        if drone.max() > 1:
            drone = drone.astype(float) / 255
    except Exception as e:
        return {'error': f'Failed to load drone image: {e}'}
    
    # Load Sentinel image
    try:
        with rasterio.open(sentinel_image_path) as src:
            sentinel = src.read()
            # Transpose to (H, W, bands)
            sentinel = np.transpose(sentinel, (1, 2, 0))
            if sentinel.max() > 1:
                sentinel = sentinel.astype(float) / 10000  # Sentinel scaling
    except Exception as e:
        return {'error': f'Failed to load Sentinel image: {e}'}
    
    # Perform fusion
    fusion = SpectralFusion()
    
    if method == 'brovey':
        fused = fusion.brovey_transform(drone, sentinel)
    elif method == 'ihs':
        fused = fusion.ihs_fusion(drone, sentinel)
    elif method == 'wavelet':
        # Apply wavelet to each band
        fused = np.zeros((drone.shape[0], drone.shape[1], sentinel.shape[2]))
        for i in range(sentinel.shape[2]):
            pan = np.mean(drone, axis=2) if i < 3 else drone[:, :, min(i, 2)]
            fused[:, :, i] = fusion.wavelet_fusion(pan, sentinel[:, :, i])
    else:
        return {'error': f'Unknown method: {method}'}
    
    return {
        'status': 'success',
        'method': method,
        'drone_shape': drone.shape,
        'sentinel_shape': sentinel.shape,
        'fused_shape': fused.shape,
        'fused_image': fused
    }
