"""
Segment Anything Model (SAM) Processor

Implements:
- Field boundary segmentation using SAM
- Crop segmentation from Sentinel-2 imagery
- Automatic point prompt generation
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from PIL import Image
import logging
import os

logger = logging.getLogger(__name__)

# Check if SAM is available
SAM_AVAILABLE = False
try:
    from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
    SAM_AVAILABLE = True
except ImportError:
    logger.warning("segment-anything not installed. Install with: pip install segment-anything")


class SAMProcessor:
    """
    Segment Anything Model processor for agricultural imagery.
    
    Supports:
    - Automatic segmentation (no prompts)
    - Point-prompted segmentation
    - Box-prompted segmentation
    """
    
    MODEL_URLS = {
        'vit_h': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth',
        'vit_l': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth',
        'vit_b': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth'
    }
    
    def __init__(self, 
                 model_type: str = 'vit_b',
                 checkpoint_path: str = None,
                 device: str = 'cpu'):
        """
        Initialize SAM processor.
        
        Args:
            model_type: SAM model variant ('vit_h', 'vit_l', 'vit_b')
            checkpoint_path: Path to model checkpoint
            device: 'cpu' or 'cuda'
        """
        self.model_type = model_type
        self.device = device
        self.checkpoint_path = checkpoint_path or f'models/sam_{model_type}.pth'
        self.predictor = None
        self.mask_generator = None
        self._model_loaded = False
    
    def load_model(self) -> bool:
        """
        Load SAM model from checkpoint.
        
        Returns:
            True if model loaded successfully
        """
        if not SAM_AVAILABLE:
            logger.error("segment-anything package not installed")
            return False
        
        if not os.path.exists(self.checkpoint_path):
            logger.warning(f"Checkpoint not found at {self.checkpoint_path}")
            logger.info(f"Download from: {self.MODEL_URLS.get(self.model_type)}")
            return False
        
        try:
            sam = sam_model_registry[self.model_type](checkpoint=self.checkpoint_path)
            sam.to(device=self.device)
            
            self.predictor = SamPredictor(sam)
            self.mask_generator = SamAutomaticMaskGenerator(
                sam,
                points_per_side=32,
                pred_iou_thresh=0.86,
                stability_score_thresh=0.92,
                crop_n_layers=1,
                crop_n_points_downscale_factor=2,
                min_mask_region_area=100
            )
            
            self._model_loaded = True
            logger.info(f"SAM model {self.model_type} loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load SAM model: {e}")
            return False
    
    def segment_automatic(self, image: np.ndarray) -> Dict:
        """
        Automatically segment all objects in image.
        
        Args:
            image: RGB image as numpy array (H, W, 3)
            
        Returns:
            Segmentation results with masks
        """
        if not self._model_loaded:
            return {'error': 'Model not loaded. Call load_model() first.'}
        
        if not SAM_AVAILABLE:
            return self._fallback_segmentation(image)
        
        try:
            masks = self.mask_generator.generate(image)
            
            # Sort by area
            masks = sorted(masks, key=lambda x: x['area'], reverse=True)
            
            return {
                'num_segments': len(masks),
                'segments': [
                    {
                        'id': i,
                        'area': int(m['area']),
                        'bbox': m['bbox'],  # [x, y, w, h]
                        'predicted_iou': float(m['predicted_iou']),
                        'stability_score': float(m['stability_score']),
                        'mask_shape': list(m['segmentation'].shape)
                    }
                    for i, m in enumerate(masks[:50])  # Limit to top 50
                ],
                'masks': [m['segmentation'] for m in masks[:50]]
            }
            
        except Exception as e:
            logger.error(f"Automatic segmentation failed: {e}")
            return {'error': str(e)}
    
    def segment_with_points(self, 
                           image: np.ndarray,
                           points: List[Tuple[int, int]],
                           labels: List[int] = None) -> Dict:
        """
        Segment using point prompts.
        
        Args:
            image: RGB image as numpy array
            points: List of (x, y) coordinates
            labels: 1 for foreground, 0 for background
            
        Returns:
            Segmentation mask
        """
        if not self._model_loaded:
            return {'error': 'Model not loaded'}
        
        if not SAM_AVAILABLE:
            return self._fallback_segmentation(image)
        
        if labels is None:
            labels = [1] * len(points)
        
        try:
            self.predictor.set_image(image)
            
            points_np = np.array(points)
            labels_np = np.array(labels)
            
            masks, scores, logits = self.predictor.predict(
                point_coords=points_np,
                point_labels=labels_np,
                multimask_output=True
            )
            
            # Get best mask
            best_idx = np.argmax(scores)
            best_mask = masks[best_idx]
            
            return {
                'mask': best_mask,
                'score': float(scores[best_idx]),
                'all_masks': masks,
                'all_scores': scores.tolist(),
                'area': int(np.sum(best_mask))
            }
            
        except Exception as e:
            logger.error(f"Point-prompted segmentation failed: {e}")
            return {'error': str(e)}
    
    def segment_with_box(self, 
                        image: np.ndarray,
                        box: Tuple[int, int, int, int]) -> Dict:
        """
        Segment using bounding box prompt.
        
        Args:
            image: RGB image as numpy array
            box: (x1, y1, x2, y2) bounding box
            
        Returns:
            Segmentation mask
        """
        if not self._model_loaded:
            return {'error': 'Model not loaded'}
        
        if not SAM_AVAILABLE:
            return self._fallback_segmentation(image)
        
        try:
            self.predictor.set_image(image)
            
            box_np = np.array(box)
            
            masks, scores, logits = self.predictor.predict(
                box=box_np,
                multimask_output=True
            )
            
            best_idx = np.argmax(scores)
            best_mask = masks[best_idx]
            
            return {
                'mask': best_mask,
                'score': float(scores[best_idx]),
                'area': int(np.sum(best_mask))
            }
            
        except Exception as e:
            logger.error(f"Box-prompted segmentation failed: {e}")
            return {'error': str(e)}
    
    def _fallback_segmentation(self, image: np.ndarray) -> Dict:
        """
        Fallback segmentation using simple thresholding when SAM not available.
        """
        logger.warning("Using fallback segmentation (NDVI-based)")
        
        # Simple vegetation segmentation using color
        if len(image.shape) == 3 and image.shape[2] >= 3:
            # Calculate Excess Green Index
            r = image[:, :, 0].astype(float)
            g = image[:, :, 1].astype(float)
            b = image[:, :, 2].astype(float)
            
            egi = 2 * g - r - b
            
            # Threshold
            vegetation_mask = egi > np.percentile(egi, 70)
            
            return {
                'mask': vegetation_mask,
                'method': 'fallback_egi',
                'area': int(np.sum(vegetation_mask)),
                'note': 'SAM not available, using EGI thresholding'
            }
        
        return {'error': 'Invalid image format'}


class FieldBoundarySegmenter:
    """
    Segment agricultural field boundaries from satellite imagery.
    """
    
    def __init__(self, sam_processor: SAMProcessor = None):
        self.sam = sam_processor or SAMProcessor()
    
    def segment_fields(self, 
                      image: np.ndarray,
                      min_field_area: int = 1000,
                      max_field_area: int = None) -> Dict:
        """
        Segment field boundaries from satellite image.
        
        Args:
            image: RGB satellite image
            min_field_area: Minimum field area in pixels
            max_field_area: Maximum field area in pixels
            
        Returns:
            Field segmentation results
        """
        if not self.sam._model_loaded:
            loaded = self.sam.load_model()
            if not loaded:
                # Use fallback
                return self._edge_based_segmentation(image, min_field_area)
        
        # Get automatic segmentation
        result = self.sam.segment_automatic(image)
        
        if 'error' in result:
            return result
        
        # Filter by area
        fields = []
        for i, seg in enumerate(result['segments']):
            area = seg['area']
            if area >= min_field_area:
                if max_field_area is None or area <= max_field_area:
                    fields.append({
                        'field_id': i,
                        'area_pixels': area,
                        'bbox': seg['bbox'],
                        'confidence': seg['predicted_iou']
                    })
        
        return {
            'num_fields': len(fields),
            'fields': fields,
            'image_shape': list(image.shape[:2]),
            'masks': result.get('masks', [])
        }
    
    def _edge_based_segmentation(self, 
                                  image: np.ndarray,
                                  min_area: int) -> Dict:
        """
        Fallback edge-based field segmentation.
        """
        from scipy import ndimage
        from skimage import filters, segmentation, measure
        
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image
        
        # Edge detection
        edges = filters.sobel(gray)
        
        # Threshold edges
        edge_mask = edges > np.percentile(edges, 80)
        
        # Close gaps
        edge_mask = ndimage.binary_dilation(edge_mask, iterations=2)
        edge_mask = ndimage.binary_erosion(edge_mask, iterations=1)
        
        # Invert to get fields
        field_mask = ~edge_mask
        
        # Label connected components
        labeled, num_features = ndimage.label(field_mask)
        
        # Get properties
        regions = measure.regionprops(labeled)
        
        fields = []
        for i, region in enumerate(regions):
            if region.area >= min_area:
                fields.append({
                    'field_id': i,
                    'area_pixels': int(region.area),
                    'bbox': list(region.bbox),
                    'centroid': list(region.centroid),
                    'confidence': 0.5  # Lower confidence for fallback
                })
        
        return {
            'num_fields': len(fields),
            'fields': fields,
            'image_shape': list(image.shape[:2]),
            'method': 'edge_detection_fallback'
        }


class CropSegmenter:
    """
    Segment different crop types from multispectral imagery.
    """
    
    def __init__(self, sam_processor: SAMProcessor = None):
        self.sam = sam_processor or SAMProcessor()
    
    def segment_crops(self,
                     image: np.ndarray,
                     ndvi: np.ndarray = None) -> Dict:
        """
        Segment crop regions from imagery.
        
        Uses NDVI to identify crop areas, then SAM to refine boundaries.
        
        Args:
            image: RGB image
            ndvi: Optional NDVI array (same dimensions)
            
        Returns:
            Crop segmentation results
        """
        # Generate NDVI-based points for SAM prompts
        if ndvi is not None:
            # Find high-NDVI regions as crop centers
            crop_mask = ndvi > 0.4
            points = self._sample_points_from_mask(crop_mask, n_points=10)
        else:
            # Sample from green regions
            if len(image.shape) == 3:
                green_ratio = image[:, :, 1] / (np.sum(image, axis=2) + 1e-10)
                crop_mask = green_ratio > 0.4
                points = self._sample_points_from_mask(crop_mask, n_points=10)
            else:
                points = []
        
        if not points:
            # Fall back to automatic segmentation
            return self.sam.segment_automatic(image)
        
        if not self.sam._model_loaded:
            self.sam.load_model()
        
        # Segment each crop region
        results = []
        for point in points:
            seg = self.sam.segment_with_points(image, [point], [1])
            if 'mask' in seg:
                results.append({
                    'center_point': point,
                    'area': seg['area'],
                    'score': seg['score']
                })
        
        return {
            'num_crop_regions': len(results),
            'regions': results
        }
    
    def _sample_points_from_mask(self, 
                                  mask: np.ndarray,
                                  n_points: int = 10) -> List[Tuple[int, int]]:
        """
        Sample points from a binary mask.
        """
        from scipy import ndimage
        
        # Label connected components
        labeled, num_features = ndimage.label(mask)
        
        points = []
        for i in range(1, min(num_features + 1, n_points + 1)):
            coords = np.where(labeled == i)
            if len(coords[0]) > 0:
                # Get centroid
                cy = int(np.mean(coords[0]))
                cx = int(np.mean(coords[1]))
                points.append((cx, cy))
        
        return points


def segment_field_from_sentinel(field_id: int, 
                                image_date: str = None) -> Dict:
    """
    Segment a field using SAM from Sentinel-2 imagery.
    
    Args:
        field_id: ID of the field
        image_date: Optional specific image date
        
    Returns:
        Segmentation results
    """
    from core.models import FieldBoundary
    from imagery.models import SatelliteImage
    
    try:
        field = FieldBoundary.objects.get(id=field_id)
    except FieldBoundary.DoesNotExist:
        return {'error': f'Field {field_id} not found'}
    
    # Get satellite image
    query = SatelliteImage.objects.filter(farm=field.farm)
    if image_date:
        query = query.filter(acquisition_date=image_date)
    
    sat_image = query.order_by('-acquisition_date').first()
    
    if not sat_image:
        return {'error': 'No satellite imagery available'}
    
    # Load RGB bands
    # This would need to read from MinIO/storage
    # For now, return structure
    
    return {
        'field_id': field_id,
        'field_name': field.name,
        'image_date': sat_image.acquisition_date.isoformat() if sat_image.acquisition_date else None,
        'status': 'ready_for_segmentation',
        'note': 'Load image from storage and call SAMProcessor'
    }
