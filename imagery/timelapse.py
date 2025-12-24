"""
Timelapse Generator

Create timelapse visualizations from:
- NDVI time-series
- Satellite imagery sequences
- SpaceEye processed imagery
"""

import os
import numpy as np
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import logging
from PIL import Image, ImageDraw, ImageFont
import tempfile

logger = logging.getLogger(__name__)

# Check for video libraries
try:
    import imageio
    IMAGEIO_AVAILABLE = True
except ImportError:
    IMAGEIO_AVAILABLE = False
    logger.warning("imageio not installed. Install with: pip install imageio[ffmpeg]")


class TimelapseGenerator:
    """
    Generate timelapse videos/GIFs from image sequences.
    """
    
    def __init__(self, output_dir: str = 'media/timelapse'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def create_gif(self,
                   images: List[np.ndarray],
                   output_name: str,
                   dates: List[datetime] = None,
                   duration: float = 0.5,
                   loop: int = 0,
                   add_timestamp: bool = True,
                   add_colorbar: bool = False,
                   title: str = None) -> Dict:
        """
        Create an animated GIF from image sequence.
        
        Args:
            images: List of images as numpy arrays
            output_name: Output filename (without extension)
            dates: List of dates for timestamps
            duration: Duration per frame in seconds
            loop: Number of loops (0 = infinite)
            add_timestamp: Whether to add date overlay
            add_colorbar: Whether to add NDVI colorbar
            title: Optional title text
            
        Returns:
            Result with output path
        """
        if not images:
            return {'error': 'No images provided'}
        
        if not IMAGEIO_AVAILABLE:
            return {'error': 'imageio not installed'}
        
        # Process images
        processed_frames = []
        
        for i, img in enumerate(images):
            # Convert to PIL Image
            if img.dtype != np.uint8:
                # Normalize to 0-255
                img_norm = ((img - img.min()) / (img.max() - img.min() + 1e-10) * 255).astype(np.uint8)
            else:
                img_norm = img
            
            # Handle different array shapes
            if len(img_norm.shape) == 2:
                # Grayscale - convert to RGB
                pil_img = Image.fromarray(img_norm, mode='L').convert('RGB')
            elif img_norm.shape[2] == 4:
                # RGBA
                pil_img = Image.fromarray(img_norm, mode='RGBA').convert('RGB')
            else:
                # RGB
                pil_img = Image.fromarray(img_norm, mode='RGB')
            
            # Add overlays
            if add_timestamp and dates and i < len(dates):
                pil_img = self._add_timestamp(pil_img, dates[i])
            
            if title and i == 0:
                pil_img = self._add_title(pil_img, title)
            
            if add_colorbar:
                pil_img = self._add_colorbar(pil_img)
            
            processed_frames.append(np.array(pil_img))
        
        # Create GIF
        output_path = os.path.join(self.output_dir, f"{output_name}.gif")
        
        try:
            imageio.mimsave(
                output_path,
                processed_frames,
                duration=duration,
                loop=loop
            )
            
            return {
                'status': 'success',
                'output_path': output_path,
                'num_frames': len(processed_frames),
                'duration_total': len(processed_frames) * duration
            }
            
        except Exception as e:
            logger.error(f"Failed to create GIF: {e}")
            return {'error': str(e)}
    
    def create_video(self,
                     images: List[np.ndarray],
                     output_name: str,
                     dates: List[datetime] = None,
                     fps: int = 2,
                     codec: str = 'libx264',
                     add_timestamp: bool = True,
                     title: str = None) -> Dict:
        """
        Create an MP4 video from image sequence.
        
        Args:
            images: List of images as numpy arrays
            output_name: Output filename (without extension)
            dates: List of dates for timestamps
            fps: Frames per second
            codec: Video codec
            add_timestamp: Whether to add date overlay
            title: Optional title text
            
        Returns:
            Result with output path
        """
        if not images:
            return {'error': 'No images provided'}
        
        if not IMAGEIO_AVAILABLE:
            return {'error': 'imageio not installed'}
        
        # Ensure consistent frame sizes
        target_size = images[0].shape[:2]
        
        processed_frames = []
        
        for i, img in enumerate(images):
            # Resize if needed
            if img.shape[:2] != target_size:
                pil_img = Image.fromarray(img)
                pil_img = pil_img.resize((target_size[1], target_size[0]))
                img = np.array(pil_img)
            
            # Normalize
            if img.dtype != np.uint8:
                img = ((img - img.min()) / (img.max() - img.min() + 1e-10) * 255).astype(np.uint8)
            
            # Convert to RGB if needed
            if len(img.shape) == 2:
                img = np.stack([img, img, img], axis=2)
            elif img.shape[2] == 4:
                img = img[:, :, :3]
            
            # Add overlays
            pil_img = Image.fromarray(img)
            
            if add_timestamp and dates and i < len(dates):
                pil_img = self._add_timestamp(pil_img, dates[i])
            
            if title:
                pil_img = self._add_title(pil_img, title)
            
            processed_frames.append(np.array(pil_img))
        
        # Create video
        output_path = os.path.join(self.output_dir, f"{output_name}.mp4")
        
        try:
            writer = imageio.get_writer(
                output_path,
                fps=fps,
                codec=codec,
                quality=8
            )
            
            for frame in processed_frames:
                writer.append_data(frame)
            
            writer.close()
            
            return {
                'status': 'success',
                'output_path': output_path,
                'num_frames': len(processed_frames),
                'fps': fps,
                'duration_seconds': len(processed_frames) / fps
            }
            
        except Exception as e:
            logger.error(f"Failed to create video: {e}")
            return {'error': str(e)}
    
    def _add_timestamp(self, 
                       image: Image.Image, 
                       date: datetime,
                       position: str = 'bottom_right') -> Image.Image:
        """
        Add timestamp overlay to image.
        """
        draw = ImageDraw.Draw(image)
        
        # Format date
        date_str = date.strftime('%Y-%m-%d')
        
        # Get font (use default if custom not available)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
        except:
            font = ImageFont.load_default()
        
        # Calculate position
        bbox = draw.textbbox((0, 0), date_str, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        padding = 10
        
        if position == 'bottom_right':
            x = image.width - text_width - padding
            y = image.height - text_height - padding
        elif position == 'bottom_left':
            x = padding
            y = image.height - text_height - padding
        elif position == 'top_right':
            x = image.width - text_width - padding
            y = padding
        else:
            x = padding
            y = padding
        
        # Draw background rectangle
        draw.rectangle(
            [x - 5, y - 5, x + text_width + 5, y + text_height + 5],
            fill=(0, 0, 0, 180)
        )
        
        # Draw text
        draw.text((x, y), date_str, fill=(255, 255, 255), font=font)
        
        return image
    
    def _add_title(self, 
                   image: Image.Image,
                   title: str) -> Image.Image:
        """
        Add title to image.
        """
        draw = ImageDraw.Draw(image)
        
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
        except:
            font = ImageFont.load_default()
        
        bbox = draw.textbbox((0, 0), title, font=font)
        text_width = bbox[2] - bbox[0]
        
        x = (image.width - text_width) // 2
        y = 10
        
        # Draw background
        draw.rectangle(
            [x - 10, y - 5, x + text_width + 10, y + 30],
            fill=(0, 0, 0, 180)
        )
        
        # Draw text
        draw.text((x, y), title, fill=(255, 255, 255), font=font)
        
        return image
    
    def _add_colorbar(self, 
                      image: Image.Image,
                      label: str = 'NDVI') -> Image.Image:
        """
        Add NDVI colorbar to image.
        """
        # Create colorbar
        colorbar_width = 20
        colorbar_height = 150
        
        # NDVI colormap (red -> yellow -> green)
        colorbar = np.zeros((colorbar_height, colorbar_width, 3), dtype=np.uint8)
        
        for i in range(colorbar_height):
            val = 1 - i / colorbar_height  # 1 at top, 0 at bottom
            
            if val < 0.5:
                # Red to Yellow
                r = 255
                g = int(val * 2 * 255)
                b = 0
            else:
                # Yellow to Green
                r = int((1 - (val - 0.5) * 2) * 255)
                g = 255
                b = 0
            
            colorbar[i, :] = [r, g, b]
        
        # Paste colorbar
        colorbar_img = Image.fromarray(colorbar)
        
        x = image.width - colorbar_width - 30
        y = (image.height - colorbar_height) // 2
        
        image.paste(colorbar_img, (x, y))
        
        # Add labels
        draw = ImageDraw.Draw(image)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
        except:
            font = ImageFont.load_default()
        
        draw.text((x - 5, y - 20), label, fill=(255, 255, 255), font=font)
        draw.text((x + colorbar_width + 5, y), '1.0', fill=(255, 255, 255), font=font)
        draw.text((x + colorbar_width + 5, y + colorbar_height - 15), '0.0', fill=(255, 255, 255), font=font)
        
        return image


class NDVITimelapseGenerator:
    """
    Generate timelapse specifically for NDVI data.
    """
    
    def __init__(self):
        self.generator = TimelapseGenerator()
    
    def ndvi_to_rgb(self, ndvi: np.ndarray) -> np.ndarray:
        """
        Convert NDVI array to RGB visualization.
        
        Uses standard NDVI colormap:
        - < 0: Blue (water)
        - 0-0.2: Brown (bare soil)
        - 0.2-0.4: Yellow-Green (sparse vegetation)
        - 0.4-0.6: Light Green (moderate vegetation)
        - 0.6-0.8: Green (dense vegetation)
        - > 0.8: Dark Green (very dense vegetation)
        """
        rgb = np.zeros((*ndvi.shape, 3), dtype=np.uint8)
        
        # Water (NDVI < 0)
        water = ndvi < 0
        rgb[water] = [0, 0, 139]  # Dark blue
        
        # Bare soil (0-0.2)
        bare = (ndvi >= 0) & (ndvi < 0.2)
        rgb[bare] = [139, 69, 19]  # Brown
        
        # Sparse vegetation (0.2-0.4)
        sparse = (ndvi >= 0.2) & (ndvi < 0.4)
        rgb[sparse] = [154, 205, 50]  # Yellow-green
        
        # Moderate vegetation (0.4-0.6)
        moderate = (ndvi >= 0.4) & (ndvi < 0.6)
        rgb[moderate] = [34, 139, 34]  # Forest green
        
        # Dense vegetation (0.6-0.8)
        dense = (ndvi >= 0.6) & (ndvi < 0.8)
        rgb[dense] = [0, 100, 0]  # Dark green
        
        # Very dense vegetation (> 0.8)
        very_dense = ndvi >= 0.8
        rgb[very_dense] = [0, 50, 0]  # Very dark green
        
        return rgb
    
    def create_ndvi_timelapse(self,
                              ndvi_arrays: List[np.ndarray],
                              dates: List[datetime],
                              output_name: str,
                              format: str = 'gif') -> Dict:
        """
        Create timelapse from NDVI arrays.
        
        Args:
            ndvi_arrays: List of NDVI arrays
            dates: Corresponding dates
            output_name: Output filename
            format: 'gif' or 'mp4'
            
        Returns:
            Result with output path
        """
        # Convert NDVI to RGB
        rgb_frames = [self.ndvi_to_rgb(ndvi) for ndvi in ndvi_arrays]
        
        if format == 'gif':
            return self.generator.create_gif(
                rgb_frames,
                output_name,
                dates=dates,
                add_timestamp=True,
                add_colorbar=True,
                title='NDVI Time-Series'
            )
        else:
            return self.generator.create_video(
                rgb_frames,
                output_name,
                dates=dates,
                add_timestamp=True,
                title='NDVI Time-Series'
            )


def create_field_timelapse(field_id: int,
                           start_date: datetime = None,
                           end_date: datetime = None,
                           format: str = 'gif') -> Dict:
    """
    Create timelapse for a specific field.
    
    Args:
        field_id: Field ID
        start_date: Start of period
        end_date: End of period
        format: 'gif' or 'mp4'
        
    Returns:
        Timelapse result
    """
    from core.models import FieldBoundary
    from analytics.models import AnalyticsResult
    
    try:
        field = FieldBoundary.objects.get(id=field_id)
    except FieldBoundary.DoesNotExist:
        return {'error': f'Field {field_id} not found'}
    
    # Get NDVI results
    query = AnalyticsResult.objects.filter(
        field=field,
        avg_ndvi__isnull=False
    ).order_by('analysis_date')
    
    if start_date:
        query = query.filter(analysis_date__gte=start_date)
    if end_date:
        query = query.filter(analysis_date__lte=end_date)
    
    results = list(query)
    
    if len(results) < 2:
        return {
            'error': 'Not enough data points for timelapse',
            'data_points': len(results)
        }
    
    # For now, create a simple visualization
    # In production, would load actual NDVI rasters
    
    dates = [r.analysis_date for r in results]
    ndvi_values = [r.avg_ndvi for r in results]
    
    # Create simple NDVI bar visualization for each date
    width, height = 400, 300
    frames = []
    
    for i, (date, ndvi) in enumerate(zip(dates, ndvi_values)):
        # Create frame
        frame = np.ones((height, width, 3), dtype=np.uint8) * 255
        
        # Draw NDVI bar
        bar_height = int(ndvi * (height - 50))
        bar_width = 100
        bar_x = (width - bar_width) // 2
        bar_y = height - 30 - bar_height
        
        # Color based on NDVI
        if ndvi < 0.3:
            color = [139, 69, 19]  # Brown
        elif ndvi < 0.5:
            color = [154, 205, 50]  # Yellow-green
        elif ndvi < 0.7:
            color = [34, 139, 34]  # Forest green
        else:
            color = [0, 100, 0]  # Dark green
        
        frame[bar_y:height-30, bar_x:bar_x+bar_width] = color
        frames.append(frame)
    
    generator = TimelapseGenerator()
    
    if format == 'gif':
        return generator.create_gif(
            frames,
            f"field_{field_id}_timelapse",
            dates=dates,
            title=f"NDVI: {field.name}"
        )
    else:
        return generator.create_video(
            frames,
            f"field_{field_id}_timelapse",
            dates=dates,
            title=f"NDVI: {field.name}"
        )
