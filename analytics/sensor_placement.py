"""
Optimal Sensor Placement

Implements algorithms to find optimal locations for IoT sensors
in agricultural fields based on spatial variability.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from scipy.ndimage import gaussian_filter
import logging

logger = logging.getLogger(__name__)


class OptimalSensorPlacement:
    """
    Find optimal sensor placement locations using spatial analysis.
    
    Methods:
    1. K-means clustering on NDVI variability
    2. Maximum variance locations
    3. Stratified random sampling
    4. Grid-based optimal coverage
    """
    
    def __init__(self, n_sensors: int = 5):
        """
        Initialize with target number of sensors.
        
        Args:
            n_sensors: Number of sensors to place
        """
        self.n_sensors = n_sensors
    
    def find_optimal_locations_kmeans(self,
                                      raster: np.ndarray,
                                      mask: np.ndarray = None,
                                      geo_transform: Tuple = None) -> Dict:
        """
        Find optimal sensor locations using K-means clustering.
        
        The algorithm:
        1. Calculate local variance to identify high-variability zones
        2. Use K-means to cluster these zones
        3. Select cluster centers as sensor locations
        
        Args:
            raster: NDVI or other raster array
            mask: Optional mask for valid area
            geo_transform: (x_origin, pixel_width, 0, y_origin, 0, pixel_height)
            
        Returns:
            Optimal locations with metadata
        """
        if mask is None:
            mask = np.ones_like(raster, dtype=bool)
        
        # Calculate local variance (5x5 window)
        from scipy.ndimage import generic_filter
        
        def local_variance(values):
            return np.var(values) if len(values) > 0 else 0
        
        variance_map = generic_filter(raster, local_variance, size=5)
        variance_map[~mask] = 0
        
        # Get coordinates of high-variance pixels
        threshold = np.percentile(variance_map[mask], 70)
        high_var_coords = np.argwhere(variance_map > threshold)
        
        if len(high_var_coords) < self.n_sensors:
            # Not enough high-variance points, use all valid points
            high_var_coords = np.argwhere(mask)
        
        if len(high_var_coords) < self.n_sensors:
            return {
                'error': 'Not enough valid pixels for sensor placement',
                'valid_pixels': len(high_var_coords)
            }
        
        # K-means clustering
        kmeans = KMeans(n_clusters=self.n_sensors, random_state=42, n_init=10)
        kmeans.fit(high_var_coords)
        
        # Get cluster centers (row, col format)
        centers = kmeans.cluster_centers_.astype(int)
        
        # Convert to geographic coordinates if transform provided
        if geo_transform:
            x_origin, pixel_width, _, y_origin, _, pixel_height = geo_transform
            locations = [
                {
                    'sensor_id': i + 1,
                    'pixel_row': int(c[0]),
                    'pixel_col': int(c[1]),
                    'latitude': y_origin + c[0] * pixel_height,
                    'longitude': x_origin + c[1] * pixel_width,
                    'local_variance': float(variance_map[c[0], c[1]]),
                    'ndvi_value': float(raster[c[0], c[1]])
                }
                for i, c in enumerate(centers)
            ]
        else:
            locations = [
                {
                    'sensor_id': i + 1,
                    'pixel_row': int(c[0]),
                    'pixel_col': int(c[1]),
                    'local_variance': float(variance_map[c[0], c[1]]),
                    'ndvi_value': float(raster[c[0], c[1]])
                }
                for i, c in enumerate(centers)
            ]
        
        return {
            'method': 'kmeans',
            'n_sensors': self.n_sensors,
            'locations': locations,
            'coverage_score': self._calculate_coverage(centers, mask),
            'variance_map_stats': {
                'mean': float(np.mean(variance_map[mask])),
                'max': float(np.max(variance_map[mask])),
                'threshold_used': float(threshold)
            }
        }
    
    def find_optimal_locations_maxvar(self,
                                      raster: np.ndarray,
                                      mask: np.ndarray = None,
                                      min_distance: int = 20) -> Dict:
        """
        Find locations with maximum variance, maintaining minimum distance.
        
        Args:
            raster: NDVI or other raster array
            mask: Optional mask for valid area
            min_distance: Minimum pixel distance between sensors
            
        Returns:
            Optimal locations
        """
        if mask is None:
            mask = np.ones_like(raster, dtype=bool)
        
        # Calculate local variance
        variance_map = gaussian_filter(np.abs(raster - np.mean(raster)), sigma=3)
        variance_map[~mask] = 0
        
        locations = []
        remaining_mask = mask.copy()
        
        for i in range(self.n_sensors):
            if not np.any(remaining_mask):
                break
            
            # Find maximum variance location
            masked_variance = np.where(remaining_mask, variance_map, -np.inf)
            max_idx = np.unravel_index(np.argmax(masked_variance), variance_map.shape)
            
            locations.append({
                'sensor_id': i + 1,
                'pixel_row': int(max_idx[0]),
                'pixel_col': int(max_idx[1]),
                'local_variance': float(variance_map[max_idx]),
                'ndvi_value': float(raster[max_idx])
            })
            
            # Mask out area around selected location
            rows, cols = np.ogrid[:raster.shape[0], :raster.shape[1]]
            distance = np.sqrt((rows - max_idx[0])**2 + (cols - max_idx[1])**2)
            remaining_mask[distance < min_distance] = False
        
        return {
            'method': 'maximum_variance',
            'n_sensors': len(locations),
            'min_distance_pixels': min_distance,
            'locations': locations
        }
    
    def find_optimal_locations_stratified(self,
                                          raster: np.ndarray,
                                          mask: np.ndarray = None,
                                          n_strata: int = None) -> Dict:
        """
        Stratified sampling based on NDVI values.
        
        Divides field into strata based on NDVI range and places
        sensors in each stratum.
        
        Args:
            raster: NDVI array
            mask: Valid area mask
            n_strata: Number of strata (defaults to n_sensors)
            
        Returns:
            Optimal locations
        """
        if mask is None:
            mask = np.ones_like(raster, dtype=bool)
        
        n_strata = n_strata or self.n_sensors
        
        # Get NDVI range
        valid_values = raster[mask]
        percentiles = np.linspace(0, 100, n_strata + 1)
        thresholds = np.percentile(valid_values, percentiles)
        
        locations = []
        sensors_per_stratum = max(1, self.n_sensors // n_strata)
        
        for i in range(n_strata):
            # Define stratum
            lower = thresholds[i]
            upper = thresholds[i + 1]
            
            stratum_mask = mask & (raster >= lower) & (raster <= upper)
            
            if not np.any(stratum_mask):
                continue
            
            # Find representative points in stratum
            stratum_coords = np.argwhere(stratum_mask)
            
            if len(stratum_coords) < sensors_per_stratum:
                n_select = len(stratum_coords)
            else:
                n_select = sensors_per_stratum
            
            # Use K-means within stratum
            if len(stratum_coords) > n_select:
                kmeans = KMeans(n_clusters=n_select, random_state=42, n_init=10)
                kmeans.fit(stratum_coords)
                selected = kmeans.cluster_centers_.astype(int)
            else:
                selected = stratum_coords
            
            for j, coord in enumerate(selected):
                locations.append({
                    'sensor_id': len(locations) + 1,
                    'pixel_row': int(coord[0]),
                    'pixel_col': int(coord[1]),
                    'stratum': i + 1,
                    'ndvi_range': f'{lower:.2f}-{upper:.2f}',
                    'ndvi_value': float(raster[coord[0], coord[1]])
                })
        
        return {
            'method': 'stratified',
            'n_sensors': len(locations),
            'n_strata': n_strata,
            'locations': locations
        }
    
    def find_optimal_locations_grid(self,
                                    raster: np.ndarray,
                                    mask: np.ndarray = None) -> Dict:
        """
        Find optimal grid-based placement for maximum coverage.
        
        Args:
            raster: NDVI array (used for mask and values)
            mask: Valid area mask
            
        Returns:
            Grid-optimized locations
        """
        if mask is None:
            mask = np.ones_like(raster, dtype=bool)
        
        # Calculate approximate grid spacing
        valid_area = np.sum(mask)
        area_per_sensor = valid_area / self.n_sensors
        spacing = int(np.sqrt(area_per_sensor))
        
        # Generate grid points
        rows = np.arange(spacing // 2, raster.shape[0], spacing)
        cols = np.arange(spacing // 2, raster.shape[1], spacing)
        
        grid_points = []
        for r in rows:
            for c in cols:
                if r < raster.shape[0] and c < raster.shape[1] and mask[r, c]:
                    grid_points.append((r, c))
        
        # If too many points, select subset with highest variance
        if len(grid_points) > self.n_sensors:
            # Sort by local variance
            variances = []
            for r, c in grid_points:
                r_min = max(0, r - 5)
                r_max = min(raster.shape[0], r + 5)
                c_min = max(0, c - 5)
                c_max = min(raster.shape[1], c + 5)
                local_var = np.var(raster[r_min:r_max, c_min:c_max])
                variances.append(local_var)
            
            # Select top n_sensors
            sorted_indices = np.argsort(variances)[::-1][:self.n_sensors]
            grid_points = [grid_points[i] for i in sorted_indices]
        
        locations = [
            {
                'sensor_id': i + 1,
                'pixel_row': int(r),
                'pixel_col': int(c),
                'ndvi_value': float(raster[r, c])
            }
            for i, (r, c) in enumerate(grid_points)
        ]
        
        return {
            'method': 'grid_optimized',
            'n_sensors': len(locations),
            'grid_spacing': spacing,
            'locations': locations,
            'coverage_score': self._calculate_coverage(
                np.array([(l['pixel_row'], l['pixel_col']) for l in locations]),
                mask
            )
        }
    
    def _calculate_coverage(self, 
                            locations: np.ndarray,
                            mask: np.ndarray) -> float:
        """
        Calculate coverage score for sensor placement.
        
        Score is based on how well the sensors cover the field area.
        """
        if len(locations) == 0:
            return 0.0
        
        # Get all valid coordinates
        valid_coords = np.argwhere(mask)
        
        if len(valid_coords) == 0:
            return 0.0
        
        # Calculate minimum distance from each valid pixel to nearest sensor
        distances = cdist(valid_coords, locations)
        min_distances = np.min(distances, axis=1)
        
        # Coverage score: inverse of mean minimum distance, normalized
        max_possible_distance = np.sqrt(mask.shape[0]**2 + mask.shape[1]**2)
        mean_min_distance = np.mean(min_distances)
        
        coverage = 1 - (mean_min_distance / max_possible_distance)
        
        return float(coverage)
    
    def optimize_placement(self,
                          raster: np.ndarray,
                          mask: np.ndarray = None,
                          method: str = 'auto') -> Dict:
        """
        Find optimal sensor placement using best method.
        
        Args:
            raster: NDVI or other raster
            mask: Valid area mask
            method: 'kmeans', 'maxvar', 'stratified', 'grid', or 'auto'
            
        Returns:
            Best placement result
        """
        if method == 'auto':
            # Try all methods and select best coverage
            results = {}
            
            results['kmeans'] = self.find_optimal_locations_kmeans(raster, mask)
            results['maxvar'] = self.find_optimal_locations_maxvar(raster, mask)
            results['stratified'] = self.find_optimal_locations_stratified(raster, mask)
            results['grid'] = self.find_optimal_locations_grid(raster, mask)
            
            # Select method with best coverage
            best_method = max(
                results.keys(),
                key=lambda k: results[k].get('coverage_score', 0)
            )
            
            result = results[best_method]
            result['all_methods_compared'] = {
                k: v.get('coverage_score', 0) for k, v in results.items()
            }
            
            return result
        
        elif method == 'kmeans':
            return self.find_optimal_locations_kmeans(raster, mask)
        elif method == 'maxvar':
            return self.find_optimal_locations_maxvar(raster, mask)
        elif method == 'stratified':
            return self.find_optimal_locations_stratified(raster, mask)
        elif method == 'grid':
            return self.find_optimal_locations_grid(raster, mask)
        else:
            return {'error': f'Unknown method: {method}'}


def find_sensor_locations_for_field(field_id: int, 
                                     n_sensors: int = 5,
                                     method: str = 'auto') -> Dict:
    """
    Find optimal sensor placement for a specific field.
    
    Args:
        field_id: Field ID
        n_sensors: Number of sensors to place
        method: Optimization method
        
    Returns:
        Optimal locations with geographic coordinates
    """
    from core.models import FieldBoundary
    from analytics.models import AnalyticsResult
    
    try:
        field = FieldBoundary.objects.get(id=field_id)
    except FieldBoundary.DoesNotExist:
        return {'error': f'Field {field_id} not found'}
    
    # Get latest NDVI result
    latest_result = AnalyticsResult.objects.filter(
        field=field,
        avg_ndvi__isnull=False
    ).order_by('-analysis_date').first()
    
    if not latest_result:
        return {
            'error': 'No NDVI data available for field',
            'note': 'Run NDVI analysis first'
        }
    
    # For full implementation, would load NDVI raster from storage
    # For now, create synthetic raster based on available data
    
    # Get field extent
    bounds = field.boundary.extent  # (xmin, ymin, xmax, ymax)
    
    # Create simple variability map
    # In production, use actual NDVI raster
    raster_size = 100
    np.random.seed(field_id)  # Reproducible for same field
    
    base_ndvi = latest_result.avg_ndvi or 0.5
    synthetic_raster = np.random.normal(base_ndvi, 0.15, (raster_size, raster_size))
    synthetic_raster = np.clip(synthetic_raster, 0, 1)
    
    # Add some spatial structure
    synthetic_raster = gaussian_filter(synthetic_raster, sigma=5)
    
    # Create geo transform
    x_range = bounds[2] - bounds[0]
    y_range = bounds[3] - bounds[1]
    geo_transform = (
        bounds[0],  # x origin
        x_range / raster_size,  # pixel width
        0,
        bounds[3],  # y origin
        0,
        -y_range / raster_size  # pixel height (negative for north-up)
    )
    
    # Find optimal locations
    optimizer = OptimalSensorPlacement(n_sensors=n_sensors)
    result = optimizer.optimize_placement(
        synthetic_raster,
        method=method
    )
    
    # Add geographic coordinates to results
    if 'locations' in result:
        for loc in result['locations']:
            row = loc['pixel_row']
            col = loc['pixel_col']
            loc['latitude'] = geo_transform[3] + row * geo_transform[5]
            loc['longitude'] = geo_transform[0] + col * geo_transform[1]
    
    result['field_id'] = field_id
    result['field_name'] = field.name
    result['field_bounds'] = {
        'xmin': bounds[0],
        'ymin': bounds[1],
        'xmax': bounds[2],
        'ymax': bounds[3]
    }
    
    return result
