"""
NDVI Time-Series Analysis Module

Implements:
- Crop cycle detection (peak detection in NDVI time-series)
- Harvest and germination period detection
- Land degradation trend analysis
"""

import numpy as np
from scipy import signal
from scipy import stats
from scipy.interpolate import interp1d
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class NDVITimeSeries:
    """
    Analyzes NDVI time-series data to detect crop cycles,
    harvest/germination periods, and land degradation trends.
    """
    
    def __init__(self, dates: List[datetime], ndvi_values: List[float]):
        """
        Initialize with time-series data.
        
        Args:
            dates: List of datetime objects for each observation
            ndvi_values: Corresponding NDVI values (0-1 range)
        """
        self.dates = np.array(dates)
        self.ndvi_values = np.array(ndvi_values)
        
        # Convert dates to days since start for numerical analysis
        self.days = np.array([(d - dates[0]).days for d in dates])
        
        # Remove NaN values
        valid_mask = ~np.isnan(self.ndvi_values)
        self.dates = self.dates[valid_mask]
        self.ndvi_values = self.ndvi_values[valid_mask]
        self.days = self.days[valid_mask]
    
    def interpolate(self, interval_days: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Interpolate NDVI values to regular intervals.
        
        Args:
            interval_days: Days between interpolated points
            
        Returns:
            Tuple of (interpolated_days, interpolated_ndvi)
        """
        if len(self.days) < 2:
            return self.days, self.ndvi_values
            
        # Create interpolation function
        f = interp1d(self.days, self.ndvi_values, kind='cubic', 
                     fill_value='extrapolate')
        
        # Generate regular intervals
        new_days = np.arange(self.days[0], self.days[-1], interval_days)
        new_ndvi = f(new_days)
        
        # Clip to valid NDVI range
        new_ndvi = np.clip(new_ndvi, 0, 1)
        
        return new_days, new_ndvi
    
    def smooth(self, window_size: int = 15) -> np.ndarray:
        """
        Apply Savitzky-Golay filter to smooth NDVI time-series.
        
        Args:
            window_size: Size of smoothing window (must be odd)
            
        Returns:
            Smoothed NDVI values
        """
        if len(self.ndvi_values) < window_size:
            return self.ndvi_values
            
        # Ensure window size is odd
        if window_size % 2 == 0:
            window_size += 1
            
        smoothed = signal.savgol_filter(
            self.ndvi_values, 
            window_length=window_size,
            polyorder=3
        )
        
        return np.clip(smoothed, 0, 1)
    
    def detect_crop_cycles(self, 
                          min_cycle_length_days: int = 60,
                          min_peak_prominence: float = 0.1,
                          min_peak_height: float = 0.3) -> Dict:
        """
        Detect number of crop cycles in the time-series using peak detection.
        
        A crop cycle is identified by:
        1. A peak in NDVI (maximum vegetation)
        2. Followed by a trough (harvest/bare soil)
        
        Args:
            min_cycle_length_days: Minimum days between crop cycles
            min_peak_prominence: Minimum prominence of peaks
            min_peak_height: Minimum NDVI value for peaks
            
        Returns:
            Dictionary with cycle information
        """
        if len(self.ndvi_values) < 10:
            return {
                'num_cycles': 0,
                'peaks': [],
                'troughs': [],
                'cycle_durations': [],
                'error': 'Insufficient data points'
            }
        
        # Interpolate and smooth
        interp_days, interp_ndvi = self.interpolate(interval_days=5)
        
        if len(interp_ndvi) < 15:
            smoothed = interp_ndvi
        else:
            smoothed = signal.savgol_filter(interp_ndvi, 15, 3)
            smoothed = np.clip(smoothed, 0, 1)
        
        # Detect peaks (crop maturity)
        min_distance = min_cycle_length_days // 5  # In terms of interpolated points
        peaks, peak_props = signal.find_peaks(
            smoothed,
            height=min_peak_height,
            prominence=min_peak_prominence,
            distance=min_distance
        )
        
        # Detect troughs (harvest/bare soil)
        troughs, trough_props = signal.find_peaks(
            -smoothed,
            prominence=min_peak_prominence * 0.5,
            distance=min_distance
        )
        
        # Convert back to actual dates
        start_date = self.dates[0]
        peak_dates = [start_date + timedelta(days=int(interp_days[p])) for p in peaks]
        trough_dates = [start_date + timedelta(days=int(interp_days[t])) for t in troughs]
        
        # Calculate cycle durations
        cycle_durations = []
        for i in range(len(peaks) - 1):
            duration = interp_days[peaks[i + 1]] - interp_days[peaks[i]]
            cycle_durations.append(int(duration))
        
        # Determine crop cycle pattern
        num_cycles = len(peaks)
        
        return {
            'num_cycles': num_cycles,
            'peaks': [
                {
                    'date': d.isoformat() if hasattr(d, 'isoformat') else str(d),
                    'ndvi': float(smoothed[p]),
                    'day': int(interp_days[p])
                }
                for d, p in zip(peak_dates, peaks)
            ],
            'troughs': [
                {
                    'date': d.isoformat() if hasattr(d, 'isoformat') else str(d),
                    'ndvi': float(smoothed[t]),
                    'day': int(interp_days[t])
                }
                for d, t in zip(trough_dates, troughs)
            ],
            'cycle_durations': cycle_durations,
            'avg_cycle_duration': float(np.mean(cycle_durations)) if cycle_durations else None,
            'cropping_intensity': self._classify_cropping_intensity(num_cycles),
            'smoothed_ndvi': smoothed.tolist(),
            'interpolated_days': interp_days.tolist()
        }
    
    def _classify_cropping_intensity(self, num_cycles: int) -> str:
        """Classify cropping intensity based on number of cycles per year."""
        if num_cycles == 0:
            return 'fallow'
        elif num_cycles == 1:
            return 'single_crop'
        elif num_cycles == 2:
            return 'double_crop'
        elif num_cycles == 3:
            return 'triple_crop'
        else:
            return 'intensive_cropping'
    
    def detect_harvest_germination(self, 
                                   ndvi_threshold_germination: float = 0.25,
                                   ndvi_threshold_harvest: float = 0.3,
                                   rate_threshold: float = 0.01) -> Dict:
        """
        Detect harvest and germination periods from NDVI time-series.
        
        Germination: Rapid increase in NDVI from low baseline
        Harvest: Rapid decrease in NDVI from high values
        
        Args:
            ndvi_threshold_germination: NDVI value below which germination starts
            ndvi_threshold_harvest: NDVI value above which harvest occurs
            rate_threshold: Minimum rate of change (NDVI/day)
            
        Returns:
            Dictionary with harvest and germination periods
        """
        if len(self.ndvi_values) < 5:
            return {
                'germination_periods': [],
                'harvest_periods': [],
                'error': 'Insufficient data points'
            }
        
        # Interpolate for regular intervals
        interp_days, interp_ndvi = self.interpolate(interval_days=5)
        
        # Calculate rate of change (derivative)
        rate_of_change = np.gradient(interp_ndvi, interp_days)
        
        # Smooth the rate of change
        if len(rate_of_change) > 7:
            rate_of_change = signal.savgol_filter(rate_of_change, 7, 2)
        
        germination_periods = []
        harvest_periods = []
        start_date = self.dates[0]
        
        # Detect germination (increasing NDVI from low values)
        in_germination = False
        germ_start = None
        
        for i, (day, ndvi, rate) in enumerate(zip(interp_days, interp_ndvi, rate_of_change)):
            if not in_germination and ndvi < ndvi_threshold_germination and rate > rate_threshold:
                in_germination = True
                germ_start = day
            elif in_germination and (rate < rate_threshold * 0.5 or ndvi > 0.6):
                in_germination = False
                if germ_start is not None:
                    germination_periods.append({
                        'start_date': (start_date + timedelta(days=int(germ_start))).isoformat(),
                        'end_date': (start_date + timedelta(days=int(day))).isoformat(),
                        'duration_days': int(day - germ_start),
                        'start_ndvi': float(interp_ndvi[np.argmin(np.abs(interp_days - germ_start))]),
                        'end_ndvi': float(ndvi)
                    })
        
        # Detect harvest (decreasing NDVI from high values)
        in_harvest = False
        harvest_start = None
        
        for i, (day, ndvi, rate) in enumerate(zip(interp_days, interp_ndvi, rate_of_change)):
            if not in_harvest and ndvi > ndvi_threshold_harvest and rate < -rate_threshold:
                in_harvest = True
                harvest_start = day
            elif in_harvest and (rate > -rate_threshold * 0.5 or ndvi < 0.2):
                in_harvest = False
                if harvest_start is not None:
                    harvest_periods.append({
                        'start_date': (start_date + timedelta(days=int(harvest_start))).isoformat(),
                        'end_date': (start_date + timedelta(days=int(day))).isoformat(),
                        'duration_days': int(day - harvest_start),
                        'start_ndvi': float(interp_ndvi[np.argmin(np.abs(interp_days - harvest_start))]),
                        'end_ndvi': float(ndvi)
                    })
        
        return {
            'germination_periods': germination_periods,
            'harvest_periods': harvest_periods,
            'num_germinations': len(germination_periods),
            'num_harvests': len(harvest_periods),
            'growing_season_summary': self._summarize_growing_seasons(
                germination_periods, harvest_periods
            )
        }
    
    def _summarize_growing_seasons(self, germination_periods: List, harvest_periods: List) -> Dict:
        """Summarize growing seasons from germination and harvest data."""
        if not germination_periods and not harvest_periods:
            return {'status': 'no_growing_seasons_detected'}
        
        seasons = []
        
        # Match germination to harvest periods
        for i, germ in enumerate(germination_periods):
            # Find next harvest after this germination
            germ_end = datetime.fromisoformat(germ['end_date'])
            
            matching_harvest = None
            for harv in harvest_periods:
                harv_start = datetime.fromisoformat(harv['start_date'])
                if harv_start > germ_end:
                    matching_harvest = harv
                    break
            
            if matching_harvest:
                season_length = (
                    datetime.fromisoformat(matching_harvest['end_date']) - 
                    datetime.fromisoformat(germ['start_date'])
                ).days
                
                seasons.append({
                    'season': i + 1,
                    'germination_start': germ['start_date'],
                    'harvest_end': matching_harvest['end_date'],
                    'total_duration_days': season_length
                })
        
        return {
            'num_complete_seasons': len(seasons),
            'seasons': seasons,
            'avg_season_length': np.mean([s['total_duration_days'] for s in seasons]) if seasons else None
        }
    
    def detect_land_degradation(self, 
                                significance_level: float = 0.05) -> Dict:
        """
        Detect land degradation trends using linear regression on NDVI time-series.
        
        Negative trend indicates potential land degradation.
        
        Args:
            significance_level: P-value threshold for trend significance
            
        Returns:
            Dictionary with degradation analysis
        """
        if len(self.ndvi_values) < 10:
            return {
                'trend': 'insufficient_data',
                'slope': None,
                'error': 'Need at least 10 observations'
            }
        
        # Linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            self.days, self.ndvi_values
        )
        
        # Mann-Kendall trend test for robustness
        mk_result = self._mann_kendall_test(self.ndvi_values)
        
        # Annual change rate
        annual_change = slope * 365  # NDVI change per year
        
        # Classify degradation
        if p_value > significance_level:
            trend_status = 'no_significant_trend'
            degradation_level = 'stable'
        elif slope < -0.02:
            trend_status = 'decreasing'
            degradation_level = 'severe_degradation'
        elif slope < -0.01:
            trend_status = 'decreasing'
            degradation_level = 'moderate_degradation'
        elif slope < 0:
            trend_status = 'slightly_decreasing'
            degradation_level = 'mild_degradation'
        elif slope > 0.02:
            trend_status = 'increasing'
            degradation_level = 'significant_improvement'
        elif slope > 0.01:
            trend_status = 'increasing'
            degradation_level = 'moderate_improvement'
        else:
            trend_status = 'slightly_increasing'
            degradation_level = 'mild_improvement'
        
        # Calculate residual variability
        predicted = intercept + slope * self.days
        residuals = self.ndvi_values - predicted
        variability = np.std(residuals)
        
        return {
            'trend': trend_status,
            'degradation_level': degradation_level,
            'slope': float(slope),
            'annual_change': float(annual_change),
            'intercept': float(intercept),
            'r_squared': float(r_value ** 2),
            'p_value': float(p_value),
            'is_significant': p_value < significance_level,
            'std_error': float(std_err),
            'mann_kendall': mk_result,
            'residual_variability': float(variability),
            'recommendation': self._get_degradation_recommendation(degradation_level),
            'trend_line': {
                'start': float(intercept),
                'end': float(intercept + slope * self.days[-1])
            }
        }
    
    def _mann_kendall_test(self, data: np.ndarray) -> Dict:
        """
        Perform Mann-Kendall trend test.
        
        Non-parametric test for monotonic trend detection.
        """
        n = len(data)
        s = 0
        
        for i in range(n - 1):
            for j in range(i + 1, n):
                diff = data[j] - data[i]
                if diff > 0:
                    s += 1
                elif diff < 0:
                    s -= 1
        
        # Calculate variance
        var_s = (n * (n - 1) * (2 * n + 5)) / 18
        
        # Calculate Z-score
        if s > 0:
            z = (s - 1) / np.sqrt(var_s)
        elif s < 0:
            z = (s + 1) / np.sqrt(var_s)
        else:
            z = 0
        
        # Two-tailed p-value
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))
        
        if z > 0:
            trend = 'increasing'
        elif z < 0:
            trend = 'decreasing'
        else:
            trend = 'no_trend'
        
        return {
            'statistic_s': int(s),
            'z_score': float(z),
            'p_value': float(p_value),
            'trend': trend
        }
    
    def _get_degradation_recommendation(self, level: str) -> str:
        """Get recommendation based on degradation level."""
        recommendations = {
            'severe_degradation': 'Immediate intervention required. Consider soil restoration, cover cropping, and reduced tillage.',
            'moderate_degradation': 'Monitor closely. Implement conservation practices and soil health improvements.',
            'mild_degradation': 'Early warning. Review farming practices and consider preventive measures.',
            'stable': 'Land condition is stable. Continue current management practices.',
            'mild_improvement': 'Positive trend observed. Current practices are effective.',
            'moderate_improvement': 'Good improvement. Consider documenting practices for replication.',
            'significant_improvement': 'Excellent improvement. Land is recovering well.'
        }
        return recommendations.get(level, 'Unable to determine recommendation.')


def analyze_field_ndvi_timeseries(field_id: int, 
                                   start_date: datetime = None,
                                   end_date: datetime = None) -> Dict:
    """
    Analyze NDVI time-series for a specific field.
    
    Args:
        field_id: ID of the field to analyze
        start_date: Start of analysis period
        end_date: End of analysis period
        
    Returns:
        Complete time-series analysis results
    """
    from imagery.models import ImageReading
    from core.models import FieldBoundary
    
    try:
        field = FieldBoundary.objects.get(id=field_id)
    except FieldBoundary.DoesNotExist:
        return {'error': f'Field {field_id} not found'}
    
    # Get NDVI/NDWI data from ImageReading (real satellite data from MinIO)
    queryset = ImageReading.objects.filter(
        field=field,
        ndvi_mean__isnull=False
    ).order_by('acquisition_date')
    
    if start_date:
        queryset = queryset.filter(acquisition_date__gte=start_date)
    if end_date:
        queryset = queryset.filter(acquisition_date__lte=end_date)
    
    results = list(queryset.values('acquisition_date', 'ndvi_mean', 'ndwi_mean'))
    
    if len(results) < 5:
        return {
            'field_id': field_id,
            'field_name': field.name,
            'error': 'Insufficient NDVI data for time-series analysis',
            'data_points': len(results)
        }
    
    dates = [r['acquisition_date'] for r in results]
    ndvi_values = [r['ndvi_mean'] for r in results]
    ndwi_values = [r['ndwi_mean'] for r in results]
    
    # Initialize time-series analyzer
    ts = NDVITimeSeries(dates, ndvi_values)
    
    # Run all analyses
    return {
        'field_id': field_id,
        'field_name': field.name,
        'farm_name': field.farm.name if field.farm else None,
        'analysis_period': {
            'start': dates[0].isoformat() if dates else None,
            'end': dates[-1].isoformat() if dates else None,
            'data_points': len(dates)
        },
        'crop_cycles': ts.detect_crop_cycles(),
        'harvest_germination': ts.detect_harvest_germination(),
        'land_degradation': ts.detect_land_degradation(),
        'raw_data': {
            'dates': [d.isoformat() for d in dates],
            'ndvi_values': ndvi_values,
            'ndwi_values': ndwi_values
        }
    }
