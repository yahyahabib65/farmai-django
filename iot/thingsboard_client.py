"""
ThingsBoard IoT Data Client
Pulls sensor data from http://icarus.lums.edu.pk using GET endpoints
Supports both device token and JWT authentication

API Endpoints used:
- GET /api/plugins/telemetry/{entityType}/{entityId}/values/timeseries - Get time-series data
- GET /api/plugins/telemetry/{entityType}/{entityId}/values/timeseries?keys - Get latest time-series
- GET /api/plugins/telemetry/{entityType}/{entityId}/keys/timeseries - Get time-series keys
- GET /api/v1/{token}/attributes - Get device attributes (device token auth)
"""

import requests
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any
from django.utils import timezone


class ThingsBoardClient:
    """
    REST API client for ThingsBoard IoT platform
    Supports both device token and JWT authentication for telemetry endpoints
    """
    
    BASE_URL = "http://icarus.lums.edu.pk"
    
    # Device access token endpoints (no JWT required)
    TELEMETRY_URL = "/api/v1/{token}/telemetry"
    ATTRIBUTES_URL = "/api/v1/{token}/attributes"
    
    # Plugin telemetry endpoints (JWT required for historical data)
    TIMESERIES_URL = "/api/plugins/telemetry/{entityType}/{entityId}/values/timeseries"
    TIMESERIES_KEYS_URL = "/api/plugins/telemetry/{entityType}/{entityId}/keys/timeseries"
    ATTRIBUTES_SCOPE_URL = "/api/plugins/telemetry/{entityType}/{entityId}/values/attributes/{scope}"
    
    def __init__(self, base_url: str = None, jwt_token: str = None):
        """
        Initialize ThingsBoard client
        
        Args:
            base_url: ThingsBoard server URL (default: icarus.lums.edu.pk)
            jwt_token: Optional JWT token for authenticated API endpoints
        """
        self.base_url = base_url or self.BASE_URL
        self.jwt_token = jwt_token
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        })
        if jwt_token:
            self.session.headers.update({
                'X-Authorization': f'Bearer {jwt_token}'
            })
    
    def set_jwt_token(self, token: str):
        """Set JWT token for authenticated requests"""
        self.jwt_token = token
        self.session.headers.update({
            'X-Authorization': f'Bearer {token}'
        })
    
    def get_latest_telemetry(self, device_token: str, keys: List[str] = None) -> Dict:
        """
        Get latest telemetry values for a device using device token.
        
        Note: ThingsBoard HTTP API device endpoints:
        - POST /api/v1/{token}/telemetry - Send telemetry
        - GET /api/v1/{token}/attributes - Get shared attributes
        
        For reading telemetry history, we use the attributes endpoint
        which can store the last values.
        
        Args:
            device_token: Device access token
            keys: Optional list of telemetry keys to retrieve
        
        Returns:
            Dict with latest telemetry values
        """
        # Use attributes endpoint for GET (telemetry endpoint is POST-only)
        url = f"{self.base_url}/api/v1/{device_token}/attributes"
        params = {}
        if keys:
            params['sharedKeys'] = ','.join(keys)
            params['clientKeys'] = ','.join(keys)
        
        try:
            response = self.session.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                return {
                    'success': True,
                    'data': data,
                    'timestamp': datetime.now().isoformat()
                }
            elif response.status_code == 401:
                return {'success': False, 'error': 'Invalid device token'}
            else:
                return {'success': False, 'error': f'HTTP {response.status_code}'}
                
        except requests.exceptions.Timeout:
            return {'success': False, 'error': 'Request timeout'}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def get_attributes(self, device_token: str, keys: List[str] = None) -> Dict:
        """
        Get device attributes
        
        Args:
            device_token: Device access token
            keys: Optional list of attribute keys
        
        Returns:
            Dict with device attributes
        """
        url = f"{self.base_url}/api/v1/{device_token}/attributes"
        params = {}
        if keys:
            params['sharedKeys'] = ','.join(keys)
        
        try:
            response = self.session.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                return {
                    'success': True,
                    'data': response.json(),
                    'timestamp': datetime.now().isoformat()
                }
            else:
                return {'success': False, 'error': f'HTTP {response.status_code}'}
                
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def get_timeseries_keys(self, entity_type: str, entity_id: str) -> Dict:
        """
        Get available time-series keys for an entity.
        Uses: GET /api/plugins/telemetry/{entityType}/{entityId}/keys/timeseries
        
        Args:
            entity_type: Entity type (e.g., 'DEVICE')
            entity_id: Entity UUID
        
        Returns:
            Dict with list of available timeseries keys
        """
        url = f"{self.base_url}/api/plugins/telemetry/{entity_type}/{entity_id}/keys/timeseries"
        
        try:
            response = self.session.get(url, timeout=10)
            
            if response.status_code == 200:
                return {
                    'success': True,
                    'keys': response.json(),
                    'timestamp': datetime.now().isoformat()
                }
            elif response.status_code == 401:
                return {'success': False, 'error': 'JWT authentication required'}
            else:
                return {'success': False, 'error': f'HTTP {response.status_code}'}
                
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def get_timeseries(self, entity_type: str, entity_id: str,
                       keys: List[str] = None,
                       start_ts: int = None,
                       end_ts: int = None,
                       interval: int = None,
                       limit: int = 100,
                       agg: str = None,
                       order_by: str = 'DESC',
                       use_strict_data_types: bool = True) -> Dict:
        """
        Get time-series data for an entity.
        Uses: GET /api/plugins/telemetry/{entityType}/{entityId}/values/timeseries
        
        Args:
            entity_type: Entity type (e.g., 'DEVICE')
            entity_id: Entity UUID
            keys: List of telemetry keys to retrieve
            start_ts: Start timestamp in milliseconds
            end_ts: End timestamp in milliseconds
            interval: Aggregation interval in milliseconds
            limit: Maximum number of data points per key
            agg: Aggregation function ('MIN', 'MAX', 'AVG', 'SUM', 'COUNT', 'NONE')
            order_by: Order by timestamp ('ASC' or 'DESC')
            use_strict_data_types: Whether to use strict data types
        
        Returns:
            Dict with timeseries data
        """
        url = f"{self.base_url}/api/plugins/telemetry/{entity_type}/{entity_id}/values/timeseries"
        
        params = {
            'limit': limit,
            'orderBy': order_by,
            'useStrictDataTypes': str(use_strict_data_types).lower()
        }
        
        if keys:
            params['keys'] = ','.join(keys)
        if start_ts:
            params['startTs'] = start_ts
        if end_ts:
            params['endTs'] = end_ts
        if interval:
            params['interval'] = interval
        if agg:
            params['agg'] = agg
        
        try:
            response = self.session.get(url, params=params, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                return {
                    'success': True,
                    'data': data,
                    'timestamp': datetime.now().isoformat(),
                    'entity_type': entity_type,
                    'entity_id': entity_id
                }
            elif response.status_code == 401:
                return {'success': False, 'error': 'JWT authentication required'}
            else:
                return {'success': False, 'error': f'HTTP {response.status_code}'}
                
        except requests.exceptions.Timeout:
            return {'success': False, 'error': 'Request timeout'}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def get_latest_timeseries(self, entity_type: str, entity_id: str,
                               keys: List[str] = None,
                               use_strict_data_types: bool = True) -> Dict:
        """
        Get latest time-series values for an entity.
        Uses: GET /api/plugins/telemetry/{entityType}/{entityId}/values/timeseries?keys
        
        Args:
            entity_type: Entity type (e.g., 'DEVICE')
            entity_id: Entity UUID
            keys: List of telemetry keys to retrieve
            use_strict_data_types: Whether to use strict data types
        
        Returns:
            Dict with latest timeseries values
        """
        url = f"{self.base_url}/api/plugins/telemetry/{entity_type}/{entity_id}/values/timeseries"
        
        params = {
            'useStrictDataTypes': str(use_strict_data_types).lower()
        }
        
        if keys:
            params['keys'] = ','.join(keys)
        
        try:
            response = self.session.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                # Parse to more usable format
                latest_values = {}
                for key, values in data.items():
                    if values and len(values) > 0:
                        latest_values[key] = {
                            'value': values[0].get('value'),
                            'ts': values[0].get('ts'),
                            'timestamp': datetime.fromtimestamp(values[0].get('ts', 0) / 1000).isoformat() if values[0].get('ts') else None
                        }
                
                return {
                    'success': True,
                    'data': data,
                    'latest_values': latest_values,
                    'timestamp': datetime.now().isoformat()
                }
            elif response.status_code == 401:
                return {'success': False, 'error': 'JWT authentication required'}
            else:
                return {'success': False, 'error': f'HTTP {response.status_code}'}
                
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def get_telemetry_time_series(self, device_token: str, device_id: str,
                                   jwt_token: str = None,
                                   keys: List[str] = None,
                                   start_time: datetime = None,
                                   end_time: datetime = None,
                                   interval: int = 3600000,
                                   limit: int = 100) -> Dict:
        """
        Get telemetry time series data using public API
        
        Note: For historical data, ThingsBoard typically requires JWT auth
        This method provides a fallback using device token where possible
        
        Args:
            device_token: Device access token
            device_id: Device UUID 
            jwt_token: Optional JWT for authenticated endpoints
            keys: Telemetry keys to retrieve
            start_time: Start of time range
            end_time: End of time range
            interval: Aggregation interval in milliseconds
            limit: Maximum data points
        
        Returns:
            Time series telemetry data
        """
        # Default time range: last 24 hours
        if end_time is None:
            end_time = datetime.now()
        if start_time is None:
            start_time = end_time - timedelta(hours=24)
        
        # Try the public telemetry endpoint first
        latest = self.get_latest_telemetry(device_token)
        
        if latest.get('success'):
            # Return latest as single data point if no JWT available
            return {
                'success': True,
                'data': latest.get('data', {}),
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'note': 'Latest telemetry only - historical data requires JWT authentication'
            }
        
        return latest  # Return error if failed


class FarmIoTManager:
    """
    Manages IoT data for farms from ThingsBoard
    Integrates with Django models
    """
    
    def __init__(self):
        self.client = ThingsBoardClient()
    
    def sync_device_data(self, device) -> Dict:
        """
        Sync telemetry data from ThingsBoard for a device
        
        Args:
            device: Device model instance (must have token field)
        
        Returns:
            Dict with sync result
        """
        from iot.models import SensorReading, WeatherData
        
        if not device.token:
            return {'success': False, 'error': 'Device has no ThingsBoard token configured'}
        
        result = {
            'success': False,
            'device_id': str(device.id),
            'device_name': device.name,
            'readings_created': 0,
            'error': None
        }
        
        # Get latest telemetry
        telemetry = self.client.get_latest_telemetry(device.token)
        
        if not telemetry.get('success'):
            result['error'] = telemetry.get('error', 'Failed to fetch telemetry')
            return result
        
        data = telemetry.get('data', {})
        
        # Check if data is empty
        if not data:
            result['success'] = True  # API call succeeded, just no data
            result['error'] = None
            result['data'] = {}
            result['note'] = 'No telemetry data available - device may need to send data first'
            return result
        
        # Parse telemetry based on device type
        if device.device_type == 'weather_station':
            weather_data = self._parse_weather_data(data)
            if weather_data:
                try:
                    WeatherData.objects.create(
                        farm=device.farm,
                        temperature=weather_data.get('temperature'),
                        humidity=weather_data.get('humidity'),
                        rainfall=weather_data.get('rainfall', 0),
                        wind_speed=weather_data.get('wind_speed'),
                        wind_direction=weather_data.get('wind_direction'),
                        atmospheric_pressure=weather_data.get('pressure'),
                        timestamp=timezone.now()
                    )
                    result['readings_created'] = 1
                    result['success'] = True
                except Exception as e:
                    result['error'] = str(e)
        else:
            # Soil sensor or other sensor types - store all telemetry in results JSONField
            sensor_data = self._parse_sensor_data(data)
            if sensor_data:
                try:
                    SensorReading.objects.create(
                        device=device,
                        results=sensor_data,  # Store all telemetry as JSON
                        timestamp=timezone.now()
                    )
                    result['readings_created'] = 1
                    result['success'] = True
                    result['data'] = sensor_data
                except Exception as e:
                    result['error'] = str(e)
        
        return result
    
    def sync_all_devices(self, farm=None) -> Dict:
        """
        Sync data from all devices, optionally filtered by farm
        
        Args:
            farm: Optional farm to filter devices
        
        Returns:
            Summary of sync operation
        """
        from core.models import Device
        
        queryset = Device.objects.filter(status='active')
        if farm:
            queryset = queryset.filter(farm=farm)
        
        results = {
            'total_devices': queryset.count(),
            'successful': 0,
            'failed': 0,
            'readings_created': 0,
            'errors': []
        }
        
        for device in queryset:
            sync_result = self.sync_device_data(device)
            
            if sync_result.get('success'):
                results['successful'] += 1
                results['readings_created'] += sync_result.get('readings_created', 0)
            else:
                results['failed'] += 1
                results['errors'].append({
                    'device': device.name,
                    'error': sync_result.get('error')
                })
        
        return results
    
    def get_farm_sensor_summary(self, farm) -> Dict:
        """
        Get summary of latest sensor readings for a farm
        """
        from iot.models import SensorReading, WeatherData
        from core.models import Device
        from django.db.models import Avg
        
        devices = Device.objects.filter(farm=farm)
        
        # Get latest readings for each device
        latest_readings = []
        for device in devices:
            reading = SensorReading.objects.filter(device=device).order_by('-timestamp').first()
            if reading:
                # Use JSONField results or fallback to legacy fields
                results = reading.results or {}
                latest_readings.append({
                    'device_id': device.id,
                    'device_name': device.name,
                    'device_type': device.device_type,
                    'temperature': results.get('temperature') or reading.temperature,
                    'moisture': results.get('moisture') or results.get('soil_moisture') or reading.moisture,
                    'humidity': results.get('humidity') or reading.humidity,
                    'results': results,  # Include full results for flexibility
                    'timestamp': reading.timestamp.isoformat() if reading.timestamp else None
                })
        
        # Get latest weather
        weather = WeatherData.objects.filter(farm=farm).order_by('-timestamp').first()
        
        # Count all readings
        readings_count = SensorReading.objects.filter(device__farm=farm).count()
        
        # Calculate averages for today
        today_start = timezone.now().replace(hour=0, minute=0, second=0, microsecond=0)
        today_readings = SensorReading.objects.filter(
            device__farm=farm,
            timestamp__gte=today_start
        )
        
        # Since we're using JSONField, we need to aggregate differently
        avg_temp = None
        avg_moisture = None
        if today_readings.exists():
            temps = [r.temperature for r in today_readings if r.temperature]
            moistures = [r.moisture for r in today_readings if r.moisture]
            if temps:
                avg_temp = round(sum(temps) / len(temps), 2)
            if moistures:
                avg_moisture = round(sum(moistures) / len(moistures), 2)
        
        return {
            'farm_id': farm.id,
            'farm_name': farm.name,
            'devices_count': devices.count(),
            'readings_count': readings_count,
            'active_devices': len(latest_readings),
            'latest_readings': latest_readings,
            'weather': {
                'temperature': weather.temperature if weather else None,
                'humidity': weather.humidity if weather else None,
                'precipitation': weather.precipitation if weather else None,
                'recorded_at': weather.timestamp.isoformat() if weather else None
            } if weather else None,
            'daily_averages': {
                'temperature': avg_temp,
                'moisture': avg_moisture,
            },
            'thingsboard': {
                'url': self.client.base_url,
                'configured': True
            },
            'last_sync': latest_readings[0]['timestamp'] if latest_readings else None
        }
    
    def _parse_weather_data(self, telemetry: Dict) -> Optional[Dict]:
        """Parse weather data from ThingsBoard telemetry format"""
        if not telemetry:
            return None
        
        # ThingsBoard telemetry format: {"key": [{"ts": timestamp, "value": value}]}
        # or simpler format: {"key": value}
        
        def get_value(key):
            if key in telemetry:
                val = telemetry[key]
                if isinstance(val, list) and len(val) > 0:
                    return val[0].get('value')
                return val
            return None
        
        return {
            'temperature': get_value('temperature') or get_value('temp'),
            'humidity': get_value('humidity'),
            'rainfall': get_value('rainfall') or get_value('rain'),
            'wind_speed': get_value('windSpeed') or get_value('wind_speed'),
            'wind_direction': get_value('windDirection') or get_value('wind_direction'),
            'pressure': get_value('pressure') or get_value('atmospheric_pressure')
        }
    
    def _parse_sensor_data(self, telemetry: Dict) -> Optional[Dict]:
        """Parse sensor data from ThingsBoard telemetry format"""
        if not telemetry:
            return None
        
        def get_value(key):
            if key in telemetry:
                val = telemetry[key]
                if isinstance(val, list) and len(val) > 0:
                    return val[0].get('value')
                return val
            return None
        
        return {
            'soil_moisture': get_value('soilMoisture') or get_value('soil_moisture') or get_value('moisture'),
            'temperature': get_value('temperature') or get_value('temp'),
            'ph': get_value('pH') or get_value('ph'),
            'nitrogen': get_value('nitrogen') or get_value('N'),
            'phosphorus': get_value('phosphorus') or get_value('P'),
            'potassium': get_value('potassium') or get_value('K')
        }


# Convenience functions for easy use

def sync_device(device_id: int) -> Dict:
    """Sync a single device by ID"""
    from core.models import Device
    
    try:
        device = Device.objects.get(id=device_id)
    except Device.DoesNotExist:
        return {'success': False, 'error': f'Device {device_id} not found'}
    
    manager = FarmIoTManager()
    return manager.sync_device_data(device)


def sync_farm_devices(farm_id: int) -> Dict:
    """Sync all devices for a farm"""
    from core.models import Farm
    
    try:
        farm = Farm.objects.get(id=farm_id)
    except Farm.DoesNotExist:
        return {'success': False, 'error': f'Farm {farm_id} not found'}
    
    manager = FarmIoTManager()
    return manager.sync_all_devices(farm)


def get_device_telemetry(device_token: str) -> Dict:
    """Get latest telemetry for a device token"""
    client = ThingsBoardClient()
    return client.get_latest_telemetry(device_token)
