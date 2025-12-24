"""
ThingsBoard IoT Data Client
Pulls sensor data from http://icarus.lums.edu.pk using GET endpoints

Authentication:
- POST /api/auth/login - Get JWT token using username/password

Primary API Endpoints (JWT authentication required):
- GET /api/plugins/telemetry/{entityType}/{entityId}/values/timeseries
      ?agg&endTs&interval&keys&limit&orderBy&startTs&useStrictDataTypes
      Get time-series data (getTimeseries)
      All timestamps are in UTC milliseconds
      
- GET /api/plugins/telemetry/{entityType}/{entityId}/keys/timeseries
      Get time-series keys (getTimeseriesKeys)
      
entityType = DEVICE for all sensor/device telemetry operations

Example keys from LUMS IoT sensors:
- soilMoisture_%
- soilMoisture_adc
- temperature
"""

import requests
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any
from django.utils import timezone
from django.conf import settings


class ThingsBoardClient:
    """
    REST API client for ThingsBoard IoT platform
    Uses plugin telemetry endpoints with JWT authentication
    
    Primary endpoints:
    - POST /api/auth/login - Authenticate and get JWT token
    - GET /api/plugins/telemetry/DEVICE/{entityId}/values/timeseries - Get time-series data
    - GET /api/plugins/telemetry/DEVICE/{entityId}/keys/timeseries - Get available keys
    
    All timestamps use UTC (milliseconds since epoch)
    """
    
    BASE_URL = "http://icarus.lums.edu.pk"
    ENTITY_TYPE = "DEVICE"  # Default entity type for all operations
    
    # Authentication endpoint
    LOGIN_URL = "/api/auth/login"
    
    # Plugin telemetry endpoints (JWT required)
    TIMESERIES_URL = "/api/plugins/telemetry/{entityType}/{entityId}/values/timeseries"
    TIMESERIES_KEYS_URL = "/api/plugins/telemetry/{entityType}/{entityId}/keys/timeseries"
    
    def __init__(self, base_url: str = None, jwt_token: str = None,
                 username: str = None, password: str = None):
        """
        Initialize ThingsBoard client
        
        Args:
            base_url: ThingsBoard server URL (default: icarus.lums.edu.pk)
            jwt_token: JWT token for authenticated API endpoints
            username: ThingsBoard username for auto-login
            password: ThingsBoard password for auto-login
        """
        self.base_url = base_url or self.BASE_URL
        self.jwt_token = jwt_token
        self.refresh_token = None
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        })
        
        # Auto-login if credentials provided
        if username and password and not jwt_token:
            self.login(username, password)
        elif jwt_token:
            self.session.headers.update({
                'X-Authorization': f'Bearer {jwt_token}'
            })
    
    def login(self, username: str, password: str) -> Dict:
        """
        Login to ThingsBoard and get JWT token
        Uses: POST /api/auth/login
        
        Args:
            username: ThingsBoard username (e.g., tenant@thingsboard.org)
            password: ThingsBoard password
        
        Returns:
            Dict with token and refreshToken on success
        """
        url = f"{self.base_url}{self.LOGIN_URL}"
        
        try:
            response = self.session.post(
                url,
                json={"username": username, "password": password},
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                self.jwt_token = data.get('token')
                self.refresh_token = data.get('refreshToken')
                
                # Set authorization header for future requests
                self.session.headers.update({
                    'X-Authorization': f'Bearer {self.jwt_token}'
                })
                
                return {
                    'success': True,
                    'token': self.jwt_token,
                    'refreshToken': self.refresh_token,
                    'timestamp': datetime.utcnow().isoformat()
                }
            elif response.status_code == 401:
                error_data = response.json() if response.text else {}
                return {
                    'success': False,
                    'error': error_data.get('message', 'Authentication failed'),
                    'errorCode': error_data.get('errorCode')
                }
            else:
                return {'success': False, 'error': f'HTTP {response.status_code}: {response.text}'}
                
        except requests.exceptions.Timeout:
            return {'success': False, 'error': 'Login request timeout'}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def set_jwt_token(self, token: str):
        """Set JWT token for authenticated requests"""
        self.jwt_token = token
        self.session.headers.update({
            'X-Authorization': f'Bearer {token}'
        })
    
    def get_timeseries_keys(self, entity_id: str, entity_type: str = None) -> Dict:
        """
        Get available time-series keys for a device.
        Uses: GET /api/plugins/telemetry/DEVICE/{entityId}/keys/timeseries
        
        Args:
            entity_id: Device UUID
            entity_type: Entity type (default: 'DEVICE')
        
        Returns:
            Dict with list of available timeseries keys
        """
        entity_type = entity_type or self.ENTITY_TYPE
        url = f"{self.base_url}/api/plugins/telemetry/{entity_type}/{entity_id}/keys/timeseries"
        
        try:
            response = self.session.get(url, timeout=10)
            
            if response.status_code == 200:
                return {
                    'success': True,
                    'keys': response.json(),
                    'entity_type': entity_type,
                    'entity_id': entity_id,
                    'timestamp': datetime.now().isoformat()
                }
            elif response.status_code == 401:
                return {'success': False, 'error': 'JWT authentication required'}
            elif response.status_code == 404:
                return {'success': False, 'error': f'Device {entity_id} not found'}
            else:
                return {'success': False, 'error': f'HTTP {response.status_code}: {response.text}'}
                
        except requests.exceptions.Timeout:
            return {'success': False, 'error': 'Request timeout'}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def get_timeseries(self, entity_id: str,
                       keys: List[str] = None,
                       start_ts: int = None,
                       end_ts: int = None,
                       interval: int = None,
                       limit: int = 100,
                       agg: str = None,
                       order_by: str = 'DESC',
                       use_strict_data_types: bool = True,
                       entity_type: str = None) -> Dict:
        """
        Get time-series data for a device.
        Uses: GET /api/plugins/telemetry/DEVICE/{entityId}/values/timeseries
              ?agg&endTs&interval&keys&limit&orderBy&startTs&useStrictDataTypes
        
        Args:
            entity_id: Device UUID
            keys: List of telemetry keys to retrieve (if None, fetches keys first)
            start_ts: Start timestamp in milliseconds
            end_ts: End timestamp in milliseconds
            interval: Aggregation interval in milliseconds
            limit: Maximum number of data points per key
            agg: Aggregation function ('MIN', 'MAX', 'AVG', 'SUM', 'COUNT', 'NONE')
            order_by: Order by timestamp ('ASC' or 'DESC')
            use_strict_data_types: Whether to use strict data types
            entity_type: Entity type (default: 'DEVICE')
        
        Returns:
            Dict with timeseries data
        """
        entity_type = entity_type or self.ENTITY_TYPE
        
        # If no keys provided, fetch available keys first
        if not keys:
            keys_result = self.get_timeseries_keys(entity_id, entity_type)
            if keys_result.get('success') and keys_result.get('keys'):
                keys = keys_result['keys']
            elif not keys_result.get('success'):
                return keys_result  # Return the error
        
        url = f"{self.base_url}/api/plugins/telemetry/{entity_type}/{entity_id}/values/timeseries"
        
        params = {
            'limit': limit,
            'orderBy': order_by,
            'useStrictDataTypes': str(use_strict_data_types).lower()
        }
        
        if keys:
            params['keys'] = ','.join(keys) if isinstance(keys, list) else keys
        if start_ts is not None:
            params['startTs'] = start_ts
        if end_ts is not None:
            params['endTs'] = end_ts
        if interval is not None:
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
                    'entity_id': entity_id,
                    'keys_requested': keys,
                    'params': params
                }
            elif response.status_code == 401:
                return {'success': False, 'error': 'JWT authentication required'}
            elif response.status_code == 404:
                return {'success': False, 'error': f'Device {entity_id} not found'}
            else:
                return {'success': False, 'error': f'HTTP {response.status_code}: {response.text}'}
                
        except requests.exceptions.Timeout:
            return {'success': False, 'error': 'Request timeout'}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def get_latest_timeseries(self, entity_id: str,
                               keys: List[str] = None,
                               use_strict_data_types: bool = True,
                               entity_type: str = None) -> Dict:
        """
        Get latest time-series values for a device.
        Uses: GET /api/plugins/telemetry/DEVICE/{entityId}/values/timeseries
              with limit=1 and orderBy=DESC to get most recent values
        
        Args:
            entity_id: Device UUID
            keys: List of telemetry keys to retrieve (if None, fetches all keys)
            use_strict_data_types: Whether to use strict data types
            entity_type: Entity type (default: 'DEVICE')
        
        Returns:
            Dict with latest timeseries values
        """
        entity_type = entity_type or self.ENTITY_TYPE
        
        # If no keys provided, fetch available keys first
        if not keys:
            keys_result = self.get_timeseries_keys(entity_id, entity_type)
            if keys_result.get('success') and keys_result.get('keys'):
                keys = keys_result['keys']
            elif not keys_result.get('success'):
                return keys_result  # Return the error
            else:
                return {'success': False, 'error': 'No telemetry keys available for this device'}
        
        # Use get_timeseries with limit=1 to get latest values
        result = self.get_timeseries(
            entity_id=entity_id,
            keys=keys,
            limit=1,
            order_by='DESC',
            use_strict_data_types=use_strict_data_types,
            entity_type=entity_type
        )
        
        if not result.get('success'):
            return result
        
        # Parse to more usable format
        data = result.get('data', {})
        latest_values = {}
        for key, values in data.items():
            if values and len(values) > 0:
                ts = values[0].get('ts')
                latest_values[key] = {
                    'value': values[0].get('value'),
                    'ts': ts,
                    'timestamp_utc': datetime.utcfromtimestamp(ts / 1000).isoformat() + 'Z' if ts else None
                }
        
        return {
            'success': True,
            'data': data,
            'latest_values': latest_values,
            'entity_type': entity_type,
            'entity_id': entity_id,
            'keys': keys,
            'timestamp': datetime.utcnow().isoformat() + 'Z'
        }
    
    def get_timeseries_range(self, entity_id: str,
                              start_time: datetime = None,
                              end_time: datetime = None,
                              keys: List[str] = None,
                              interval: int = 3600000,
                              agg: str = 'AVG',
                              limit: int = 1000,
                              entity_type: str = None) -> Dict:
        """
        Get time-series data for a specific time range.
        Uses: GET /api/plugins/telemetry/DEVICE/{entityId}/values/timeseries
              with startTs and endTs parameters
        
        Args:
            entity_id: Device UUID
            start_time: Start of time range (default: 24 hours ago) - will be converted to UTC
            end_time: End of time range (default: now) - will be converted to UTC
            keys: Telemetry keys to retrieve (if None, fetches all)
            interval: Aggregation interval in milliseconds (default: 1 hour)
            agg: Aggregation function ('MIN', 'MAX', 'AVG', 'SUM', 'COUNT', 'NONE')
            limit: Maximum data points per key
            entity_type: Entity type (default: 'DEVICE')
        
        Returns:
            Time series telemetry data
        """
        entity_type = entity_type or self.ENTITY_TYPE
        
        # Default time range: last 24 hours (in UTC)
        if end_time is None:
            end_time = datetime.utcnow()
        if start_time is None:
            start_time = end_time - timedelta(hours=24)
        
        # Convert to UTC milliseconds timestamp
        # If datetime is timezone-aware, convert to UTC first
        start_ts = self._to_utc_millis(start_time)
        end_ts = self._to_utc_millis(end_time)
        
        return self.get_timeseries(
            entity_id=entity_id,
            keys=keys,
            start_ts=start_ts,
            end_ts=end_ts,
            interval=interval,
            agg=agg,
            limit=limit,
            entity_type=entity_type
        )
    
    @staticmethod
    def _to_utc_millis(dt: datetime) -> int:
        """
        Convert datetime to UTC milliseconds timestamp
        
        Args:
            dt: datetime object (naive assumed UTC, aware converted to UTC)
        
        Returns:
            Milliseconds since Unix epoch (UTC)
        """
        if dt.tzinfo is not None:
            # Convert timezone-aware datetime to UTC
            import pytz
            utc_dt = dt.astimezone(pytz.UTC)
            return int(utc_dt.timestamp() * 1000)
        else:
            # Naive datetime assumed to be UTC
            return int(dt.timestamp() * 1000)
    
    @staticmethod
    def _from_utc_millis(ts: int) -> datetime:
        """
        Convert UTC milliseconds timestamp to datetime
        
        Args:
            ts: Milliseconds since Unix epoch (UTC)
        
        Returns:
            UTC datetime object
        """
        return datetime.utcfromtimestamp(ts / 1000)


class FarmIoTManager:
    """
    Manages IoT data for farms from ThingsBoard
    Integrates with Django models
    
    Uses plugin telemetry endpoints:
    - POST /api/auth/login - Authenticate with username/password
    - GET /api/plugins/telemetry/DEVICE/{entityId}/keys/timeseries
    - GET /api/plugins/telemetry/DEVICE/{entityId}/values/timeseries
    
    All timestamps are in UTC
    """
    
    def __init__(self, jwt_token: str = None, username: str = None, password: str = None):
        """
        Initialize manager with JWT token or credentials
        
        Args:
            jwt_token: JWT token for ThingsBoard authentication
            username: ThingsBoard username for auto-login
            password: ThingsBoard password for auto-login
        """
        self.client = ThingsBoardClient(
            jwt_token=jwt_token,
            username=username,
            password=password
        )
    
    def login(self, username: str, password: str) -> Dict:
        """Login to ThingsBoard and get JWT token"""
        return self.client.login(username, password)
    
    def set_jwt_token(self, token: str):
        """Set JWT token for authentication"""
        self.client.set_jwt_token(token)
    
    def sync_device_data(self, device) -> Dict:
        """
        Sync telemetry data from ThingsBoard for a device
        Uses: GET /api/plugins/telemetry/DEVICE/{entityId}/values/timeseries
        
        Args:
            device: Device model instance (must have thingsboard_id field)
        
        Returns:
            Dict with sync result
        """
        from iot.models import SensorReading, WeatherData
        
        # Device must have ThingsBoard entity ID (UUID)
        device_uuid = getattr(device, 'thingsboard_id', None) or getattr(device, 'external_id', None)
        
        if not device_uuid:
            return {'success': False, 'error': 'Device has no ThingsBoard entity ID configured'}
        
        if not self.client.jwt_token:
            return {'success': False, 'error': 'JWT token required for ThingsBoard API access'}
        
        result = {
            'success': False,
            'device_id': str(device.id),
            'device_name': device.name,
            'thingsboard_id': str(device_uuid),
            'readings_created': 0,
            'error': None
        }
        
        # First, get available telemetry keys for this device
        keys_result = self.client.get_timeseries_keys(entity_id=str(device_uuid))
        
        if not keys_result.get('success'):
            result['error'] = keys_result.get('error', 'Failed to fetch telemetry keys')
            return result
        
        keys = keys_result.get('keys', [])
        
        if not keys:
            result['success'] = True
            result['data'] = {}
            result['note'] = 'No telemetry keys available - device may need to send data first'
            return result
        
        # Get latest telemetry values using the values/timeseries endpoint
        telemetry = self.client.get_latest_timeseries(
            entity_id=str(device_uuid),
            keys=keys
        )
        
        if not telemetry.get('success'):
            result['error'] = telemetry.get('error', 'Failed to fetch telemetry')
            return result
        
        # Extract latest values
        latest_values = telemetry.get('latest_values', {})
        
        # Check if data is empty
        if not latest_values:
            result['success'] = True
            result['data'] = {}
            result['note'] = 'No telemetry data available'
            return result
        
        # Convert to simple key-value format - store ALL raw telemetry keys directly
        # This preserves original key names like 'soilMoisture_%', 'soilMoisture_adc', 'temperature'
        raw_data = {key: val.get('value') for key, val in latest_values.items()}
        
        # Get timestamps for each key
        timestamps = {key: val.get('ts') for key, val in latest_values.items()}
        
        # Parse telemetry based on device type
        if device.device_type == 'weather_station':
            weather_data = self._parse_weather_data(raw_data)
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
                    result['data'] = weather_data
                    result['raw_keys'] = list(raw_data.keys())
                except Exception as e:
                    result['error'] = str(e)
        else:
            # Store ALL raw telemetry directly in JSONField - no parsing/transformation
            # This preserves keys like 'soilMoisture_%', 'soilMoisture_adc', 'temperature'
            if raw_data:
                try:
                    SensorReading.objects.create(
                        device=device,
                        results=raw_data,  # Store raw telemetry as-is
                        timestamp=timezone.now()
                    )
                    result['readings_created'] = 1
                    result['success'] = True
                    result['data'] = raw_data
                    result['keys_stored'] = list(raw_data.keys())
                except Exception as e:
                    result['error'] = str(e)
        
        return result
    
    def sync_device_historical_data(self, device, start_time: datetime = None,
                                     end_time: datetime = None,
                                     agg: str = 'NONE',
                                     interval: int = 3600000,
                                     limit: int = 10000) -> Dict:
        """
        Sync historical telemetry data from ThingsBoard for a device.
        Creates SensorReading records for each data point.
        
        Args:
            device: Device model instance
            start_time: Start of time range (default: 30 days ago)
            end_time: End of time range (default: now)
            agg: Aggregation function ('NONE' for raw data, 'AVG', 'MIN', 'MAX')
            interval: Aggregation interval in milliseconds (only used if agg != 'NONE')
            limit: Maximum number of data points to retrieve
        
        Returns:
            Dict with sync result including number of readings created
        """
        from iot.models import SensorReading
        
        device_uuid = getattr(device, 'thingsboard_id', None) or getattr(device, 'external_id', None)
        
        if not device_uuid:
            return {'success': False, 'error': 'Device has no ThingsBoard entity ID configured'}
        
        if not self.client.jwt_token:
            return {'success': False, 'error': 'JWT token required for ThingsBoard API access'}
        
        result = {
            'success': False,
            'device_id': str(device.id),
            'device_name': device.name,
            'thingsboard_id': str(device_uuid),
            'readings_created': 0,
            'readings_skipped': 0,
            'error': None
        }
        
        # Default time range
        if end_time is None:
            end_time = datetime.utcnow()
        if start_time is None:
            start_time = datetime(2024, 8, 1)  # August 2024 - when device was set up
        
        result['start_time'] = start_time.isoformat()
        result['end_time'] = end_time.isoformat()
        
        # Get available keys first
        keys_result = self.client.get_timeseries_keys(entity_id=str(device_uuid))
        if not keys_result.get('success'):
            result['error'] = keys_result.get('error', 'Failed to fetch telemetry keys')
            return result
        
        keys = keys_result.get('keys', [])
        if not keys:
            result['success'] = True
            result['note'] = 'No telemetry keys available'
            return result
        
        result['keys'] = keys
        
        # Convert to UTC milliseconds
        start_ts = int(start_time.timestamp() * 1000)
        end_ts = int(end_time.timestamp() * 1000)
        
        # Fetch historical data
        timeseries = self.client.get_timeseries(
            entity_id=str(device_uuid),
            keys=keys,
            start_ts=start_ts,
            end_ts=end_ts,
            agg=agg,
            interval=interval if agg != 'NONE' else None,
            limit=limit,
            order_by='ASC'  # Oldest first
        )
        
        if not timeseries.get('success'):
            result['error'] = timeseries.get('error', 'Failed to fetch historical data')
            return result
        
        data = timeseries.get('data', {})
        
        if not data:
            result['success'] = True
            result['note'] = 'No historical data available in the specified time range'
            return result
        
        # Process data - ThingsBoard returns {key: [{ts: ..., value: ...}, ...]}
        # We need to pivot this to create readings per timestamp
        
        # Collect all unique timestamps
        all_timestamps = set()
        for key, values in data.items():
            for v in values:
                if v.get('ts'):
                    all_timestamps.add(v['ts'])
        
        all_timestamps = sorted(all_timestamps)
        result['total_timestamps'] = len(all_timestamps)
        
        # Create a reading for each timestamp
        readings_to_create = []
        for ts in all_timestamps:
            # Check if reading already exists for this timestamp
            reading_time = datetime.utcfromtimestamp(ts / 1000)
            
            # Build results dict for this timestamp
            reading_data = {}
            for key, values in data.items():
                for v in values:
                    if v.get('ts') == ts:
                        reading_data[key] = v.get('value')
                        break
            
            if reading_data:
                # Check for duplicate
                exists = SensorReading.objects.filter(
                    device=device,
                    timestamp=timezone.make_aware(reading_time)
                ).exists()
                
                if not exists:
                    readings_to_create.append(SensorReading(
                        device=device,
                        results=reading_data,
                        timestamp=timezone.make_aware(reading_time)
                    ))
                else:
                    result['readings_skipped'] += 1
        
        # Bulk create readings
        if readings_to_create:
            try:
                SensorReading.objects.bulk_create(readings_to_create, ignore_conflicts=True)
                result['readings_created'] = len(readings_to_create)
                result['success'] = True
            except Exception as e:
                result['error'] = str(e)
        else:
            result['success'] = True
            result['note'] = 'All readings already exist in database'
        
        return result
    
    def get_device_timeseries_range(self, device, start_time: datetime = None,
                                     end_time: datetime = None,
                                     keys: List[str] = None,
                                     agg: str = 'AVG',
                                     interval: int = 3600000) -> Dict:
        """
        Get historical timeseries data for a device
        Uses: GET /api/plugins/telemetry/DEVICE/{entityId}/values/timeseries
              with startTs, endTs, agg, interval parameters
        
        Args:
            device: Device model instance
            start_time: Start of time range (default: 24 hours ago)
            end_time: End of time range (default: now)
            keys: Telemetry keys to retrieve (if None, fetches all)
            agg: Aggregation function ('MIN', 'MAX', 'AVG', 'SUM', 'COUNT', 'NONE')
            interval: Aggregation interval in milliseconds
        
        Returns:
            Time series telemetry data
        """
        device_uuid = getattr(device, 'thingsboard_id', None) or getattr(device, 'external_id', None)
        
        if not device_uuid:
            return {'success': False, 'error': 'Device has no ThingsBoard entity ID configured'}
        
        if not self.client.jwt_token:
            return {'success': False, 'error': 'JWT token required for ThingsBoard API access'}
        
        return self.client.get_timeseries_range(
            entity_id=str(device_uuid),
            start_time=start_time,
            end_time=end_time,
            keys=keys,
            agg=agg,
            interval=interval
        )
    
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
        
        # Helper to get moisture from various key names
        def get_moisture(results):
            return (results.get('soilMoisture_%') or 
                    results.get('soilMoisture_adc') or
                    results.get('soilMoisture') or
                    results.get('soil_moisture') or 
                    results.get('moisture'))
        
        # Get latest readings for each device
        latest_readings = []
        all_keys = set()
        for device in devices:
            reading = SensorReading.objects.filter(device=device).order_by('-timestamp').first()
            if reading:
                # Store raw results from JSONField
                results = reading.results or {}
                all_keys.update(results.keys())
                
                latest_readings.append({
                    'device_id': device.id,
                    'device_name': device.name,
                    'device_type': device.device_type,
                    'thingsboard_id': str(device.thingsboard_id) if device.thingsboard_id else None,
                    'temperature': results.get('temperature') or results.get('temp'),
                    'moisture': get_moisture(results),
                    'humidity': results.get('humidity'),
                    'results': results,  # Include full raw results
                    'keys': list(results.keys()),  # Available keys for this device
                    'timestamp': reading.timestamp.isoformat() if reading.timestamp else None
                })
        
        # Get latest weather
        weather = WeatherData.objects.filter(farm=farm).order_by('-timestamp').first()
        
        # Count all readings
        readings_count = SensorReading.objects.filter(device__farm=farm).count()
        
        # Calculate averages for today using the model's helper properties
        today_start = timezone.now().replace(hour=0, minute=0, second=0, microsecond=0)
        today_readings = SensorReading.objects.filter(
            device__farm=farm,
            timestamp__gte=today_start
        )
        
        # Aggregate using the helper properties which handle key variations
        avg_temp = None
        avg_moisture = None
        if today_readings.exists():
            temps = [r.temperature for r in today_readings if r.temperature is not None]
            moistures = [r.moisture for r in today_readings if r.moisture is not None]
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
            'available_keys': list(all_keys),  # All unique keys across all devices
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

def login_thingsboard(username: str, password: str, base_url: str = None) -> Dict:
    """
    Login to ThingsBoard and get JWT token
    Uses: POST /api/auth/login
    
    Args:
        username: ThingsBoard username
        password: ThingsBoard password
        base_url: Optional ThingsBoard server URL
    
    Returns:
        Dict with 'token' and 'refreshToken' on success
    
    Example:
        result = login_thingsboard('tenant@thingsboard.org', 'tenant')
        if result['success']:
            jwt_token = result['token']
    """
    client = ThingsBoardClient(base_url=base_url)
    return client.login(username, password)


def sync_device(device_id: int, jwt_token: str = None,
                username: str = None, password: str = None) -> Dict:
    """
    Sync a single device by ID
    
    Args:
        device_id: Device ID in local database
        jwt_token: JWT token for ThingsBoard authentication
        username: ThingsBoard username (alternative to jwt_token)
        password: ThingsBoard password (alternative to jwt_token)
    
    Returns:
        Dict with sync result
    """
    from core.models import Device
    
    try:
        device = Device.objects.get(id=device_id)
    except Device.DoesNotExist:
        return {'success': False, 'error': f'Device {device_id} not found'}
    
    manager = FarmIoTManager(jwt_token=jwt_token, username=username, password=password)
    return manager.sync_device_data(device)


def sync_farm_devices(farm_id: int, jwt_token: str = None,
                      username: str = None, password: str = None) -> Dict:
    """
    Sync all devices for a farm
    
    Args:
        farm_id: Farm ID in local database
        jwt_token: JWT token for ThingsBoard authentication
        username: ThingsBoard username (alternative to jwt_token)
        password: ThingsBoard password (alternative to jwt_token)
    
    Returns:
        Dict with sync summary
    """
    from core.models import Farm
    
    try:
        farm = Farm.objects.get(id=farm_id)
    except Farm.DoesNotExist:
        return {'success': False, 'error': f'Farm {farm_id} not found'}
    
    manager = FarmIoTManager(jwt_token=jwt_token, username=username, password=password)
    return manager.sync_all_devices(farm)


def get_device_timeseries(entity_id: str, jwt_token: str = None,
                          username: str = None, password: str = None,
                          keys: List[str] = None,
                          start_time: datetime = None,
                          end_time: datetime = None,
                          agg: str = None,
                          interval: int = None,
                          limit: int = 100) -> Dict:
    """
    Get timeseries data for a device using ThingsBoard API
    Uses: GET /api/plugins/telemetry/DEVICE/{entityId}/values/timeseries
    
    All timestamps should be in UTC. Response timestamps are also in UTC.
    
    Args:
        entity_id: ThingsBoard device UUID
        jwt_token: JWT token for authentication
        username: ThingsBoard username (alternative to jwt_token)
        password: ThingsBoard password (alternative to jwt_token)
        keys: Telemetry keys to retrieve (if None, fetches all available keys)
              Example keys: ['soilMoisture_%', 'soilMoisture_adc', 'temperature']
        start_time: Start of time range (UTC datetime)
        end_time: End of time range (UTC datetime)
        agg: Aggregation function ('MIN', 'MAX', 'AVG', 'SUM', 'COUNT', 'NONE')
        interval: Aggregation interval in milliseconds
        limit: Maximum data points per key
    
    Returns:
        Dict with timeseries data
    """
    client = ThingsBoardClient(jwt_token=jwt_token, username=username, password=password)
    
    if start_time or end_time:
        return client.get_timeseries_range(
            entity_id=entity_id,
            start_time=start_time,
            end_time=end_time,
            keys=keys,
            agg=agg,
            interval=interval,
            limit=limit
        )
    else:
        return client.get_latest_timeseries(entity_id=entity_id, keys=keys)


def get_device_keys(entity_id: str, jwt_token: str = None,
                    username: str = None, password: str = None) -> Dict:
    """
    Get available timeseries keys for a device
    Uses: GET /api/plugins/telemetry/DEVICE/{entityId}/keys/timeseries
    
    Args:
        entity_id: ThingsBoard device UUID
        jwt_token: JWT token for authentication
        username: ThingsBoard username (alternative to jwt_token)
        password: ThingsBoard password (alternative to jwt_token)
    
    Returns:
        Dict with list of available keys
        Example: {'success': True, 'keys': ['soilMoisture_%', 'soilMoisture_adc', 'temperature']}
    """
    client = ThingsBoardClient(jwt_token=jwt_token, username=username, password=password)
    return client.get_timeseries_keys(entity_id=entity_id)
