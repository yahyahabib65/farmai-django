from rest_framework import viewsets, status
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import AllowAny
from .models import SensorReading, WeatherData
from .serializers import SensorReadingSerializer
from .thingsboard_client import (
    ThingsBoardClient, FarmIoTManager, sync_farm_devices,
    login_thingsboard, get_device_keys, get_device_timeseries
)
from core.models import Device, Farm
from django.db.models import Avg
from django.conf import settings
from datetime import datetime, timedelta


class SensorReadingViewSet(viewsets.ModelViewSet):
	queryset = SensorReading.objects.select_related('device').all()
	serializer_class = SensorReadingSerializer
	permission_classes = (AllowAny,)


class ThingsBoardSyncView(APIView):
    """
    Sync IoT data from ThingsBoard using the telemetry plugin API.
    
    Uses endpoints:
    - POST /api/auth/login - Authenticate and get JWT token
    - GET /api/plugins/telemetry/DEVICE/{entityId}/keys/timeseries
    - GET /api/plugins/telemetry/DEVICE/{entityId}/values/timeseries
    """
    permission_classes = (AllowAny,)

    def get(self, request):
        """Get sync status and device info"""
        devices = Device.objects.all()
        devices_with_uuid = devices.filter(thingsboard_id__isnull=False)
        
        return Response({
            'message': 'ThingsBoard IoT Sync (Plugin API)',
            'thingsboard_url': 'http://icarus.lums.edu.pk',
            'total_devices': devices.count(),
            'devices_with_thingsboard_id': devices_with_uuid.count(),
            'devices': [{
                'id': d.id,
                'name': d.name,
                'type': d.device_type,
                'thingsboard_id': str(d.thingsboard_id) if d.thingsboard_id else None,
                'farm': d.farm.name if d.farm else None
            } for d in devices[:20]],
            'instructions': 'POST to sync data from ThingsBoard',
            'required': {
                'username': 'ThingsBoard username for authentication',
                'password': 'ThingsBoard password for authentication'
            },
            'optional': {
                'farm_id': 'Sync all devices for a farm',
                'device_id': 'Sync a specific device',
                'entity_id': 'Test with ThingsBoard device UUID directly'
            }
        })

    def post(self, request):
        """Sync data from ThingsBoard using JWT authentication"""
        farm_id = request.data.get('farm_id')
        device_id = request.data.get('device_id')
        entity_id = request.data.get('entity_id')  # ThingsBoard UUID
        
        # Get credentials from request or settings
        username = request.data.get('username') or getattr(settings, 'THINGSBOARD_USERNAME', None)
        password = request.data.get('password') or getattr(settings, 'THINGSBOARD_PASSWORD', None)
        jwt_token = request.data.get('jwt_token')
        
        # Authenticate if no JWT provided
        if not jwt_token:
            if not username or not password:
                return Response({
                    'error': 'JWT token or credentials (username/password) required',
                    'hint': 'Provide username and password for ThingsBoard authentication, or configure THINGSBOARD_USERNAME and THINGSBOARD_PASSWORD in settings'
                }, status=status.HTTP_400_BAD_REQUEST)
            
            # Login to ThingsBoard
            login_result = login_thingsboard(username, password)
            if not login_result.get('success'):
                return Response({
                    'error': 'ThingsBoard authentication failed',
                    'details': login_result.get('error')
                }, status=status.HTTP_401_UNAUTHORIZED)
            
            jwt_token = login_result.get('token')
        
        # Test a specific ThingsBoard entity ID
        if entity_id:
            keys_result = get_device_keys(entity_id, jwt_token)
            if not keys_result.get('success'):
                return Response({
                    'action': 'test_entity',
                    'success': False,
                    'error': keys_result.get('error')
                })
            
            # Get latest values
            data_result = get_device_timeseries(entity_id, jwt_token, keys=keys_result.get('keys'))
            return Response({
                'action': 'test_entity',
                'entity_id': entity_id,
                'success': data_result.get('success'),
                'keys': keys_result.get('keys'),
                'data': data_result.get('latest_values', {}),
                'error': data_result.get('error')
            })
        
        # Create manager with JWT
        manager = FarmIoTManager(jwt_token=jwt_token)
        
        # Sync specific device
        if device_id:
            try:
                device = Device.objects.get(id=device_id)
                result = manager.sync_device_data(device)
                return Response({
                    'action': 'sync_device',
                    'device': device.name,
                    **result
                })
            except Device.DoesNotExist:
                return Response({'error': f'Device {device_id} not found'}, 
                              status=status.HTTP_404_NOT_FOUND)
        
        # Sync farm devices
        if farm_id:
            try:
                farm = Farm.objects.get(id=farm_id)
                result = manager.sync_all_devices(farm)
                return Response({
                    'action': 'sync_farm',
                    'farm_id': farm_id,
                    'farm_name': farm.name,
                    **result
                })
            except Farm.DoesNotExist:
                return Response({'error': f'Farm {farm_id} not found'}, 
                              status=status.HTTP_404_NOT_FOUND)
        
        # Sync all devices
        result = manager.sync_all_devices()
        return Response({
            'action': 'sync_all',
            **result
        })


class ThingsBoardHistoricalSyncView(APIView):
    """
    Sync historical IoT data from ThingsBoard.
    
    This endpoint pulls data from a specified date range (default: August 2024 to now).
    """
    permission_classes = (AllowAny,)

    def post(self, request):
        """Sync historical data from ThingsBoard"""
        # Get ThingsBoard credentials
        username = request.data.get('username')
        password = request.data.get('password')
        
        if not username or not password:
            return Response({
                'error': 'ThingsBoard username and password required in request body'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        # Login to ThingsBoard
        login_result = login_thingsboard(username, password)
        if not login_result.get('success'):
            return Response({
                'error': 'ThingsBoard login failed',
                'details': login_result.get('error')
            }, status=status.HTTP_401_UNAUTHORIZED)
        
        jwt_token = login_result.get('token')
        
        # Get parameters
        device_id = request.data.get('device_id')
        start_date = request.data.get('start_date')  # Format: YYYY-MM-DD
        end_date = request.data.get('end_date')  # Format: YYYY-MM-DD
        
        if not device_id:
            return Response({
                'error': 'device_id is required'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        # Get device
        try:
            device = Device.objects.get(id=device_id)
        except Device.DoesNotExist:
            return Response({
                'error': f'Device {device_id} not found'
            }, status=status.HTTP_404_NOT_FOUND)
        
        if not device.thingsboard_id:
            return Response({
                'error': f'Device {device.name} has no ThingsBoard ID configured'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        # Parse dates
        start_time = None
        end_time = None
        
        if start_date:
            try:
                start_time = datetime.strptime(start_date, '%Y-%m-%d')
            except ValueError:
                return Response({
                    'error': 'Invalid start_date format. Use YYYY-MM-DD'
                }, status=status.HTTP_400_BAD_REQUEST)
        
        if end_date:
            try:
                end_time = datetime.strptime(end_date, '%Y-%m-%d')
            except ValueError:
                return Response({
                    'error': 'Invalid end_date format. Use YYYY-MM-DD'
                }, status=status.HTTP_400_BAD_REQUEST)
        
        # Create manager and sync
        manager = FarmIoTManager(jwt_token=jwt_token)
        result = manager.sync_device_historical_data(
            device=device,
            start_time=start_time,
            end_time=end_time
        )
        
        return Response({
            'action': 'sync_historical',
            'device': device.name,
            'device_id': str(device.id),
            'thingsboard_id': str(device.thingsboard_id),
            **result
        })


class FarmSensorSummaryView(APIView):
    """Get summary of IoT sensor data for a farm"""
    permission_classes = (AllowAny,)

    def get(self, request, farm_id=None):
        """Get sensor summary for a farm"""
        if farm_id:
            try:
                farm = Farm.objects.get(id=farm_id)
            except Farm.DoesNotExist:
                return Response({'error': 'Farm not found'}, status=status.HTTP_404_NOT_FOUND)
        else:
            farm = Farm.objects.first()
            if not farm:
                return Response({'error': 'No farms exist'}, status=status.HTTP_404_NOT_FOUND)
        
        manager = FarmIoTManager()
        summary = manager.get_farm_sensor_summary(farm)
        
        return Response(summary)


class SensorTimeSeriesView(APIView):
    """Get time series sensor data for charts - grouped by sensor"""
    permission_classes = (AllowAny,)

    def get(self, request, farm_id=None):
        """Get time series data for all sensors in the farm"""
        if farm_id:
            try:
                farm = Farm.objects.get(id=farm_id)
            except Farm.DoesNotExist:
                return Response({'error': 'Farm not found'}, status=status.HTTP_404_NOT_FOUND)
        else:
            farm = Farm.objects.first()
        
        days_param = request.query_params.get('days', '7')
        
        # Get sensor readings
        from django.utils import timezone
        
        # Handle "all" time or specific number of days
        if days_param == 'all':
            readings = SensorReading.objects.filter(
                device__farm=farm
            ).select_related('device').order_by('timestamp')
            period_label = 'All Time'
        else:
            days = int(days_param)
            start_date = timezone.now() - timedelta(days=days)
            readings = SensorReading.objects.filter(
                device__farm=farm,
                timestamp__gte=start_date
            ).select_related('device').order_by('timestamp')
            period_label = f'Last {days} days'
        
        # Get all devices for this farm
        devices = Device.objects.filter(farm=farm)
        
        # Group data by sensor/device
        sensors_data = {}
        all_timestamps = []
        
        for reading in readings:
            device_id = reading.device_id
            device_name = reading.device.name if reading.device else f'Sensor {device_id}'
            
            if device_id not in sensors_data:
                sensors_data[device_id] = {
                    'device_id': device_id,
                    'device_name': device_name,
                    'device_type': reading.device.device_type if reading.device else 'unknown',
                    'thingsboard_id': str(reading.device.thingsboard_id) if reading.device and reading.device.thingsboard_id else None,
                    'data': [],
                    'keys': set()
                }
            
            # Use JSONField results directly
            results = reading.results or {}
            
            # Track all available keys
            sensors_data[device_id]['keys'].update(results.keys())
            
            # Extract values with flexible key names
            temp = (results.get('temperature') or 
                    results.get('temp') or 
                    reading.temperature)
            
            moisture = (results.get('soilMoisture_%') or 
                       results.get('soilMoisture_adc') or
                       results.get('soilMoisture') or 
                       results.get('soil_moisture') or 
                       results.get('moisture') or 
                       reading.moisture)
            
            humidity = results.get('humidity') or reading.humidity
            
            timestamp = reading.timestamp.isoformat() if reading.timestamp else None
            
            sensors_data[device_id]['data'].append({
                'timestamp': timestamp,
                'temperature': float(temp) if temp is not None else None,
                'moisture': float(moisture) if moisture is not None else None,
                'humidity': float(humidity) if humidity is not None else None,
                'raw': results  # Include raw data for any other keys
            })
            
            if timestamp and timestamp not in all_timestamps:
                all_timestamps.append(timestamp)
        
        # Convert sets to lists for JSON serialization
        for device_id in sensors_data:
            sensors_data[device_id]['keys'] = list(sensors_data[device_id]['keys'])
            sensors_data[device_id]['reading_count'] = len(sensors_data[device_id]['data'])
        
        # Sort timestamps
        all_timestamps.sort()
        
        # Build legacy format for backward compatibility (combined data)
        combined_moisture = []
        combined_temperature = []
        combined_timestamps = []
        
        for reading in readings:
            results = reading.results or {}
            temp = (results.get('temperature') or results.get('temp') or reading.temperature)
            moisture = (results.get('soilMoisture_%') or results.get('soilMoisture_adc') or
                       results.get('soilMoisture') or results.get('soil_moisture') or 
                       results.get('moisture') or reading.moisture)
            
            if temp is not None:
                combined_temperature.append(float(temp))
            if moisture is not None:
                combined_moisture.append(float(moisture))
            combined_timestamps.append(reading.timestamp.isoformat() if reading.timestamp else None)
        
        return Response({
            'farm_name': farm.name if farm else 'All Farms',
            'period': period_label,
            'sensor_count': len(sensors_data),
            'sensors': list(sensors_data.values()),
            # Legacy format for backward compatibility
            'moisture': combined_moisture,
            'temperature': combined_temperature,
            'timestamps': combined_timestamps,
            'available_keys': list(set().union(*[set(s['keys']) for s in sensors_data.values()])) if sensors_data else []
        })


class ThingsBoardTimeSeriesView(APIView):
    """
    Get time-series data directly from ThingsBoard using the telemetry plugin API.
    
    Uses endpoints:
    - GET /api/plugins/telemetry/{entityType}/{entityId}/values/timeseries
    - GET /api/plugins/telemetry/{entityType}/{entityId}/keys/timeseries
    """
    permission_classes = (AllowAny,)

    def get(self, request, device_id=None):
        """
        Get timeseries data from ThingsBoard for a device.
        
        Query params:
        - entity_id: ThingsBoard device UUID (required if no device_id)
        - keys: Comma-separated telemetry keys to retrieve
        - start_ts: Start timestamp in ms (default: 24 hours ago)
        - end_ts: End timestamp in ms (default: now)
        - limit: Max data points per key (default: 100)
        - agg: Aggregation (MIN, MAX, AVG, SUM, COUNT, NONE)
        - interval: Aggregation interval in ms
        """
        entity_id = request.query_params.get('entity_id')
        entity_type = request.query_params.get('entity_type', 'DEVICE')
        
        # Try to get entity_id from device if device_id provided
        if device_id and not entity_id:
            try:
                device = Device.objects.get(id=device_id)
                # Use device's thingsboard_id if available
                entity_id = getattr(device, 'thingsboard_id', None)
                if not entity_id:
                    return Response({
                        'error': f'Device {device_id} has no ThingsBoard entity ID configured',
                        'device_name': device.name,
                        'help': 'Set entity_id query parameter or configure thingsboard_id on device'
                    }, status=status.HTTP_400_BAD_REQUEST)
            except Device.DoesNotExist:
                return Response({'error': f'Device {device_id} not found'}, 
                              status=status.HTTP_404_NOT_FOUND)
        
        if not entity_id:
            return Response({
                'error': 'entity_id query parameter is required',
                'example': '/api/iot/thingsboard-timeseries/?entity_id=YOUR_DEVICE_UUID&keys=temperature,humidity'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        client = ThingsBoardClient()
        
        # Parse optional parameters
        keys = request.query_params.get('keys', '').split(',') if request.query_params.get('keys') else None
        start_ts = request.query_params.get('start_ts')
        end_ts = request.query_params.get('end_ts')
        limit = int(request.query_params.get('limit', 100))
        agg = request.query_params.get('agg')
        interval = request.query_params.get('interval')
        
        # Convert to int if provided
        if start_ts:
            start_ts = int(start_ts)
        if end_ts:
            end_ts = int(end_ts)
        if interval:
            interval = int(interval)
        
        # Default time range: last 24 hours
        if not end_ts:
            end_ts = int(datetime.now().timestamp() * 1000)
        if not start_ts:
            start_ts = end_ts - (24 * 60 * 60 * 1000)  # 24 hours in ms
        
        result = client.get_timeseries(
            entity_type=entity_type,
            entity_id=entity_id,
            keys=keys,
            start_ts=start_ts,
            end_ts=end_ts,
            interval=interval,
            limit=limit,
            agg=agg
        )
        
        if result.get('success'):
            # Format data for charts
            formatted_data = self._format_for_charts(result.get('data', {}))
            return Response({
                'success': True,
                'entity_id': entity_id,
                'entity_type': entity_type,
                'time_range': {
                    'start_ts': start_ts,
                    'end_ts': end_ts,
                    'start': datetime.fromtimestamp(start_ts / 1000).isoformat(),
                    'end': datetime.fromtimestamp(end_ts / 1000).isoformat()
                },
                'raw_data': result.get('data', {}),
                'chart_data': formatted_data
            })
        else:
            return Response({
                'success': False,
                'error': result.get('error'),
                'entity_id': entity_id
            }, status=status.HTTP_400_BAD_REQUEST if 'authentication' in str(result.get('error', '')).lower() else status.HTTP_500_INTERNAL_SERVER_ERROR)
    
    def _format_for_charts(self, data: dict) -> dict:
        """Format ThingsBoard timeseries data for Chart.js"""
        chart_data = {
            'labels': [],
            'datasets': []
        }
        
        all_timestamps = set()
        
        # Collect all unique timestamps
        for key, values in data.items():
            for point in values:
                all_timestamps.add(point.get('ts', 0))
        
        # Sort timestamps
        sorted_ts = sorted(all_timestamps)
        chart_data['labels'] = [datetime.fromtimestamp(ts / 1000).strftime('%Y-%m-%d %H:%M') for ts in sorted_ts]
        
        # Create datasets for each key
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']
        
        for idx, (key, values) in enumerate(data.items()):
            # Create lookup for this key's values
            ts_to_value = {point.get('ts'): point.get('value') for point in values}
            
            dataset = {
                'label': key,
                'data': [ts_to_value.get(ts) for ts in sorted_ts],
                'borderColor': colors[idx % len(colors)],
                'backgroundColor': colors[idx % len(colors)] + '33',  # 20% opacity
                'fill': False,
                'tension': 0.1
            }
            chart_data['datasets'].append(dataset)
        
        return chart_data


class ThingsBoardKeysView(APIView):
    """Get available telemetry keys from ThingsBoard device"""
    permission_classes = (AllowAny,)

    def get(self, request):
        """
        Get available timeseries keys for a ThingsBoard entity.
        
        Query params:
        - entity_id: ThingsBoard device UUID (required)
        - entity_type: Entity type (default: DEVICE)
        """
        entity_id = request.query_params.get('entity_id')
        entity_type = request.query_params.get('entity_type', 'DEVICE')
        
        if not entity_id:
            return Response({
                'error': 'entity_id query parameter is required',
                'example': '/api/iot/thingsboard-keys/?entity_id=YOUR_DEVICE_UUID'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        client = ThingsBoardClient()
        result = client.get_timeseries_keys(entity_type, entity_id)
        
        if result.get('success'):
            return Response({
                'success': True,
                'entity_id': entity_id,
                'entity_type': entity_type,
                'keys': result.get('keys', [])
            })
        else:
            return Response({
                'success': False,
                'error': result.get('error'),
                'entity_id': entity_id
            }, status=status.HTTP_400_BAD_REQUEST)

