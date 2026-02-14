from rest_framework import viewsets, status
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import AllowAny
from .models import SensorReading, WeatherData
from .serializers import SensorReadingSerializer
from .filters import KalmanFilter
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
    """
    Get time series sensor data for charts — fully dynamic keys,
    outlier filtering, downsampling, custom date ranges and Kalman smoothing.
    """
    permission_classes = (AllowAny,)

    # Maximum points to send to the chart (prevents browser lag)
    MAX_CHART_POINTS = 800

    def get(self, request, farm_id=None):
        """
        GET /api/iot/timeseries/{farm_id}/
        Query params:
          days   = 7 | 30 | all | ...  (simple preset)
          start  = ISO date or epoch‑ms (custom range start)
          end    = ISO date or epoch‑ms (custom range end)
        """
        if farm_id:
            try:
                farm = Farm.objects.get(id=farm_id)
            except Farm.DoesNotExist:
                return Response({'error': 'Farm not found'}, status=status.HTTP_404_NOT_FOUND)
        else:
            farm = Farm.objects.first()

        from django.utils import timezone as tz
        import numpy as np

        # ── 1. Resolve date range ──────────────────────────────
        start_param = request.query_params.get('start')
        end_param = request.query_params.get('end')
        days_param = request.query_params.get('days', '7')

        now = tz.now()

        if start_param and end_param:
            start_date = self._parse_dt(start_param)
            end_date = self._parse_dt(end_param)
            period_label = f'{start_date:%Y-%m-%d} to {end_date:%Y-%m-%d}'
        elif days_param == 'all':
            start_date = None
            end_date = None
            period_label = 'All Time'
        else:
            try:
                days = int(days_param)
            except (ValueError, TypeError):
                days = 7
            start_date = now - timedelta(days=days)
            end_date = None
            if days < 1:
                hours = max(int(float(days_param) * 24), 1)
                period_label = f'Last {hours} hours'
            else:
                period_label = f'Last {days} day{"s" if days != 1 else ""}'

        # ── 2. Query readings ──────────────────────────────────
        qs = SensorReading.objects.filter(device__farm=farm).select_related('device').order_by('timestamp')
        if start_date:
            qs = qs.filter(timestamp__gte=start_date)
        if end_date:
            qs = qs.filter(timestamp__lte=end_date)
        readings = list(qs)

        if not readings:
            return Response({
                'farm_name': farm.name if farm else 'All Farms',
                'period': period_label,
                'sensor_count': 0,
                'sensors': [],
                'timestamps': [],
                'available_keys': [],
            })

        # ── 3. Group by device, keep ALL raw keys ─────────────
        sensors_data = {}
        for reading in readings:
            did = reading.device_id
            if did not in sensors_data:
                sensors_data[did] = {
                    'device_id': did,
                    'device_name': reading.device.name if reading.device else f'Sensor {did}',
                    'device_type': reading.device.device_type if reading.device else 'unknown',
                    'thingsboard_id': str(reading.device.thingsboard_id) if reading.device and reading.device.thingsboard_id else None,
                    'data': [],
                    'keys': set(),
                }

            results = reading.results or {}
            sensors_data[did]['keys'].update(results.keys())

            # Store every key as a numeric value (if castable)
            point = {'timestamp': reading.timestamp.isoformat() if reading.timestamp else None}
            for k, v in results.items():
                try:
                    point[k] = float(v)
                except (ValueError, TypeError):
                    point[k] = v  # keep string values for metadata
            sensors_data[did]['data'].append(point)

        # ── 4. Outlier removal (per-key IQR) ──────────────────
        for did, info in sensors_data.items():
            numeric_keys = [k for k in info['keys'] if k != 'timestamp']
            for key in numeric_keys:
                vals = [p.get(key) for p in info['data'] if isinstance(p.get(key), (int, float))]
                if len(vals) < 4:
                    continue
                q1 = float(np.percentile(vals, 25))
                q3 = float(np.percentile(vals, 75))
                iqr = q3 - q1
                lower = q1 - 3.0 * iqr
                upper = q3 + 3.0 * iqr
                for p in info['data']:
                    v = p.get(key)
                    if isinstance(v, (int, float)) and (v < lower or v > upper):
                        p[key] = None  # remove outlier — chart will gap

        # ── 5. Downsample if too many points ──────────────────
        for did, info in sensors_data.items():
            n = len(info['data'])
            if n > self.MAX_CHART_POINTS:
                step = max(1, n // self.MAX_CHART_POINTS)
                info['data'] = info['data'][::step]

        # ── 6. Kalman smoothing (per device per numeric key) ──
        for did, info in sensors_data.items():
            numeric_keys = [k for k in info['keys'] if k != 'timestamp']
            kalman_filters = {}
            for key in numeric_keys:
                kalman_filters[key] = KalmanFilter(R=10.0, Q=0.1)

            for point in info['data']:
                for key in numeric_keys:
                    raw = point.get(key)
                    if isinstance(raw, (int, float)):
                        point[f'{key}_kalman'] = round(kalman_filters[key].filter(raw), 2)
                    else:
                        point[f'{key}_kalman'] = None

        # ── 7. Build flat timestamps list ─────────────────────
        all_timestamps = sorted({p['timestamp'] for info in sensors_data.values() for p in info['data'] if p.get('timestamp')})

        # ── 8. Serialise ──────────────────────────────────────
        for did in sensors_data:
            sensors_data[did]['keys'] = sorted(sensors_data[did]['keys'])
            sensors_data[did]['reading_count'] = len(sensors_data[did]['data'])

        response_data = list(sensors_data.values())

        return Response({
            'farm_name': farm.name if farm else 'All Farms',
            'period': period_label,
            'sensor_count': len(sensors_data),
            'sensors': response_data,
            'devices': response_data,
            'timestamps': all_timestamps,
            'available_keys': sorted({k for info in sensors_data.values() for k in info['keys']}),
        })

    # ── helpers ────────────────────────────────────────────────
    @staticmethod
    def _parse_dt(val):
        """Parse ISO string or epoch-ms to timezone-aware datetime."""
        from django.utils import timezone as tz
        try:
            ms = int(val)
            return datetime.fromtimestamp(ms / 1000, tz=tz.utc)
        except (ValueError, TypeError):
            pass
        from django.utils.dateparse import parse_datetime, parse_date
        dt = parse_datetime(val)
        if dt:
            return dt if dt.tzinfo else tz.make_aware(dt)
        d = parse_date(val)
        if d:
            return tz.make_aware(datetime.combine(d, datetime.min.time()))
        raise ValueError(f'Cannot parse date: {val}')


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

