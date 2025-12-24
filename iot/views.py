from rest_framework import viewsets, status
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import AllowAny
from .models import SensorReading, WeatherData
from .serializers import SensorReadingSerializer
from .thingsboard_client import ThingsBoardClient, FarmIoTManager, sync_farm_devices
from core.models import Device, Farm
from django.db.models import Avg
from datetime import datetime, timedelta


class SensorReadingViewSet(viewsets.ModelViewSet):
	queryset = SensorReading.objects.select_related('device').all()
	serializer_class = SensorReadingSerializer
	permission_classes = (AllowAny,)


class ThingsBoardSyncView(APIView):
    """Sync IoT data from ThingsBoard using device tokens (GET endpoints only)"""
    permission_classes = (AllowAny,)

    def get(self, request):
        """Get sync status and device info"""
        devices = Device.objects.all()
        devices_with_token = devices.filter(token__isnull=False).exclude(token='')
        
        return Response({
            'message': 'ThingsBoard IoT Sync',
            'thingsboard_url': 'http://icarus.lums.edu.pk',
            'total_devices': devices.count(),
            'devices_with_tokens': devices_with_token.count(),
            'devices': [{
                'id': d.id,
                'name': d.name,
                'type': d.device_type,
                'has_token': bool(d.token),
                'farm': d.farm.name if d.farm else None
            } for d in devices[:20]],
            'instructions': 'POST to sync data from ThingsBoard',
            'parameters': {
                'farm_id': 'Optional - sync all devices for a farm',
                'device_id': 'Optional - sync a specific device',
                'token': 'Optional - test a specific token'
            }
        })

    def post(self, request):
        """Sync data from ThingsBoard"""
        farm_id = request.data.get('farm_id')
        device_id = request.data.get('device_id')
        token = request.data.get('token')
        
        client = ThingsBoardClient()
        manager = FarmIoTManager()
        
        # Test a specific token
        if token:
            result = client.get_latest_telemetry(token)
            return Response({
                'action': 'test_token',
                'success': result.get('success'),
                'data': result.get('data', {}),
                'error': result.get('error')
            })
        
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
            result = sync_farm_devices(farm_id)
            return Response({
                'action': 'sync_farm',
                'farm_id': farm_id,
                **result
            })
        
        # Sync all devices
        result = manager.sync_all_devices()
        return Response({
            'action': 'sync_all',
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
    """Get time series sensor data for charts"""
    permission_classes = (AllowAny,)

    def get(self, request, farm_id=None):
        """Get time series data for the last 7 days"""
        if farm_id:
            try:
                farm = Farm.objects.get(id=farm_id)
            except Farm.DoesNotExist:
                return Response({'error': 'Farm not found'}, status=status.HTTP_404_NOT_FOUND)
        else:
            farm = Farm.objects.first()
        
        days = int(request.query_params.get('days', 7))
        
        # Get sensor readings
        from django.utils import timezone
        start_date = timezone.now() - timedelta(days=days)
        
        readings = SensorReading.objects.filter(
            device__farm=farm,
            timestamp__gte=start_date
        ).order_by('timestamp')
        
        weather = WeatherData.objects.filter(
            farm=farm,
            timestamp__gte=start_date
        ).order_by('timestamp')
        
        # Format for charts - using JSONField results
        time_series = {
            'farm_name': farm.name if farm else 'All Farms',
            'period': f'Last {days} days',
            'moisture': [],
            'temperature': [],
            'humidity': [],
            'timestamps': []
        }
        
        for reading in readings:
            # Use JSONField results or fallback to helper properties
            results = reading.results or {}
            temp = results.get('temperature') or reading.temperature
            moisture = results.get('moisture') or results.get('soil_moisture') or reading.moisture
            humidity = results.get('humidity') or reading.humidity
            
            if temp is not None:
                time_series['temperature'].append(temp)
            if moisture is not None:
                time_series['moisture'].append(moisture)
            if humidity is not None:
                time_series['humidity'].append(humidity)
            
            time_series['timestamps'].append(reading.timestamp.isoformat() if reading.timestamp else None)
        
        # Add weather humidity data
        for w in weather:
            if w.humidity and w.humidity not in time_series['humidity']:
                time_series['humidity'].append(w.humidity)
        
        return Response(time_series)


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

