from rest_framework import viewsets, status
from rest_framework.permissions import AllowAny
from rest_framework.views import APIView
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response
import pandas as pd
from django.http import JsonResponse
from core.models import Farm
from datetime import datetime

from .models import Farm, FieldBoundary, Device, Asset
from .serializers import FarmSerializer, FieldBoundarySerializer, DeviceSerializer, AssetSerializer
from django.contrib.gis.geos import GEOSGeometry, Point
from django.contrib.auth import get_user_model


class FarmViewSet(viewsets.ModelViewSet):
	queryset = Farm.objects.all()
	serializer_class = FarmSerializer
	permission_classes = (AllowAny,)


class FieldBoundaryViewSet(viewsets.ModelViewSet):
	queryset = FieldBoundary.objects.all()
	serializer_class = FieldBoundarySerializer
	permission_classes = (AllowAny,)


class DeviceViewSet(viewsets.ModelViewSet):
	queryset = Device.objects.all()
	serializer_class = DeviceSerializer
	permission_classes = (AllowAny,)


class AssetViewSet(viewsets.ModelViewSet):
	queryset = Asset.objects.all()
	serializer_class = AssetSerializer
	permission_classes = (AllowAny,)


class ManualImportView(APIView):
	"""Accept an Excel file (multipart/form-data) with sheets: farms, fields, devices, assets

	Each sheet should follow columns used by the CLI importer:
	  farms: owner_username,name,lon,lat
	  fields: farm_name,name,crop_type,boundary_wkt
	  devices: farm_name,field_name,name,thingsboard_id,device_type,token
	  assets: farm_name,field_name,name,asset_type,serial_number,device_thingsboard_id,lon,lat
	"""
	parser_classes = (MultiPartParser, FormParser)
	permission_classes = (AllowAny,)

	def post(self, request, format=None):
		file_obj = request.FILES.get('file')
		if not file_obj:
			return Response({'detail': 'No file uploaded (use form field "file" )'}, status=status.HTTP_400_BAD_REQUEST)

		try:
			sheets = pd.read_excel(file_obj, sheet_name=None)
		except Exception as e:
			return Response({'detail': f'Failed to read Excel: {e}'}, status=status.HTTP_400_BAD_REQUEST)

		User = get_user_model()
		created = { 'farms': 0, 'fields': 0, 'devices': 0, 'assets': 0 }

		# farms
		df = sheets.get('farms') or sheets.get('Farms')
		if df is not None:
			for _, row in df.fillna('').iterrows():
				owner = User.objects.filter(username=row.get('owner_username')).first()
				if not owner:
					continue
				try:
					pt = Point(float(row['lon']), float(row['lat']), srid=4326)
				except Exception:
					pt = None
				farm, _ = Farm.objects.get_or_create(owner=owner, name=row.get('name',''), defaults={'location': pt})
				created['farms'] += 1

		# fields
		df = sheets.get('fields') or sheets.get('Fields')
		if df is not None:
			for _, row in df.fillna('').iterrows():
				farm = Farm.objects.filter(name=row.get('farm_name')).first()
				if not farm:
					continue
				try:
					geom = GEOSGeometry(row['boundary_wkt']) if row.get('boundary_wkt') else None
				except Exception:
					geom = None
				FieldBoundary.objects.get_or_create(farm=farm, name=row.get('name',''), defaults={'crop_type': row.get('crop_type',''), 'boundary': geom})
				created['fields'] += 1

		# devices
		df = sheets.get('devices') or sheets.get('Devices')
		if df is not None:
			for _, row in df.fillna('').iterrows():
				farm = Farm.objects.filter(name=row.get('farm_name')).first()
				if not farm:
					continue
				field = None
				if row.get('field_name'):
					field = FieldBoundary.objects.filter(farm__name=row.get('farm_name'), name=row.get('field_name')).first()
				Device.objects.get_or_create(thingsboard_id=row.get('thingsboard_id'), defaults={
					'farm': farm,
					'field': field,
					'name': row.get('name',''),
					'device_type': row.get('device_type','sensor'),
					'token': row.get('token',''),
				})
				created['devices'] += 1

		# assets
		df = sheets.get('assets') or sheets.get('Assets')
		if df is not None:
			for _, row in df.fillna('').iterrows():
				farm = Farm.objects.filter(name=row.get('farm_name')).first()
				if not farm:
					continue
				field = None
				if row.get('field_name'):
					field = FieldBoundary.objects.filter(farm=farm, name=row.get('field_name')).first()
				device = None
				if row.get('device_thingsboard_id'):
					device = Device.objects.filter(thingsboard_id=row.get('device_thingsboard_id')).first()
				location = None
				try:
					if row.get('lon') and row.get('lat'):
						location = Point(float(row['lon']), float(row['lat']), srid=4326)
				except Exception:
					location = None
				Asset.objects.get_or_create(name=row.get('name',''), farm=farm, defaults={
					'field': field,
					'asset_type': row.get('asset_type','iot_sensor'),
					'serial_number': row.get('serial_number'),
					'device': device,
					'location': location
				})
				created['assets'] += 1

		return Response({'created': created}, status=status.HTTP_200_OK)


def farm_data_view(request, farm_id):
    try:
        farm = Farm.objects.get(id=farm_id)
        # Example data; replace with actual farm-specific data
        data = {
            'name': farm.name,
            'location': farm.location,
            'ndvi': 'NDVI data placeholder',
        }
        return JsonResponse(data)
    except Farm.DoesNotExist:
        return JsonResponse({'error': 'Farm not found'}, status=404)


def farm_filter_view(request):
    farm_id = request.GET.get('farm')
    start_date = request.GET.get('start_date')
    end_date = request.GET.get('end_date')
    analytics = request.GET.getlist('analytics')

    try:
        farm = Farm.objects.get(id=farm_id) if farm_id else None
        # Example filtered data; replace with actual logic
        data = {
            'farm': farm.name if farm else 'All Farms',
            'start_date': start_date,
            'end_date': end_date,
            'analytics': analytics,
            'results': 'Filtered data placeholder',
        }
        return JsonResponse(data)
    except Farm.DoesNotExist:
        return JsonResponse({'error': 'Farm not found'}, status=404)
