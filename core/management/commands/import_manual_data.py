import csv
import os
from django.core.management.base import BaseCommand
from django.contrib.gis.geos import GEOSGeometry, Point
from core.models import Farm, FieldBoundary, Device, Asset
from django.contrib.auth import get_user_model

User = get_user_model()

class Command(BaseCommand):
    help = "Import manual data from CSV files. Place files in a folder and pass the path."

    def add_arguments(self, parser):
        parser.add_argument('path', type=str, help='Directory containing CSVs: farms.csv, fields.csv, devices.csv, assets.csv')

    def handle(self, *args, **options):
        path = options['path']
        # farms.csv columns: owner_username,name,lon,lat
        farms_csv = os.path.join(path, 'farms.csv')
        if os.path.exists(farms_csv):
            with open(farms_csv) as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    owner = User.objects.filter(username=row.get('owner_username')).first()
                    if not owner:
                        self.stdout.write(self.style.WARNING(f"Owner {row.get('owner_username')} not found; skipping farm {row.get('name')}"))
                        continue
                    pt = Point(float(row['lon']), float(row['lat']), srid=4326)
                    farm, _ = Farm.objects.get_or_create(owner=owner, name=row['name'], defaults={'location': pt})
                    self.stdout.write(f"Created/Found farm {farm}")

        # fields.csv columns: farm_name,name,crop_type,boundary_wkt
        fields_csv = os.path.join(path, 'fields.csv')
        if os.path.exists(fields_csv):
            with open(fields_csv) as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    farm = Farm.objects.filter(name=row['farm_name']).first()
                    if not farm:
                        self.stdout.write(self.style.WARNING(f"Farm {row['farm_name']} not found; skipping field {row.get('name')}"))
                        continue
                    geom = GEOSGeometry(row['boundary_wkt'])
                    fld, _ = FieldBoundary.objects.get_or_create(farm=farm, name=row['name'], defaults={'crop_type': row.get('crop_type',''), 'boundary': geom})
                    self.stdout.write(f"Created/Found field {fld}")

        # devices.csv columns: farm_name,field_name,name,thingsboard_id,device_type,token
        devices_csv = os.path.join(path, 'devices.csv')
        if os.path.exists(devices_csv):
            with open(devices_csv) as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    farm = Farm.objects.filter(name=row['farm_name']).first()
                    field = None
                    if row.get('field_name'):
                        field = FieldBoundary.objects.filter(farm__name=row['farm_name'], name=row['field_name']).first()
                    if not farm:
                        self.stdout.write(self.style.WARNING(f"Farm {row['farm_name']} not found; skipping device {row.get('name')}"))
                        continue
                    device, _ = Device.objects.get_or_create(thingsboard_id=row['thingsboard_id'], defaults={
                        'farm': farm,
                        'field': field,
                        'name': row.get('name'),
                        'device_type': row.get('device_type','sensor'),
                        'token': row.get('token',''),
                    })
                    self.stdout.write(f"Created/Found device {device}")

        # assets.csv columns: farm_name,field_name,name,asset_type,serial_number,device_thingsboard_id,lon,lat
        assets_csv = os.path.join(path, 'assets.csv')
        if os.path.exists(assets_csv):
            with open(assets_csv) as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    farm = Farm.objects.filter(name=row['farm_name']).first()
                    if not farm:
                        self.stdout.write(self.style.WARNING(f"Farm {row['farm_name']} not found; skipping asset {row.get('name')}"))
                        continue
                    field = None
                    if row.get('field_name'):
                        field = FieldBoundary.objects.filter(farm=farm, name=row['field_name']).first()
                    device = None
                    if row.get('device_thingsboard_id'):
                        device = Device.objects.filter(thingsboard_id=row['device_thingsboard_id']).first()
                    location = None
                    if row.get('lon') and row.get('lat'):
                        location = Point(float(row['lon']), float(row['lat']), srid=4326)
                    asset, _ = Asset.objects.get_or_create(name=row['name'], farm=farm, defaults={
                        'field': field,
                        'asset_type': row.get('asset_type','iot_sensor'),
                        'serial_number': row.get('serial_number'),
                        'device': device,
                        'location': location
                    })
                    self.stdout.write(f"Created/Found asset {asset}")

        self.stdout.write(self.style.SUCCESS("Import complete"))
