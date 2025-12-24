import csv
from django.core.management.base import BaseCommand
from core.models import Farm, FieldBoundary, Device, Asset

class Command(BaseCommand):
    help = "Export manual data to CSV files."

    def handle(self, *args, **kwargs):
        # Export farms
        with open('farms.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['id', 'owner', 'name', 'location'])
            for farm in Farm.objects.all():
                writer.writerow([farm.id, farm.owner.username, farm.name, farm.location])

        # Export fields
        with open('fields.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['id', 'farm', 'name', 'crop_type', 'boundary'])
            for field in FieldBoundary.objects.all():
                writer.writerow([field.id, field.farm.name, field.name, field.crop_type, field.boundary])

        # Export devices
        with open('devices.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['id', 'farm', 'field', 'name', 'thingsboard_id', 'device_type', 'status'])
            for device in Device.objects.all():
                writer.writerow([device.id, device.farm.name, device.field.name if device.field else '', device.name, device.thingsboard_id, device.device_type, device.status])

        # Export assets
        with open('assets.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['id', 'farm', 'field', 'name', 'asset_type', 'serial_number', 'device', 'location'])
            for asset in Asset.objects.all():
                writer.writerow([asset.id, asset.farm.name, asset.field.name if asset.field else '', asset.name, asset.asset_type, asset.serial_number, asset.device.name if asset.device else '', asset.location])

        self.stdout.write(self.style.SUCCESS('Data exported successfully to farms.csv, fields.csv, devices.csv, and assets.csv'))