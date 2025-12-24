"""
Sync IoT data from ThingsBoard using device tokens (GET endpoints only)
"""
from django.core.management.base import BaseCommand
from core.models import Device, Farm
from iot.thingsboard_client import FarmIoTManager, ThingsBoardClient, sync_farm_devices


class Command(BaseCommand):
    help = 'Sync IoT data from ThingsBoard using device tokens (GET endpoints only)'

    def add_arguments(self, parser):
        parser.add_argument(
            '--farm',
            type=int,
            help='Sync devices for a specific farm ID',
        )
        parser.add_argument(
            '--device',
            type=int,
            help='Sync a specific device ID',
        )
        parser.add_argument(
            '--token',
            type=str,
            help='Test a specific device token',
        )
        parser.add_argument(
            '--all',
            action='store_true',
            help='Sync all devices',
        )

    def handle(self, *args, **options):
        self.stdout.write(self.style.NOTICE("ThingsBoard IoT Sync (GET endpoints only)"))
        self.stdout.write("-" * 50)
        
        # Test a specific token
        if options.get('token'):
            self.test_token(options['token'])
            return
        
        # Sync specific device
        if options.get('device'):
            self.sync_single_device(options['device'])
            return
        
        # Sync specific farm
        if options.get('farm'):
            self.sync_farm(options['farm'])
            return
        
        # Sync all devices
        if options.get('all'):
            self.sync_all()
            return
        
        # Default: show status and help
        self.show_status()
    
    def test_token(self, token):
        """Test a device token"""
        self.stdout.write(f"Testing token: {token[:8]}...")
        
        client = ThingsBoardClient()
        result = client.get_latest_telemetry(token)
        
        if result.get('success'):
            self.stdout.write(self.style.SUCCESS("Token is valid!"))
            self.stdout.write(f"Telemetry data: {result.get('data', {})}")
        else:
            self.stdout.write(self.style.ERROR(f"Token test failed: {result.get('error')}"))
    
    def sync_single_device(self, device_id):
        """Sync a single device"""
        try:
            device = Device.objects.get(id=device_id)
        except Device.DoesNotExist:
            self.stdout.write(self.style.ERROR(f"Device {device_id} not found"))
            return
        
        if not device.token:
            self.stdout.write(self.style.WARNING(f"Device '{device.name}' has no token configured"))
            return
        
        self.stdout.write(f"Syncing device: {device.name}")
        
        manager = FarmIoTManager()
        result = manager.sync_device_data(device)
        
        if result.get('success'):
            self.stdout.write(self.style.SUCCESS(
                f"✓ Synced {result.get('readings_created', 0)} reading(s)"
            ))
        else:
            self.stdout.write(self.style.ERROR(f"✗ Error: {result.get('error')}"))
    
    def sync_farm(self, farm_id):
        """Sync all devices for a farm"""
        try:
            farm = Farm.objects.get(id=farm_id)
        except Farm.DoesNotExist:
            self.stdout.write(self.style.ERROR(f"Farm {farm_id} not found"))
            return
        
        self.stdout.write(f"Syncing devices for farm: {farm.name}")
        
        result = sync_farm_devices(farm_id)
        
        self.stdout.write("-" * 30)
        self.stdout.write(f"Total devices: {result.get('total_devices', 0)}")
        self.stdout.write(self.style.SUCCESS(f"Successful: {result.get('successful', 0)}"))
        self.stdout.write(self.style.WARNING(f"Failed: {result.get('failed', 0)}"))
        self.stdout.write(f"Readings created: {result.get('readings_created', 0)}")
        
        if result.get('errors'):
            self.stdout.write("\nErrors:")
            for err in result['errors']:
                self.stdout.write(self.style.ERROR(f"  - {err.get('device')}: {err.get('error')}"))
    
    def sync_all(self):
        """Sync all devices"""
        self.stdout.write("Syncing ALL devices...")
        
        manager = FarmIoTManager()
        result = manager.sync_all_devices()
        
        self.stdout.write("-" * 30)
        self.stdout.write(f"Total devices: {result.get('total_devices', 0)}")
        self.stdout.write(self.style.SUCCESS(f"Successful: {result.get('successful', 0)}"))
        self.stdout.write(self.style.WARNING(f"Failed: {result.get('failed', 0)}"))
        self.stdout.write(f"Readings created: {result.get('readings_created', 0)}")
        
        if result.get('errors'):
            self.stdout.write("\nErrors:")
            for err in result['errors'][:10]:  # Show first 10 errors
                self.stdout.write(self.style.ERROR(f"  - {err.get('device')}: {err.get('error')}"))
    
    def show_status(self):
        """Show current device status"""
        devices = Device.objects.all()
        devices_with_token = devices.filter(token__isnull=False).exclude(token='')
        
        self.stdout.write(f"Total devices: {devices.count()}")
        self.stdout.write(f"Devices with ThingsBoard tokens: {devices_with_token.count()}")
        
        if devices_with_token.exists():
            self.stdout.write("\nDevices with tokens:")
            for device in devices_with_token[:10]:
                self.stdout.write(f"  - {device.name} ({device.device_type}): {device.token[:8]}...")
        
        self.stdout.write("\n" + "=" * 50)
        self.stdout.write("Usage:")
        self.stdout.write("  python manage.py sync_thingsboard --all           # Sync all devices")
        self.stdout.write("  python manage.py sync_thingsboard --farm 1        # Sync farm ID 1")
        self.stdout.write("  python manage.py sync_thingsboard --device 1      # Sync device ID 1")
        self.stdout.write("  python manage.py sync_thingsboard --token ABC123  # Test a token")
