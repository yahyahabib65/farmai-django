"""
Download satellite imagery from Microsoft Planetary Computer
Uses farm boundaries as Area of Interest (AOI)
Stores images in MinIO organized by date
"""
from datetime import datetime, timedelta
from django.core.management.base import BaseCommand
from core.models import Farm, FieldBoundary
from imagery.planetary_downloader import PlanetaryComputerDownloader


class Command(BaseCommand):
    help = 'Download satellite imagery from Planetary Computer using farm boundaries as AOI'

    def add_arguments(self, parser):
        parser.add_argument(
            '--farm',
            type=int,
            help='Download imagery for a specific farm ID',
        )
        parser.add_argument(
            '--all',
            action='store_true',
            help='Download imagery for all farms with defined boundaries',
        )
        parser.add_argument(
            '--date',
            type=str,
            help='Target date for imagery (YYYY-MM-DD format)',
        )
        parser.add_argument(
            '--days-back',
            type=int,
            default=30,
            help='Number of days to search back for imagery (default: 30)',
        )
        parser.add_argument(
            '--cloud-cover',
            type=int,
            default=30,
            help='Maximum cloud cover percentage (default: 30)',
        )
        parser.add_argument(
            '--list-available',
            action='store_true',
            help='List available imagery dates without downloading',
        )

    def handle(self, *args, **options):
        self.stdout.write(self.style.NOTICE("Planetary Computer Satellite Downloader"))
        self.stdout.write("=" * 50)
        self.stdout.write("Uses farm boundaries as Area of Interest (AOI)")
        self.stdout.write("Stores images in MinIO organized by date")
        self.stdout.write("=" * 50)
        
        downloader = PlanetaryComputerDownloader()
        
        # Parse target date if provided
        target_date = None
        if options.get('date'):
            try:
                target_date = datetime.strptime(options['date'], '%Y-%m-%d')
            except ValueError:
                self.stdout.write(self.style.ERROR("Invalid date format. Use YYYY-MM-DD"))
                return
        
        # List available dates
        if options.get('list_available'):
            self.list_available_dates(downloader, options.get('farm'), options.get('days_back', 90))
            return
        
        # Download for specific farm
        if options.get('farm'):
            self.download_for_farm(downloader, options['farm'], target_date)
            return
        
        # Download for all farms
        if options.get('all'):
            self.download_for_all_farms(downloader, target_date)
            return
        
        # Default: show help and status
        self.show_status()
    
    def list_available_dates(self, downloader, farm_id, days_back):
        """List available satellite imagery dates for a farm"""
        if farm_id:
            farms = Farm.objects.filter(id=farm_id)
        else:
            farms = Farm.objects.all()
        
        for farm in farms:
            self.stdout.write(f"\n📍 Farm: {farm.name}")
            
            # Check if farm has field boundaries
            fields = FieldBoundary.objects.filter(farm=farm)
            if not fields.exists():
                self.stdout.write(self.style.WARNING("  ⚠ No field boundaries defined"))
                continue
            
            dates = downloader.get_available_dates(farm, days_back)
            
            if dates:
                self.stdout.write(f"  Found {len(dates)} available images:")
                for d in dates[:10]:  # Show first 10
                    self.stdout.write(f"    📅 {d['date']} - Cloud cover: {d['cloud_cover']}%")
            else:
                self.stdout.write(self.style.WARNING("  No images found in the specified date range"))
    
    def download_for_farm(self, downloader, farm_id, target_date=None):
        """Download satellite imagery for a specific farm"""
        try:
            farm = Farm.objects.get(id=farm_id)
        except Farm.DoesNotExist:
            self.stdout.write(self.style.ERROR(f"Farm {farm_id} not found"))
            return
        
        self.stdout.write(f"\n🚀 Downloading imagery for farm: {farm.name}")
        
        # Check for field boundaries
        fields = FieldBoundary.objects.filter(farm=farm)
        if not fields.exists():
            self.stdout.write(self.style.ERROR("No field boundaries defined for this farm"))
            self.stdout.write("Add field boundaries first to define the Area of Interest (AOI)")
            return
        
        self.stdout.write(f"  📐 Using {fields.count()} field boundary(ies) as AOI")
        
        # Download
        result = downloader.download_imagery_for_farm(farm, target_date=target_date)
        
        if result.get('success'):
            self.stdout.write(self.style.SUCCESS(f"\n✓ Download completed!"))
            self.stdout.write(f"  📅 Image date: {result.get('image_date')}")
            self.stdout.write(f"  ☁ Cloud cover: {result.get('cloud_cover')}%")
            self.stdout.write(f"  📦 Bands downloaded: {result.get('images_downloaded')}")
            self.stdout.write(f"  🆔 Scene ID: {result.get('image_id')}")
            
            if result.get('minio_paths'):
                self.stdout.write(f"\n  MinIO storage paths:")
                for path in result['minio_paths']:
                    self.stdout.write(f"    - {path}")
        else:
            self.stdout.write(self.style.ERROR(f"\n✗ Download failed: {result.get('error')}"))
    
    def download_for_all_farms(self, downloader, target_date=None):
        """Download imagery for all farms with defined boundaries"""
        farms_with_boundaries = Farm.objects.filter(
            fields__isnull=False
        ).distinct()
        
        if not farms_with_boundaries.exists():
            self.stdout.write(self.style.ERROR("No farms with field boundaries found"))
            return
        
        self.stdout.write(f"Found {farms_with_boundaries.count()} farms with boundaries\n")
        
        successful = 0
        failed = 0
        
        for farm in farms_with_boundaries:
            self.stdout.write(f"Processing: {farm.name}")
            
            result = downloader.download_imagery_for_farm(farm, target_date=target_date)
            
            if result.get('success'):
                self.stdout.write(self.style.SUCCESS(f"  ✓ Downloaded {result.get('images_downloaded')} bands"))
                successful += 1
            else:
                self.stdout.write(self.style.ERROR(f"  ✗ {result.get('error')}"))
                failed += 1
        
        self.stdout.write("\n" + "=" * 50)
        self.stdout.write(f"Summary: {successful} successful, {failed} failed")
    
    def show_status(self):
        """Show current status and usage instructions"""
        total_farms = Farm.objects.count()
        farms_with_boundaries = Farm.objects.filter(fields__isnull=False).distinct().count()
        
        self.stdout.write(f"\nFarm Status:")
        self.stdout.write(f"  Total farms: {total_farms}")
        self.stdout.write(f"  Farms with boundaries (can download): {farms_with_boundaries}")
        
        if farms_with_boundaries < total_farms:
            self.stdout.write(self.style.WARNING(
                f"\n  ⚠ {total_farms - farms_with_boundaries} farm(s) need field boundaries"
            ))
        
        self.stdout.write("\n" + "=" * 50)
        self.stdout.write("Usage:")
        self.stdout.write("  python manage.py download_satellite --all")
        self.stdout.write("  python manage.py download_satellite --farm 1")
        self.stdout.write("  python manage.py download_satellite --farm 1 --date 2024-01-15")
        self.stdout.write("  python manage.py download_satellite --list-available --farm 1")
        self.stdout.write("\nOptions:")
        self.stdout.write("  --days-back 60     Search for imagery in last 60 days")
        self.stdout.write("  --cloud-cover 20   Only download images with <20% cloud cover")
