import os
import tempfile
import numpy as np
import rasterio
from rasterio.mask import mask
from minio import Minio
from django.core.management.base import BaseCommand
from core.models import Farm
from imagery.models import ProcessedRaster
from analytics.models import AnalyticsResult

# MinIO Client
client = Minio(
    "localhost:9000",
    access_key="minioadmin",
    secret_key="minioadmin",
    secure=False
)

class Command(BaseCommand):
    help = 'Runs AI analysis on the latest farm imagery'

    def add_arguments(self, parser):
        parser.add_argument('farm_id', type=int)

    def handle(self, *args, **options):
        farm_id = options['farm_id']
        try:
            farm = Farm.objects.get(id=farm_id)
        except Farm.DoesNotExist:
            self.stdout.write(self.style.ERROR(f"Farm {farm_id} not found"))
            return

        # 1. Get the latest NDVI Raster for this farm
        raster_obj = ProcessedRaster.objects.filter(source__farm=farm).last()
        if not raster_obj:
            self.stdout.write(self.style.WARNING("No processed raster found for this farm."))
            return

        self.stdout.write(f"Analyzing Raster: {raster_obj.minio_path}...")

        # 2. Download Raster to Temp File (Rasterio needs a file on disk)
        bucket_name = "farm-data"
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
            client.fget_object(bucket_name, raster_obj.minio_path, tmp.name)
            tmp_path = tmp.name

        try:
            with rasterio.open(tmp_path) as src:
                # 3. Iterate through every field in the farm
                for field in farm.fields.all():
                    try:
                        # Convert PostGIS Polygon to GeoJSON format for Rasterio
                        # IMPORTANT: Flip coordinates if needed (PostGIS is Lon/Lat, Rasterio expects same CRS)
                        geoms = [field.boundary.json] # uses GeoJSON string representation
                        
                        # Parse the JSON string back to dict for rasterio
                        import json
                        geom_dict = json.loads(geoms[0])
                        
                        # 4. Mask: Cut out the part of the image inside the field
                        out_image, out_transform = mask(src, [geom_dict], crop=True)
                        
                        # 5. Calculate Average (Ignore 0/Nodata values)
                        data = out_image[0] # Band 1
                        # Filter out zero/masked values
                        valid_pixels = data[data != 0]
                        
                        if valid_pixels.size == 0:
                            avg_ndvi = 0.0
                        else:
                            avg_ndvi = float(np.mean(valid_pixels))

                        # 6. Irrigation Logic (The AI Decision)
                        # Rule: If NDVI < 0.3, the crop is stressed/dead/dry.
                        alert = False
                        notes = "Healthy"
                        if avg_ndvi < 0.3:
                            alert = True
                            notes = "CRITICAL: Low vegetation index. Check water levels."
                        elif avg_ndvi < 0.5:
                            notes = "Warning: Moderate growth."

                        # 7. Save Result
                        AnalyticsResult.objects.create(
                            field=field,
                            avg_ndvi=avg_ndvi,
                            irrigation_alert=alert,
                            notes=notes
                        )
                        self.stdout.write(self.style.SUCCESS(f"Field '{field.name}': NDVI {avg_ndvi:.2f} -> {notes}"))

                    except Exception as e:
                        self.stdout.write(self.style.ERROR(f"Error processing field {field.name}: {e}"))
        
        finally:
            # Cleanup temp file
            if os.path.exists(tmp_path):
                os.remove(tmp_path)