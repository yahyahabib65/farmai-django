import os
import rasterio
from rasterio.mask import mask
from shapely.geometry import shape
from django.core.management.base import BaseCommand
from geoai import calculate_ndvi
from imagery.models import SentinelImage
from core.models import FieldBoundary
from minio import Minio

class Command(BaseCommand):
    help = 'Process Sentinel-2 imagery to calculate NDVI based on farm boundaries.'

    def handle(self, *args, **kwargs):
        # Initialize MinIO client
        minio_client = Minio(
            os.getenv('MINIO_ENDPOINT', 'localhost:9000'),
            access_key=os.getenv('MINIO_ACCESS_KEY', 'minioadmin'),
            secret_key=os.getenv('MINIO_SECRET_KEY', 'minioadmin'),
            secure=False
        )

        # Fetch unprocessed Sentinel images
        images = SentinelImage.objects.filter(processed=False)
        for image in images:
            try:
                # Download image from MinIO
                bucket_name = 'sentinel'
                object_name = image.file_name
                local_path = f'/tmp/{object_name}'
                minio_client.fget_object(bucket_name, object_name, local_path)

                # Retrieve farm boundaries
                field_boundaries = FieldBoundary.objects.filter(farm=image.farm)
                if not field_boundaries.exists():
                    self.stdout.write(self.style.WARNING(f'No boundaries found for farm {image.farm.name}'))
                    continue

                # Clip image to farm boundaries
                with rasterio.open(local_path) as src:
                    boundary_shapes = [shape(boundary.geometry) for boundary in field_boundaries]
                    out_image, out_transform = mask(src, boundary_shapes, crop=True)
                    out_meta = src.meta.copy()
                    out_meta.update({
                        'driver': 'GTiff',
                        'height': out_image.shape[1],
                        'width': out_image.shape[2],
                        'transform': out_transform
                    })

                clipped_path = f'/tmp/clipped_{object_name}'
                with rasterio.open(clipped_path, 'w', **out_meta) as dest:
                    dest.write(out_image)

                # Process clipped image using geoai
                ndvi_path = f'/tmp/ndvi_{object_name}'
                calculate_ndvi(clipped_path, ndvi_path)

                # Upload NDVI result back to MinIO
                minio_client.fput_object(bucket_name, f'ndvi/{object_name}', ndvi_path)

                # Mark image as processed
                image.processed = True
                image.save()

                self.stdout.write(self.style.SUCCESS(f'Processed {object_name} successfully.'))

            except Exception as e:
                self.stdout.write(self.style.ERROR(f'Failed to process {object_name}: {e}'))