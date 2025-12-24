import os
from django.core.management.base import BaseCommand
from geoai import calculate_ndvi, calculate_ndwi
from imagery.models import SentinelImage
from minio import Minio

class Command(BaseCommand):
    help = 'Process Sentinel-2 imagery to calculate NDVI and NDWI using geoai.'

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

                # Process image using geoai
                ndvi_path = f'/tmp/ndvi_{object_name}'
                ndwi_path = f'/tmp/ndwi_{object_name}'
                calculate_ndvi(local_path, ndvi_path)
                calculate_ndwi(local_path, ndwi_path)

                # Upload results back to MinIO
                minio_client.fput_object(bucket_name, f'ndvi/{object_name}', ndvi_path)
                minio_client.fput_object(bucket_name, f'ndwi/{object_name}', ndwi_path)

                # Mark image as processed
                image.processed = True
                image.save()

                self.stdout.write(self.style.SUCCESS(f'Processed {object_name} successfully.'))

            except Exception as e:
                self.stdout.write(self.style.ERROR(f'Failed to process {object_name}: {e}'))