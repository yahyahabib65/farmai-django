import rasterio
import numpy as np
import io
from celery import shared_task
from minio import Minio
from django.contrib.gis.geos import Polygon
from .models import DroneImage, ProcessedRaster

# Configure MinIO Client (Must match your settings)
client = Minio(
    "localhost:9000",
    access_key="minioadmin",
    secret_key="minioadmin",
    secure=False
)

@shared_task
def process_ndvi(image_id):
    try:
        # 1. Fetch the image record from DB
        drone_img = DroneImage.objects.get(id=image_id)
        bucket_name = "farm-data"
        
        # 2. Download image bytes from MinIO
        response = client.get_object(bucket_name, drone_img.minio_path)
        file_data = io.BytesIO(response.read())
        response.close()
        
        # 3. Open with Rasterio
        with rasterio.open(file_data) as src:
            # Assumes Red is Band 1, NIR is Band 2 (Adjust if using different drone)
            red = src.read(1).astype('float32')
            nir = src.read(2).astype('float32')
            
            # Avoid division by zero
            ndvi = np.where((nir + red) == 0, 0, (nir - red) / (nir + red))
            
            # 4. Get Bounds for PostGIS
            b = src.bounds
            # Create Polygon: (Left, Bottom), (Right, Bottom), (Right, Top), (Left, Top), (Left, Bottom)
            poly = Polygon.from_bbox((b.left, b.bottom, b.right, b.top))
            
            # 5. Save Output to Memory Buffer
            out_meta = src.meta.copy()
            out_meta.update(dtype=rasterio.float32, count=1)
            
            out_mem = io.BytesIO()
            with rasterio.open(out_mem, 'w', **out_meta) as dst:
                dst.write(ndvi.astype(rasterio.float32), 1)
            
            out_mem.seek(0)
            
            # 6. Upload NDVI to MinIO
            out_path = drone_img.minio_path.replace("raw", "ndvi")
            # Ensure path is different if "raw" wasn't in the name
            if out_path == drone_img.minio_path:
                out_path = f"ndvi_{drone_img.minio_path}"

            client.put_object(
                bucket_name, 
                out_path, 
                out_mem, 
                out_mem.getbuffer().nbytes
            )
            
            # 7. Save to Database
            ProcessedRaster.objects.create(
                source=drone_img,
                minio_path=out_path,
                bounds=poly
            )
            
            # Mark original as processed
            drone_img.processed = True
            drone_img.save()
            print(f"SUCCESS: Processed NDVI for Image {image_id}")

    except Exception as e:
        print(f"ERROR processing NDVI: {str(e)}")