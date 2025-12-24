from django.contrib.gis.db import models
# We use the lazy reference 'core.Farm' to avoid import errors


class DroneImage(models.Model):
    # Link to the farm
    farm = models.ForeignKey('core.Farm', on_delete=models.CASCADE)
    
    # Where the file is stored in MinIO
    minio_path = models.CharField(max_length=500) 
    
    # Has it been processed yet?
    processed = models.BooleanField(default=False)
    
    # When was it uploaded
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Image for {self.farm} - {self.minio_path}"

class ProcessedRaster(models.Model):
    # Link to the original image
    source = models.OneToOneField(DroneImage, on_delete=models.CASCADE)
    
    # Path to the result (NDVI)
    minio_path = models.CharField(max_length=500)
    
    # The geographic area this raster covers (for map lookups)
    bounds = models.PolygonField(srid=4326) 
    
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"NDVI Raster - {self.source.id}"

class SentinelImage(models.Model):
    # Link to the farm
    farm = models.ForeignKey('core.Farm', on_delete=models.CASCADE)

    # Path to the Sentinel image in MinIO
    minio_path = models.CharField(max_length=500)

    # Has it been processed yet?
    processed = models.BooleanField(default=False)

    # Metadata fields (optional, can be extended)
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Sentinel Image for {self.farm} - {self.minio_path}"