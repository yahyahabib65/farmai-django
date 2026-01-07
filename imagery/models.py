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


class ImageReading(models.Model):
    """
    Stores processed satellite imagery readings per field.
    Links NDVI/NDWI calculations from MinIO bands to specific fields.
    """
    # Link to the specific field
    field = models.ForeignKey('core.FieldBoundary', on_delete=models.CASCADE, related_name='image_readings')
    
    # Link to the farm (redundant but useful for queries)
    farm = models.ForeignKey('core.Farm', on_delete=models.CASCADE, related_name='image_readings')
    
    # Acquisition timestamp (when the satellite captured the image)
    acquisition_date = models.DateField(help_text="Satellite image acquisition date")
    
    # MinIO path to the band folder (e.g., farm_1/2025-12-02/)
    minio_folder = models.CharField(max_length=500, help_text="MinIO folder path with band data")
    
    # NDVI statistics (calculated from real band data: (B08 - B04) / (B08 + B04))
    ndvi_mean = models.FloatField(null=True, blank=True, help_text="Mean NDVI value")
    ndvi_std = models.FloatField(null=True, blank=True, help_text="NDVI standard deviation")
    ndvi_min = models.FloatField(null=True, blank=True, help_text="Minimum NDVI value")
    ndvi_max = models.FloatField(null=True, blank=True, help_text="Maximum NDVI value")
    
    # NDWI statistics (calculated from: (B03 - B08) / (B03 + B08))
    ndwi_mean = models.FloatField(null=True, blank=True, help_text="Mean NDWI value")
    ndwi_std = models.FloatField(null=True, blank=True, help_text="NDWI standard deviation")
    ndwi_min = models.FloatField(null=True, blank=True, help_text="Minimum NDWI value")
    ndwi_max = models.FloatField(null=True, blank=True, help_text="Maximum NDWI value")
    
    # Band statistics (all 13 Sentinel-2 bands)
    band_data = models.JSONField(
        null=True, 
        blank=True, 
        help_text="JSON with statistics for all bands: {B01: {mean, std, min, max}, B02: {...}, ...}"
    )
    
    # MinIO paths to generated visualization images
    ndvi_image_path = models.CharField(max_length=500, null=True, blank=True, help_text="MinIO path to NDVI heatmap PNG")
    ndwi_image_path = models.CharField(max_length=500, null=True, blank=True, help_text="MinIO path to NDWI heatmap PNG")
    
    # Quality metrics
    cloud_coverage = models.FloatField(null=True, blank=True, help_text="Cloud coverage percentage (0-100)")
    valid_pixel_percentage = models.FloatField(null=True, blank=True, help_text="Percentage of valid (non-nodata) pixels")
    
    # Processing metadata
    processed_at = models.DateTimeField(auto_now_add=True)
    processing_notes = models.TextField(blank=True, help_text="Any notes about processing or data quality")
    
    class Meta:
        ordering = ['-acquisition_date']
        unique_together = ['field', 'acquisition_date']  # One reading per field per date
        verbose_name = "Image Reading"
        verbose_name_plural = "Image Readings"

    def __str__(self):
        return f"{self.field.name} - {self.acquisition_date} (NDVI: {self.ndvi_mean:.3f})" if self.ndvi_mean else f"{self.field.name} - {self.acquisition_date}"
    
    @property
    def health_status(self):
        """Determine field health based on NDVI"""
        if self.ndvi_mean is None:
            return 'unknown'
        if self.ndvi_mean >= 0.6:
            return 'healthy'
        elif self.ndvi_mean >= 0.4:
            return 'moderate'
        elif self.ndvi_mean >= 0.2:
            return 'stressed'
        else:
            return 'critical'
    
    @property
    def water_status(self):
        """Determine water stress based on NDWI"""
        if self.ndwi_mean is None:
            return 'unknown'
        if self.ndwi_mean >= 0.3:
            return 'adequate'
        elif self.ndwi_mean >= 0.0:
            return 'moderate'
        else:
            return 'stressed'