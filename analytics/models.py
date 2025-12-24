from django.db import models

class AnalyticsResult(models.Model):
    """NDVI/NDWI analysis results per field"""
    # Link to a specific field using lazy reference
    field = models.ForeignKey('core.FieldBoundary', on_delete=models.CASCADE)
    
    # When did we run this check?
    date = models.DateField(auto_now_add=True)
    
    # The calculated health score (NDVI)
    avg_ndvi = models.FloatField(help_text="Average NDVI value (0-1)")
    
    # Water index
    avg_ndwi = models.FloatField(null=True, blank=True, help_text="Average NDWI value (-1 to 1)")
    
    # Did the system flag this field?
    irrigation_alert = models.BooleanField(default=False)
    
    # Extra notes (e.g., "Low moisture detected")
    notes = models.TextField(blank=True)

    def __str__(self):
        return f"{self.field.name} - {self.date}"


class HarvestPrediction(models.Model):
    """Harvest date prediction based on NDVI time series - FarmVibes workflow"""
    field = models.ForeignKey('core.FieldBoundary', on_delete=models.CASCADE, related_name='harvest_predictions')
    
    # Planting info
    planting_date = models.DateField(help_text="When the crop was planted")
    crop_type = models.CharField(max_length=100, blank=True)
    
    # Predicted dates
    predicted_emergence_date = models.DateField(null=True, blank=True)
    predicted_harvest_date = models.DateField(null=True, blank=True)
    
    # Days to harvest (calculated)
    days_to_harvest = models.IntegerField(null=True, blank=True, help_text="Days remaining until harvest")
    
    # Growth stage tracking
    current_growth_stage = models.CharField(max_length=50, default='unknown', 
        choices=[
            ('germination', 'Germination'),
            ('emergence', 'Emergence'),
            ('vegetative', 'Vegetative Growth'),
            ('flowering', 'Flowering'),
            ('fruiting', 'Fruiting/Grain Fill'),
            ('maturity', 'Maturity'),
            ('harvest_ready', 'Harvest Ready'),
        ])
    
    # Confidence
    confidence_score = models.FloatField(default=0.0, help_text="Prediction confidence 0-1")
    
    # Analysis date
    analyzed_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['-analyzed_at']

    def __str__(self):
        return f"{self.field.name} - Harvest: {self.predicted_harvest_date}"


class CarbonFootprint(models.Model):
    """Carbon footprint estimation - FarmVibes COMET-Farm workflow"""
    farm = models.ForeignKey('core.Farm', on_delete=models.CASCADE, related_name='carbon_estimates')
    
    # Time period
    year = models.IntegerField()
    season = models.CharField(max_length=20, choices=[
        ('spring', 'Spring'),
        ('summer', 'Summer'),
        ('fall', 'Fall'),
        ('winter', 'Winter'),
        ('annual', 'Annual'),
    ])
    
    # Carbon metrics (tonnes CO2 equivalent)
    total_emissions = models.FloatField(help_text="Total emissions in tonnes CO2e")
    soil_carbon_sequestration = models.FloatField(default=0, help_text="Carbon stored in soil")
    net_carbon = models.FloatField(help_text="Net carbon (negative = carbon sink)")
    
    # Breakdown
    fertilizer_emissions = models.FloatField(default=0)
    fuel_emissions = models.FloatField(default=0)
    livestock_emissions = models.FloatField(default=0)
    crop_residue_emissions = models.FloatField(default=0)
    
    # Practices tracked
    tillage_practice = models.CharField(max_length=50, blank=True, 
        choices=[('conventional', 'Conventional'), ('reduced', 'Reduced'), ('no_till', 'No-Till')])
    cover_crops = models.BooleanField(default=False)
    crop_rotation = models.BooleanField(default=False)
    
    # Sustainability score
    sustainability_score = models.FloatField(default=0, help_text="0-100 sustainability rating")
    
    analyzed_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-year', 'season']
        unique_together = ['farm', 'year', 'season']

    def __str__(self):
        return f"{self.farm.name} - {self.year} {self.season}: {self.net_carbon} tonnes CO2e"


class CropClassification(models.Model):
    """Crop segmentation/classification results - FarmVibes SpaceEye workflow"""
    field = models.ForeignKey('core.FieldBoundary', on_delete=models.CASCADE, related_name='crop_classifications')
    
    # Classification date
    date = models.DateField()
    
    # Detected crop
    detected_crop = models.CharField(max_length=100)
    confidence = models.FloatField(help_text="Classification confidence 0-1")
    
    # Area breakdown (percentage of field)
    healthy_area_percent = models.FloatField(default=0)
    stressed_area_percent = models.FloatField(default=0)
    bare_soil_percent = models.FloatField(default=0)
    water_area_percent = models.FloatField(default=0)
    
    # Image reference
    source_image = models.CharField(max_length=500, blank=True, help_text="Path to source imagery")
    result_image = models.CharField(max_length=500, blank=True, help_text="Path to classified result")
    
    analyzed_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-date']

    def __str__(self):
        return f"{self.field.name} - {self.detected_crop} ({self.confidence:.0%})"