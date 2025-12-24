from django.db import models
# Note: We do NOT import Device here anymore to avoid the error.

class SensorReading(models.Model):
    # We use quotes 'core.Device' to tell Django: 
    # "Look for a model named Device in the core app later."
    device = models.ForeignKey('core.Device', on_delete=models.CASCADE, related_name='readings')
    
    timestamp = models.DateTimeField()
    results = models.JSONField(default=dict, help_text="Flexible key-value store for sensor data (temperature, moisture, pH, etc.)")
    
    class Meta:
        get_latest_by = 'timestamp'

    def __str__(self):
        # We can still access self.device.name normally here
        return f"{self.device.name} - {self.timestamp}"
    
    # Helper properties to access common keys
    @property
    def temperature(self):
        return self.results.get('temperature')
    
    @property
    def moisture(self):
        return self.results.get('moisture') or self.results.get('soil_moisture')
    
    @property
    def humidity(self):
        return self.results.get('humidity')


class WeatherData(models.Model):
    """Weather data for farms - supports FarmVibes weather integration"""
    farm = models.ForeignKey('core.Farm', on_delete=models.CASCADE, related_name='weather_data')
    timestamp = models.DateTimeField()
    
    # Temperature
    temperature = models.FloatField(null=True, blank=True, help_text="Temperature in Celsius")
    feels_like = models.FloatField(null=True, blank=True)
    
    # Humidity & Precipitation
    humidity = models.FloatField(null=True, blank=True, help_text="Humidity percentage")
    precipitation = models.FloatField(null=True, blank=True, help_text="Precipitation in mm")
    precipitation_probability = models.FloatField(null=True, blank=True)
    
    # Wind
    wind_speed = models.FloatField(null=True, blank=True, help_text="Wind speed in km/h")
    wind_direction = models.CharField(max_length=10, blank=True)
    
    # Solar
    solar_radiation = models.FloatField(null=True, blank=True, help_text="Solar radiation W/m²")
    uv_index = models.FloatField(null=True, blank=True)
    
    # Conditions
    conditions = models.CharField(max_length=100, blank=True, help_text="e.g., Sunny, Cloudy, Rainy")
    
    # Data source
    source = models.CharField(max_length=50, default='manual', help_text="e.g., NOAA, OpenWeather, manual")

    class Meta:
        ordering = ['-timestamp']
        get_latest_by = 'timestamp'

    def __str__(self):
        return f"{self.farm.name} - {self.timestamp.strftime('%Y-%m-%d %H:%M')}"