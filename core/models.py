from django.contrib.gis.db import models
from django.contrib.auth.models import User
from django.utils import timezone


class Farm(models.Model):
    owner = models.ForeignKey(User, on_delete=models.CASCADE)
    name = models.CharField(max_length=255)
    location = models.PointField(srid=4326)
    created_at = models.DateTimeField(default=timezone.now)

    def __str__(self):
        return self.name


class FieldBoundary(models.Model):
    farm = models.ForeignKey(Farm, on_delete=models.CASCADE, related_name='fields')
    name = models.CharField(max_length=100)
    crop_type = models.CharField(max_length=100)
    boundary = models.PolygonField(srid=4326, geography=True)
    area_hectares = models.FloatField(editable=False, null=True)
    created_at = models.DateTimeField(default=timezone.now)

    def save(self, *args, **kwargs):
        if self.boundary:
            try:
                # geography=True area is in square meters
                self.area_hectares = self.boundary.area / 10000.0
            except Exception:
                self.area_hectares = None
        super().save(*args, **kwargs)

    def __str__(self):
        return f"{self.farm.name} / {self.name}"


class Device(models.Model):
    STATUS_CHOICES = (
        ('active', 'Active'),
        ('inactive', 'Inactive'),
        ('unknown', 'Unknown'),
    )

    farm = models.ForeignKey(Farm, on_delete=models.CASCADE)
    field = models.ForeignKey(FieldBoundary, on_delete=models.SET_NULL, null=True, blank=True)
    name = models.CharField(max_length=100)
    thingsboard_id = models.UUIDField(help_text="Device ID from Thingsboard", unique=True)
    device_type = models.CharField(max_length=100, default='sensor')
    token = models.CharField(max_length=512, null=True, blank=True, help_text="Thingsboard device token (store securely in production)")
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='unknown')
    last_seen = models.DateTimeField(null=True, blank=True)
    metadata = models.JSONField(null=True, blank=True)
    created_at = models.DateTimeField(default=timezone.now)

    def __str__(self):
        return self.name


class Asset(models.Model):
    # For now assets represent IoT sensors / units
    farm = models.ForeignKey(Farm, on_delete=models.CASCADE, related_name='assets')
    field = models.ForeignKey(FieldBoundary, on_delete=models.SET_NULL, null=True, blank=True, related_name='assets')
    name = models.CharField(max_length=150)
    asset_type = models.CharField(max_length=100, default='iot_sensor')
    serial_number = models.CharField(max_length=200, null=True, blank=True)
    device = models.ForeignKey(Device, on_delete=models.SET_NULL, null=True, blank=True, related_name='assets')
    location = models.PointField(srid=4326, null=True, blank=True)
    metadata = models.JSONField(null=True, blank=True)
    created_at = models.DateTimeField(default=timezone.now)

    def __str__(self):
        return f"{self.name} ({self.asset_type})"