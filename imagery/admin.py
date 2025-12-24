from django.contrib import admin
from .models import DroneImage, ProcessedRaster

@admin.register(DroneImage)
class DroneImageAdmin(admin.ModelAdmin):
    list_display = ('farm', 'minio_path', 'processed')

@admin.register(ProcessedRaster)
class ProcessedRasterAdmin(admin.ModelAdmin):
    list_display = ('source', 'minio_path')