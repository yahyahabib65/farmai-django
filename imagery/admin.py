from django.contrib import admin
from .models import DroneImage, ProcessedRaster, SentinelImage, ImageReading

@admin.register(DroneImage)
class DroneImageAdmin(admin.ModelAdmin):
    list_display = ('farm', 'minio_path', 'processed')

@admin.register(ProcessedRaster)
class ProcessedRasterAdmin(admin.ModelAdmin):
    list_display = ('source', 'minio_path')


@admin.register(SentinelImage)
class SentinelImageAdmin(admin.ModelAdmin):
    list_display = ('farm', 'minio_path', 'processed', 'uploaded_at')
    list_filter = ('farm', 'processed')
    search_fields = ('minio_path',)


@admin.register(ImageReading)
class ImageReadingAdmin(admin.ModelAdmin):
    list_display = ('field', 'acquisition_date', 'ndvi_mean', 'ndwi_mean', 'health_status', 'water_status', 'processed_at')
    list_filter = ('farm', 'field', 'acquisition_date')
    search_fields = ('field__name', 'processing_notes')
    readonly_fields = ('processed_at', 'health_status', 'water_status')
    date_hierarchy = 'acquisition_date'
    
    fieldsets = (
        ('Location', {
            'fields': ('farm', 'field', 'acquisition_date')
        }),
        ('NDVI Metrics', {
            'fields': ('ndvi_mean', 'ndvi_std', 'ndvi_min', 'ndvi_max', 'ndvi_image_path')
        }),
        ('NDWI Metrics', {
            'fields': ('ndwi_mean', 'ndwi_std', 'ndwi_min', 'ndwi_max', 'ndwi_image_path')
        }),
        ('Band Data', {
            'fields': ('band_data',),
            'classes': ('collapse',)
        }),
        ('Quality Metrics', {
            'fields': ('cloud_coverage', 'valid_pixel_percentage', 'processing_notes')
        }),
        ('MinIO Storage', {
            'fields': ('minio_folder',)
        }),
        ('Metadata', {
            'fields': ('processed_at',)
        }),
    )
    
    def health_status(self, obj):
        return obj.health_status
    health_status.short_description = 'Health'
    
    def water_status(self, obj):
        return obj.water_status
    water_status.short_description = 'Water'