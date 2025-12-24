from django.contrib import admin 
from .models import AnalyticsResult, HarvestPrediction, CarbonFootprint, CropClassification
from import_export.admin import ImportExportModelAdmin
from import_export import resources


class AnalyticsResultResource(resources.ModelResource):
    class Meta:
        model = AnalyticsResult
        fields = ('id', 'field', 'date', 'avg_ndvi', 'avg_ndwi', 'irrigation_alert')


class HarvestPredictionResource(resources.ModelResource):
    class Meta:
        model = HarvestPrediction
        fields = ('id', 'field', 'planting_date', 'predicted_emergence_date', 'predicted_harvest_date', 
                  'current_growth_stage', 'confidence_score')


class CarbonFootprintResource(resources.ModelResource):
    class Meta:
        model = CarbonFootprint
        fields = ('id', 'farm', 'year', 'season', 'total_emissions', 'soil_carbon_sequestration', 
                  'net_carbon', 'sustainability_score')


class CropClassificationResource(resources.ModelResource):
    class Meta:
        model = CropClassification
        fields = ('id', 'field', 'date', 'detected_crop', 'confidence', 
                  'healthy_area_percent', 'stressed_area_percent')


@admin.register(AnalyticsResult)
class AnalyticsResultAdmin(ImportExportModelAdmin):
    resource_class = AnalyticsResultResource
    list_display = ('field', 'date', 'avg_ndvi', 'avg_ndwi', 'irrigation_alert')
    list_filter = ('irrigation_alert', 'date')
    search_fields = ('field__name',)


@admin.register(HarvestPrediction)
class HarvestPredictionAdmin(ImportExportModelAdmin):
    resource_class = HarvestPredictionResource
    list_display = ('field', 'planting_date', 'predicted_harvest_date', 'current_growth_stage', 'confidence_score')
    list_filter = ('current_growth_stage', 'planting_date')
    search_fields = ('field__name',)


@admin.register(CarbonFootprint)
class CarbonFootprintAdmin(ImportExportModelAdmin):
    resource_class = CarbonFootprintResource
    list_display = ('farm', 'year', 'season', 'total_emissions', 'net_carbon', 'sustainability_score')
    list_filter = ('year', 'season', 'tillage_practice', 'cover_crops')
    search_fields = ('farm__name',)


@admin.register(CropClassification)
class CropClassificationAdmin(ImportExportModelAdmin):
    resource_class = CropClassificationResource
    list_display = ('field', 'date', 'detected_crop', 'confidence', 'healthy_area_percent', 'stressed_area_percent')
    list_filter = ('detected_crop', 'date')
    search_fields = ('field__name', 'detected_crop')