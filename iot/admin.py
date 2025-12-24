from django.contrib import admin
from .models import SensorReading, WeatherData
from import_export.admin import ImportExportModelAdmin
from import_export import resources


class SensorReadingResource(resources.ModelResource):
    class Meta:
        model = SensorReading
        fields = ('id', 'device', 'timestamp', 'temperature', 'moisture', 'ph', 'nutrients')


class WeatherDataResource(resources.ModelResource):
    class Meta:
        model = WeatherData
        fields = ('id', 'farm', 'timestamp', 'temperature', 'humidity', 'precipitation', 
                  'wind_speed', 'conditions', 'source')


@admin.register(SensorReading)
class SensorReadingAdmin(ImportExportModelAdmin):
    resource_class = SensorReadingResource
    list_display = ('device', 'timestamp', 'temperature', 'moisture')
    list_filter = ('device', 'timestamp')
    search_fields = ('device__name',)


@admin.register(WeatherData)
class WeatherDataAdmin(ImportExportModelAdmin):
    resource_class = WeatherDataResource
    list_display = ('farm', 'timestamp', 'temperature', 'humidity', 'precipitation', 'wind_speed', 'conditions')
    list_filter = ('farm', 'conditions', 'timestamp')
    search_fields = ('farm__name', 'conditions')