from django.contrib import admin
from django.contrib.gis import admin as gis_admin
from import_export.admin import ImportExportModelAdmin
from import_export import resources
from .models import Farm, FieldBoundary, Device, Asset


# Resources for Import/Export
class FarmResource(resources.ModelResource):
    class Meta:
        model = Farm
        fields = ('id', 'name', 'owner__username', 'created_at')
        export_order = ('id', 'name', 'owner__username', 'created_at')


class FieldBoundaryResource(resources.ModelResource):
    class Meta:
        model = FieldBoundary
        fields = ('id', 'name', 'farm__name', 'crop_type', 'area_hectares', 'created_at')


class DeviceResource(resources.ModelResource):
    class Meta:
        model = Device
        fields = ('id', 'name', 'farm__name', 'device_type', 'status', 'thingsboard_id')


class AssetResource(resources.ModelResource):
    class Meta:
        model = Asset
        fields = ('id', 'name', 'asset_type', 'farm__name', 'serial_number')


@admin.register(Farm)
class FarmAdmin(ImportExportModelAdmin):
    resource_class = FarmResource
    list_display = ('name', 'owner', 'location', 'created_at')
    search_fields = ('name', 'owner__username')
    list_filter = ('created_at',)


@admin.register(FieldBoundary)
class FieldBoundaryAdmin(ImportExportModelAdmin, gis_admin.GISModelAdmin):
    resource_class = FieldBoundaryResource
    list_display = ('name', 'farm', 'crop_type', 'area_hectares', 'created_at')
    search_fields = ('name', 'farm__name')
    list_filter = ('crop_type', 'farm')


@admin.register(Device)
class DeviceAdmin(ImportExportModelAdmin):
    resource_class = DeviceResource
    list_display = ('name', 'farm', 'field', 'device_type', 'status', 'last_seen')
    search_fields = ('name', 'thingsboard_id')
    list_filter = ('device_type', 'status', 'farm')


@admin.register(Asset)
class AssetAdmin(ImportExportModelAdmin):
    resource_class = AssetResource
    list_display = ('name', 'asset_type', 'farm', 'field', 'device', 'serial_number')
    search_fields = ('name', 'serial_number')
    list_filter = ('asset_type', 'farm')