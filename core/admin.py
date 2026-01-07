from django.contrib import admin
from django.contrib.gis import admin as gis_admin
from django.contrib.gis.forms.widgets import OSMWidget
from django.contrib.gis.geos import GEOSGeometry
from django import forms
from django.conf import settings
from django.core.exceptions import ValidationError
from import_export.admin import ImportExportModelAdmin
from import_export import resources
from .models import Farm, FieldBoundary, Device, Asset


# Get configuration from Django settings (with fallbacks)
DEFAULT_LAT = getattr(settings, 'FARMAI_DEFAULT_LAT', 0)
DEFAULT_LON = getattr(settings, 'FARMAI_DEFAULT_LON', 0)
DEFAULT_ZOOM = getattr(settings, 'FARMAI_DEFAULT_ZOOM', 10)
MAX_FIELD_DISTANCE_KM = getattr(settings, 'FARMAI_MAX_FIELD_DISTANCE_KM', 50)
MIN_FIELD_AREA_HA = getattr(settings, 'FARMAI_MIN_FIELD_AREA_HA', 0.00)
MAX_FIELD_AREA_HA = getattr(settings, 'FARMAI_MAX_FIELD_AREA_HA', 10000)


# Custom map widget with configurable defaults and multiple layers
class FarmLocationWidget(OSMWidget):
    """Custom OpenStreetMap widget with configurable defaults"""
    template_name = 'admin/gis/farm_location_widget.html'
    
    # These will be set from Django settings
    default_lat = DEFAULT_LAT
    default_lon = DEFAULT_LON
    default_zoom = DEFAULT_ZOOM
    
    class Media:
        css = {
            'all': (
                'https://unpkg.com/leaflet@1.9.4/dist/leaflet.css',
            )
        }
        js = (
            'https://unpkg.com/leaflet@1.9.4/dist/leaflet.js',
        )
    
    def get_context(self, name, value, attrs):
        context = super().get_context(name, value, attrs)
        # Pass configurable settings to template
        context['default_lat'] = DEFAULT_LAT
        context['default_lon'] = DEFAULT_LON
        context['default_zoom'] = DEFAULT_ZOOM
        return context


# Custom widget for FieldBoundary with farm location awareness
class FieldBoundaryWidget(OSMWidget):
    """Custom widget for field boundary with multiple layers and farm proximity validation"""
    template_name = 'admin/gis/field_boundary_widget.html'
    
    # These will be set from Django settings
    default_lat = DEFAULT_LAT
    default_lon = DEFAULT_LON
    default_zoom = DEFAULT_ZOOM + 2  # Slightly more zoomed in for field drawing
    
    class Media:
        css = {
            'all': (
                'https://unpkg.com/leaflet@1.9.4/dist/leaflet.css',
                'https://unpkg.com/leaflet-draw@1.0.4/dist/leaflet.draw.css',
            )
        }
        js = (
            'https://unpkg.com/leaflet@1.9.4/dist/leaflet.js',
            'https://unpkg.com/leaflet-draw@1.0.4/dist/leaflet.draw.js',
        )
    
    def get_context(self, name, value, attrs):
        context = super().get_context(name, value, attrs)
        # Pass configurable settings to template
        context['default_lat'] = DEFAULT_LAT
        context['default_lon'] = DEFAULT_LON
        context['default_zoom'] = DEFAULT_ZOOM + 2
        context['max_distance_km'] = MAX_FIELD_DISTANCE_KM
        context['min_area_ha'] = MIN_FIELD_AREA_HA
        context['max_area_ha'] = MAX_FIELD_AREA_HA
        return context


# Custom form for Farm to use our widget
class FarmAdminForm(forms.ModelForm):
    class Meta:
        model = Farm
        fields = '__all__'
        widgets = {
            'location': FarmLocationWidget(attrs={
                'default_lat': DEFAULT_LAT,
                'default_lon': DEFAULT_LON,
                'default_zoom': DEFAULT_ZOOM,
            }),
        }


# Custom form for FieldBoundary with farm proximity validation
class FieldBoundaryAdminForm(forms.ModelForm):
    class Meta:
        model = FieldBoundary
        fields = '__all__'
        widgets = {
            'boundary': FieldBoundaryWidget(attrs={
                'default_lat': DEFAULT_LAT,
                'default_lon': DEFAULT_LON,
                'default_zoom': DEFAULT_ZOOM + 2,
            }),
        }
    
    def clean(self):
        cleaned_data = super().clean()
        farm = cleaned_data.get('farm')
        boundary = cleaned_data.get('boundary')
        
        if farm and boundary and farm.location:
            # Calculate distance from farm to field centroid
            try:
                centroid = boundary.centroid
                farm_lat = farm.location.y
                farm_lon = farm.location.x
                field_lat = centroid.y
                field_lon = centroid.x
                
                # Calculate distance in km (approximate using Haversine)
                import math
                R = 6371  # Earth's radius in km
                
                lat1_rad = math.radians(farm_lat)
                lat2_rad = math.radians(field_lat)
                delta_lat = math.radians(field_lat - farm_lat)
                delta_lon = math.radians(field_lon - farm_lon)
                
                a = math.sin(delta_lat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon/2)**2
                c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
                distance_km = R * c
                
                if distance_km > MAX_FIELD_DISTANCE_KM:
                    raise ValidationError({
                        'boundary': f'Field boundary is too far from the farm location. '
                                   f'Distance: {distance_km:.2f} km (Maximum allowed: {MAX_FIELD_DISTANCE_KM} km). '
                                   f'Please draw the field boundary closer to the farm.'
                    })
                
                # Validate area is reasonable
                area_ha = boundary.area / 10000.0
                if area_ha < MIN_FIELD_AREA_HA:
                    raise ValidationError({
                        'boundary': f'Field area is too small ({area_ha:.6f} ha). Minimum is {MIN_FIELD_AREA_HA} ha.'
                    })
                if area_ha > MAX_FIELD_AREA_HA:
                    raise ValidationError({
                        'boundary': f'Field area is too large ({area_ha:.2f} ha). Maximum is {MAX_FIELD_AREA_HA} ha.'
                    })
                    
            except ValidationError:
                raise
            except Exception as e:
                # If calculation fails, allow the boundary but log the error
                pass
        
        return cleaned_data


# Resources for Import/Export
class FarmResource(resources.ModelResource):
    class Meta:
        model = Farm
        fields = ('id', 'name', 'owner__username', 'created_at')
        export_order = ('id', 'name', 'owner__username', 'created_at')


class FieldBoundaryResource(resources.ModelResource):
    class Meta:
        model = FieldBoundary
        fields = ('id', 'name', 'farm__name', 'crop_type', 'area_hectares','boundary', 'created_at')


class DeviceResource(resources.ModelResource):
    class Meta:
        model = Device
        fields = ('id', 'name', 'farm__name', 'device_type', 'status', 'thingsboard_id')


class AssetResource(resources.ModelResource):
    class Meta:
        model = Asset
        fields = ('id', 'name', 'asset_type', 'farm__name', 'serial_number')


@admin.register(Farm)
class FarmAdmin(ImportExportModelAdmin, gis_admin.GISModelAdmin):
    form = FarmAdminForm
    resource_class = FarmResource
    list_display = ('name', 'owner', 'location', 'created_at')
    search_fields = ('name', 'owner__username')
    list_filter = ('created_at',)
    
    # GIS Map settings - center on Pakistan/LUMS area
    gis_widget_kwargs = {
        'attrs': {
            'default_lat': 31.4697,
            'default_lon': 74.4101,
            'default_zoom': 12,
        }
    }


@admin.register(FieldBoundary)
class FieldBoundaryAdmin(ImportExportModelAdmin, gis_admin.GISModelAdmin):
    form = FieldBoundaryAdminForm
    resource_class = FieldBoundaryResource
    list_display = ('name', 'farm', 'crop_type', 'area_hectares', 'created_at')
    search_fields = ('name', 'farm__name')
    list_filter = ('crop_type', 'farm')
    
    class Media:
        js = (
            'https://unpkg.com/leaflet@1.9.4/dist/leaflet.js',
            'https://unpkg.com/leaflet-draw@1.0.4/dist/leaflet.draw.js',
        )
        css = {
            'all': (
                'https://unpkg.com/leaflet@1.9.4/dist/leaflet.css',
                'https://unpkg.com/leaflet-draw@1.0.4/dist/leaflet.draw.css',
            )
        }


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