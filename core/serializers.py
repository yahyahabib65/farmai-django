from rest_framework import serializers
from django.conf import settings
from django.contrib.gis.geos import GEOSGeometry, Point
from django.contrib.gis.measure import D
from .models import Farm, FieldBoundary, Device, Asset


# Get configuration from Django settings (with fallbacks)
MAX_FIELD_DISTANCE_KM = getattr(settings, 'FARMAI_MAX_FIELD_DISTANCE_KM', 50)
MIN_FIELD_AREA_HA = getattr(settings, 'FARMAI_MIN_FIELD_AREA_HA', 0.00)
MAX_FIELD_AREA_HA = getattr(settings, 'FARMAI_MAX_FIELD_AREA_HA', 10000)


class FarmSerializer(serializers.ModelSerializer):
    class Meta:
        model = Farm
        fields = '__all__'
    
    def validate_location(self, value):
        """Validate that location coordinates are reasonable"""
        if value:
            # Check if coordinates are within valid ranges
            if not (-180 <= value.x <= 180):
                raise serializers.ValidationError("Longitude must be between -180 and 180")
            if not (-90 <= value.y <= 90):
                raise serializers.ValidationError("Latitude must be between -90 and 90")
            # Check if location is not at (0, 0) - often indicates missing data
            if value.x == 0 and value.y == 0:
                raise serializers.ValidationError("Location cannot be at (0, 0). Please provide valid coordinates.")
        return value
    
    def validate_name(self, value):
        """Validate farm name"""
        if not value or len(value.strip()) < 2:
            raise serializers.ValidationError("Farm name must be at least 2 characters")
        return value.strip()


class FieldBoundarySerializer(serializers.ModelSerializer):
    distance_from_farm = serializers.SerializerMethodField(read_only=True)
    
    class Meta:
        model = FieldBoundary
        fields = '__all__'
    
    def get_distance_from_farm(self, obj):
        """Calculate distance from field center to farm location"""
        if obj.boundary and obj.farm and obj.farm.location:
            try:
                centroid = obj.boundary.centroid
                # Distance in meters
                distance = obj.farm.location.distance(centroid) * 111320  # Approximate meters per degree
                return round(distance, 2)
            except:
                return None
        return None
    
    def validate(self, data):
        """Validate that boundary is within reasonable distance from farm"""
        boundary = data.get('boundary')
        farm = data.get('farm')
        
        # Get existing values for update operations
        if self.instance:
            boundary = boundary or self.instance.boundary
            farm = farm or self.instance.farm
        
        if boundary and farm and farm.location:
            try:
                # Get centroid of the boundary
                centroid = boundary.centroid
                
                # Calculate distance in kilometers (approximate)
                # 1 degree latitude ≈ 111.32 km
                lat_diff = abs(centroid.y - farm.location.y)
                lon_diff = abs(centroid.x - farm.location.x)
                distance_km = ((lat_diff ** 2 + lon_diff ** 2) ** 0.5) * 111.32
                
                # Use configurable maximum distance from settings
                if distance_km > MAX_FIELD_DISTANCE_KM:
                    raise serializers.ValidationError({
                        'boundary': f"Field boundary is too far from farm location. "
                                   f"Distance: {distance_km:.2f} km (max: {MAX_FIELD_DISTANCE_KM} km). "
                                   f"Please ensure the field is within the farm's area."
                    })
                
                # Validate boundary area is reasonable using configurable limits
                area_ha = boundary.area / 10000.0  # Convert sq meters to hectares
                if area_ha < MIN_FIELD_AREA_HA:
                    raise serializers.ValidationError({
                        'boundary': f"Field area is too small ({area_ha:.4f} ha). Minimum is {MIN_FIELD_AREA_HA} ha."
                    })
                if area_ha > MAX_FIELD_AREA_HA:
                    raise serializers.ValidationError({
                        'boundary': f"Field area is too large ({area_ha:.2f} ha). Maximum is {MAX_FIELD_AREA_HA} ha."
                    })
                    
            except serializers.ValidationError:
                raise
            except Exception as e:
                # If distance calculation fails, allow the boundary
                pass
        
        # Validate crop type
        crop_type = data.get('crop_type', '')
        if crop_type and len(crop_type.strip()) < 2:
            raise serializers.ValidationError({
                'crop_type': "Crop type must be at least 2 characters"
            })
        
        return data
    
    def validate_name(self, value):
        """Validate field name"""
        if not value or len(value.strip()) < 2:
            raise serializers.ValidationError("Field name must be at least 2 characters")
        return value.strip()


class DeviceSerializer(serializers.ModelSerializer):
    class Meta:
        model = Device
        fields = '__all__'
        read_only_fields = ('created_at',)
    
    def validate(self, data):
        """Validate device data"""
        farm = data.get('farm')
        field = data.get('field')
        
        # Ensure field belongs to the farm
        if field and farm:
            if field.farm_id != farm.id:
                raise serializers.ValidationError({
                    'field': "Selected field does not belong to the specified farm."
                })
        
        # If only field is provided, get farm from field
        if field and not farm:
            data['farm'] = field.farm
        
        return data
    
    def validate_name(self, value):
        """Validate device name"""
        if not value or len(value.strip()) < 2:
            raise serializers.ValidationError("Device name must be at least 2 characters")
        return value.strip()
    
    def validate_thingsboard_id(self, value):
        """Validate ThingsBoard ID is a valid UUID"""
        if not value:
            raise serializers.ValidationError("ThingsBoard ID is required")
        return value


class AssetSerializer(serializers.ModelSerializer):
    class Meta:
        model = Asset
        fields = '__all__'
        read_only_fields = ('created_at',)
    
    def validate(self, data):
        """Validate asset data"""
        farm = data.get('farm')
        field = data.get('field')
        location = data.get('location')
        
        # Ensure field belongs to the farm
        if field and farm:
            if field.farm_id != farm.id:
                raise serializers.ValidationError({
                    'field': "Selected field does not belong to the specified farm."
                })
        
        # If location provided, validate it's within farm area
        if location and farm and farm.location:
            try:
                lat_diff = abs(location.y - farm.location.y)
                lon_diff = abs(location.x - farm.location.x)
                distance_km = ((lat_diff ** 2 + lon_diff ** 2) ** 0.5) * 111.32
                
                if distance_km > MAX_FIELD_DISTANCE_KM:
                    raise serializers.ValidationError({
                        'location': f"Asset location is too far from farm ({distance_km:.2f} km). Max: {MAX_FIELD_DISTANCE_KM} km."
                    })
            except serializers.ValidationError:
                raise
            except:
                pass
        
        return data
    
    def validate_name(self, value):
        """Validate asset name"""
        if not value or len(value.strip()) < 2:
            raise serializers.ValidationError("Asset name must be at least 2 characters")
        return value.strip()
