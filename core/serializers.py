from rest_framework import serializers
from .models import Farm, FieldBoundary, Device, Asset

class FarmSerializer(serializers.ModelSerializer):
    class Meta:
        model = Farm
        fields = '__all__'

class FieldBoundarySerializer(serializers.ModelSerializer):
    class Meta:
        model = FieldBoundary
        fields = '__all__'

class DeviceSerializer(serializers.ModelSerializer):
    class Meta:
        model = Device
        fields = '__all__'
        read_only_fields = ('created_at',)

class AssetSerializer(serializers.ModelSerializer):
    class Meta:
        model = Asset
        fields = '__all__'
        read_only_fields = ('created_at',)
