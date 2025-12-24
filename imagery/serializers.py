from rest_framework import serializers
from .models import DroneImage

class DroneImageSerializer(serializers.ModelSerializer):
    class Meta:
        model = DroneImage
        fields = ['id', 'farm', 'minio_path', 'processed', 'created_at']
        read_only_fields = ['processed', 'minio_path']