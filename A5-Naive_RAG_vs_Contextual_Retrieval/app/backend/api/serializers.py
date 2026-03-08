from rest_framework import serializers


class GenerateRequestSerializer(serializers.Serializer):
    prompt = serializers.CharField(max_length=4000)
    max_new_tokens = serializers.IntegerField(default=180, min_value=16, max_value=512)
