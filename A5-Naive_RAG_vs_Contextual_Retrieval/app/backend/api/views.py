from rest_framework.response import Response
from rest_framework.views import APIView

from .serializers import GenerateRequestSerializer
from .services import generate_text, get_report_payload


class HealthView(APIView):
    def get(self, request):
        _ = request
        return Response({"status": "ok"})


class ReportDataView(APIView):
    def get(self, request):
        _ = request
        return Response(get_report_payload())


class GenerateView(APIView):
    def post(self, request):
        serializer = GenerateRequestSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        response_text = generate_text(
            prompt=serializer.validated_data["prompt"],
            max_new_tokens=serializer.validated_data["max_new_tokens"],
        )
        return Response({"response": response_text})
