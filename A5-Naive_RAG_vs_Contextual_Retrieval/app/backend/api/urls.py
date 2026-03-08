from django.urls import path

from .views import GenerateView, HealthView, ReportDataView

urlpatterns = [
    path("health/", HealthView.as_view(), name="health"),
    path("report/", ReportDataView.as_view(), name="report"),
    path("generate/", GenerateView.as_view(), name="generate"),
]
