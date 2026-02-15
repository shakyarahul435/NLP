"""Application URL routes."""
from django.urls import path

from . import views

urlpatterns = [
    path("health/", views.health_check, name="health-check"),
    path("similarity/", views.sentence_similarity, name="sentence-similarity"),
    path("mask/", views.masked_prediction, name="masked-prediction"),
]
