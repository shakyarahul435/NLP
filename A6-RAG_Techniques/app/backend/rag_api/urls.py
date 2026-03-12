from django.urls import path
from .views import ask, evaluation, health

urlpatterns = [
    path("health/", health),
    path("evaluation/", evaluation),
    path("ask/", ask),
]
