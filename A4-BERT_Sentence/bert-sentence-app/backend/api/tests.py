"""Basic smoke tests for the API app."""
from django.urls import reverse
from django.test import TestCase


class HealthCheckTests(TestCase):
    def test_health_endpoint(self) -> None:
        response = self.client.get(reverse("health-check"))
        self.assertEqual(response.status_code, 200)
        self.assertIn("status", response.json())
