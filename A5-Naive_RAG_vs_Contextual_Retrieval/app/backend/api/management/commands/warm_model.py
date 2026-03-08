from django.core.management.base import BaseCommand

from api.services import get_text_generator


class Command(BaseCommand):
    help = "Preload the DPO model into process memory"

    def handle(self, *args, **options):
        _ = (args, options)
        get_text_generator()
        self.stdout.write(self.style.SUCCESS("Model loaded successfully."))
