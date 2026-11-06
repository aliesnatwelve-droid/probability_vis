"""
ASGI config for config project.

This is used instead of manage.py runserver to avoid bytecode cache issues.
"""
import os
import sys
import django

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')

from django.core.asgi import get_asgi_application
application = get_asgi_application()
