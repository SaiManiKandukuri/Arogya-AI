"""
WSGI config for medicalbot_project.

Exposes the WSGI callable as a module-level variable named `application`.
Used by Gunicorn web server for deployment.
"""

import os
from django.core.wsgi import get_wsgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'medicalbot_project.settings')

application = get_wsgi_application()
