"""
URL Configuration for medicalbot_project.

Delegates root routes ('/') to the chatbot app routing module.
"""

from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('chatbot.urls')),  # Map root routes to chatbot app
]
