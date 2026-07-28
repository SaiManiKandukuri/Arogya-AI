"""
App-level URL configuration for Chatbot endpoints.

Maps route '/' to index view and '/get/' to get_response AJAX view.
"""

from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),                     # Home chat page view
    path('get/', views.get_response, name='get_response'),       # AJAX RAG response API endpoint
    path('analyze_report/', views.analyze_report, name='analyze_report'), # AJAX Medical Report PDF analyzer endpoint
]

