"""
URL configuration for visualizer app.
"""
from django.urls import path
from . import views

app_name = 'visualizer'

urlpatterns = [
    path('', views.home, name='home'),
    path('plotting/', views.plotting, name='plotting'),
    path('plotting-3d/', views.plotting_3d, name='plotting_3d'),
    path('distributions/', views.distributions, name='distributions'),
    path('distributions/<slug:slug>/', views.distribution_detail, name='distribution_detail'),
    path('calculations/', views.calculations, name='calculations'),
    path('about/', views.about, name='about'),
    # API endpoints for AJAX
    path('api/generate-plot/', views.api_generate_plot, name='api_generate_plot'),
    path('api/generate-3d-plot/', views.api_generate_3d_plot, name='api_generate_3d_plot'),
    path('api/calculate-stats/', views.api_calculate_stats, name='api_calculate_stats'),
    path('api/export-data/', views.api_export_data, name='api_export_data'),
]
