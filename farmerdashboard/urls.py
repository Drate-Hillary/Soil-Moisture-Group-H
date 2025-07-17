from django.urls import path
from . import views

urlpatterns = [
    path('farmer_dashboard/', views.farmer_dashboard, name='farmer_dashboard'),
]