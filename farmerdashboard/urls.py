from django.urls import path
from . import views
from .views import delete_farm

urlpatterns = [
    path('farmer_dashboard/', views.farmer_dashboard, name='farmer_dashboard'),
    path('delete-farm/<int:farm_id>/', delete_farm, name='delete_farm'),
]