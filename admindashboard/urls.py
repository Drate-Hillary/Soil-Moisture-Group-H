from django.urls import path
from . import views
from .views import upload_csv, predict_moisture, upload_model, upload_model_soil_moisture 

urlpatterns = [
    path('', views.admin_dashboard, name='admin_dashboard'),
    path('upload_csv/', upload_csv, name='upload_csv'),
    path('predict_moisture/', predict_moisture, name='predict_moisture'),
    path('upload_model/', upload_model, name='upload_model'),
    path('upload_model_soil_moisture/', upload_model_soil_moisture, name='upload_model_soil_moisture'),
    path('generate_report/', views.generate_report, name='generate_report'),
    path('assign-technician/', views.assign_technician, name='assign_technician'),
    path('unassign-technician/', views.unassign_technician, name='unassign_technician'),
]