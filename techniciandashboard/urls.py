from django.urls import path
from . import views
from admindashboard.views import upload_csv, predict_moisture
from techniciandashboard.views import technician_predict_moisture_view, generate_report, send_farmer_notification  
from userAuthentication.views import edit_profile

urlpatterns = [
    path('technician_dashboard/', views.technician_dashboard, name='technician_dashboard'),
    path('upload_csv/', upload_csv, name='upload_csv'),
    path('predict_moisture/', predict_moisture, name='predict_moisture'),
    path('technician/predict/', technician_predict_moisture_view, name='technician_predict_moisture_view'),
    path('generate_report/', generate_report, name='generate_report'),
    path('edit_profile/', edit_profile, name='edit_profile'),
    path('send_farmer_notification/', send_farmer_notification, name='send_farmer_notification'),
]

