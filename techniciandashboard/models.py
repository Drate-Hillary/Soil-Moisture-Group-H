from django.db import models
from userAuthentication.models import CustomUser
from django.utils import timezone
from django.contrib.auth import get_user_model

# Create your models here.
class TechnicianLocationAssignment(models.Model):
    technician = models.ForeignKey(CustomUser, on_delete=models.CASCADE, limit_choices_to={'role': 'technician'})
    location = models.CharField(max_length=100)

    class Meta:
        db_table = 'technician_location_assignments'
        unique_together = ('technician', 'location')  # Ensure a technician can't be assigned to the same location twice

    def __str__(self):
        return f"{self.technician.get_full_name() or self.technician.username} - {self.location}"
    


class TechnicianSoilMoisturePrediction(models.Model):
    """
    Model to store soil moisture predictions made by technicians.
    """
    location = models.CharField(max_length=100, help_text="Location of the prediction")
    timestamp = models.DateTimeField(default=timezone.now, help_text="Time of prediction")
    current_moisture = models.FloatField(help_text="Current soil moisture percentage")
    temperature = models.FloatField(help_text="Temperature in Celsius")
    humidity = models.FloatField(help_text="Humidity percentage")
    precipitation = models.FloatField(default=0.0, help_text="Precipitation in mm")
    predicted_category = models.CharField(max_length=20, help_text="Predicted moisture category")
    predicted_moisture_value = models.FloatField(help_text="Predicted moisture value (%)")
    confidence = models.FloatField(null=True, blank=True, help_text="Prediction confidence score")
    created_at = models.DateTimeField(auto_now_add=True, help_text="Record creation time")
    updated_at = models.DateTimeField(auto_now=True, help_text="Record last updated time")

    class Meta:
        ordering = ['-timestamp']
        indexes = [
            models.Index(fields=['location', 'timestamp']),
        ]

    def __str__(self):
        return f"{self.location} - {self.timestamp.strftime('%Y-%m-%d %H:%M')} - {self.predicted_category}"