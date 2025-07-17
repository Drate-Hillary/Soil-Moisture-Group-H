from django.db import models
from django.utils import timezone

class SoilMoistureRecord(models.Model):
    record_id = models.IntegerField(unique=True)
    sensor_id = models.CharField(max_length=50)
    location = models.CharField(max_length=100)
    soil_moisture_percent = models.FloatField()
    temperature_celsius = models.FloatField()
    humidity_percent = models.FloatField()
    timestamp = models.DateTimeField()
    status = models.CharField(max_length=50)
    battery_voltage = models.FloatField()
    irrigation_action = models.CharField(max_length=50)

    def __str__(self):
        return f"{self.record_id} - {self.location} - {self.timestamp}"

    class Meta:
        db_table = 'soil_moisture_records'
        
        
        
class SoilMoisturePrediction(models.Model):
    location = models.CharField(max_length=100)
    predicted_moisture = models.FloatField()
    current_moisture = models.FloatField(default=0.0)
    temperature = models.FloatField(default=0.0)   
    humidity = models.FloatField(default=0.0)          
    precipitation = models.FloatField(default=0.0)
    prediction_for = models.DateTimeField(default=timezone.now)
    status = models.CharField(max_length=100, default="unknown")
    
    def __str__(self):
        return f"{self.location} - {self.predicted_moisture}%"