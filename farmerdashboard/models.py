from django.db import models
from django.contrib.auth import get_user_model
from userAuthentication.models import CustomUser
# Create your models here.
class Farm(models.Model):
    farmer = models.ForeignKey(CustomUser, on_delete=models.CASCADE, related_name='farms')
    location = models.CharField(max_length=100)
    size = models.DecimalField(max_digits=10, decimal_places=2)  # in acres
    description = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.farmer.username}'s farm at {self.location} ({self.size} acres)"