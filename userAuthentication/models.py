from django.contrib.auth.models import AbstractUser
from django.db import models
from django.conf import settings
from django.db.models.signals import post_save
from django.dispatch import receiver

class CustomUser(AbstractUser):
    ROLE_CHOICES = (
        ('admin', 'Admin'),
        ('technician', 'Technician'),
        ('farmer', 'Farmer'),
    )
    role = models.CharField(max_length=20, choices=ROLE_CHOICES, default='farmer')

    def __str__(self):
        return self.email

    def get_profile(self):
        profile, created = Profile.objects.get_or_create(user=self)
        return profile

class Profile(models.Model):
    user = models.OneToOneField(
        settings.AUTH_USER_MODEL, 
        on_delete=models.CASCADE, 
        related_name='profile'
    )
    image = models.ImageField(upload_to='profile_images/', null=True, blank=True)
    location = models.CharField(max_length=100, blank=True, null=True)
    
    def __str__(self):
        return f"{self.user.username}'s Profile"

@receiver(post_save, sender=settings.AUTH_USER_MODEL)
def create_or_update_user_profile(sender, instance, created, **kwargs):
    if created:
        Profile.objects.create(user=instance)
    else:
        # Ensure existing users have a profile
        if not hasattr(instance, 'profile'):
            Profile.objects.create(user=instance)