from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.core.mail import send_mail
from django.conf import settings
from django.contrib.auth.decorators import login_required
from urllib.parse import urlencode 
from django.urls import reverse
from django.http import HttpResponseRedirect
import logging
from .forms import CustomUserCreationForm
from techniciandashboard.models import TechnicianLocationAssignment
from django.contrib import messages
from .models import CustomUser

logger = logging.getLogger(__name__)

# Create your views here.
def home(request):
    return render(request, 'home.html')

def register(request):
    if request.method == 'POST':
        form = CustomUserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user, backend='django.contrib.auth.backends.ModelBackend')
            # Send email notification
            try:
                send_mail(
                    subject='Welcome to Farm System!',
                    message=f'Hi {user.username},\n\nYour account has been created successfully!',
                    from_email=settings.EMAIL_HOST_USER,
                    recipient_list=[user.email],
                    fail_silently=False,
                )
                logger.info(f"Welcome email sent to {user.email}")
            except Exception as e:
                logger.error(f"Failed to send email to {user.email}: {e}")
            if user.role == 'admin':
                return redirect('admin_dashboard')
            elif user.role == 'farmer':
                return redirect('farmer_dashboard')
            elif user.role == 'technician':
                return redirect('technician_dashboard')
            else:
                return redirect('home')
        else:
            logger.error(f"Form validation failed: {form.errors}")
    else:
     
        form = CustomUserCreationForm()
    return render(request, 'accounts/register.html', {'form': form})

def user_login(request):
    if request.method == 'POST':
        email = request.POST.get('email')
        password = request.POST.get('password')
        logger.debug(f"Attempting login with email: {email}")
        user = authenticate(request, email=email, password=password)
        if user is not None:
            logger.debug(f"User authenticated: {user.email}")
            login(request, user)
            if user.role == 'admin':
                return redirect('admin_dashboard')
            elif user.role == 'farmer':
                return redirect('farmer_dashboard')
            elif user.role == 'technician':
                # Get the technician's assigned farms
                assigned_farms = TechnicianLocationAssignment.objects.filter(technician=user)
                assigned_locations = assigned_farms.values_list('location', flat=True).distinct()
                if assigned_locations:
                    query_params = urlencode({'location': assigned_locations[0]})
                    base_url = reverse('technician_dashboard')
                    url_with_params = f'{base_url}?{query_params}'
                    return HttpResponseRedirect(url_with_params)
                else:
                    return redirect('technician_dashboard')
            else:
                return redirect('home')
        else:
            logger.error(f"Authentication failed for email: {email}")
            return render(request, 'accounts/login.html', {'error': 'Invalid credentials'})
    return render(request, 'accounts/login.html')


def user_logout(request):
    logout(request)
    return redirect('login')


@login_required
def edit_profile(request):
    if request.method == 'POST':
        user = request.user
        
        # Get or create profile
        profile = user.get_profile()  # Using our new method
        
        # Update user fields
        username = request.POST.get('username')
        email = request.POST.get('email')
        
        # Validation checks
        if CustomUser.objects.filter(username=username).exclude(pk=user.pk).exists():
            messages.error(request, 'Username already taken')
            return redirect('technician_dashboard')
            
        if CustomUser.objects.filter(email=email).exclude(pk=user.pk).exists():
            messages.error(request, 'Email already in use')
            return redirect('technician_dashboard')
            
        user.username = username
        user.email = email
        user.save()
        
        # Update profile fields
        profile.location = request.POST.get('location')
        
        # Handle image upload
        if 'image' in request.FILES:
            if profile.image:
                profile.image.delete()
            profile.image = request.FILES['image']
        
        # Handle image removal
        if 'remove_image' in request.POST and profile.image:
            profile.image.delete()
            profile.image = None
        
        profile.save()
        
        messages.success(request, 'Profile updated successfully!')
        return redirect('technician_dashboard')
    
    return redirect('technician_dashboard')