from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.core.mail import send_mail
from django.conf import settings
from userAuthentication.forms import CustomUserCreationForm
from userAuthentication.decorators import role_required, roles_required
from django.contrib.auth.decorators import login_required
import logging
import csv
import io
from django.contrib import messages
from admindashboard.models import SoilMoistureRecord, SoilMoisturePrediction
from datetime import datetime
from admindashboard.views import get_weather_forecast
from ml_model.ml_model import predict_soil_status, train_model, get_model_metrics, get_irrigation_schedule_recommendation
import os
import requests
from dotenv import load_dotenv
from django.db.models.functions import TruncDay
import json
from django.http import HttpResponse
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from io import BytesIO
from openpyxl import Workbook
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from io import BytesIO
from openpyxl import Workbook
from django.db.models import Avg, Min, Max
from datetime import timedelta
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from urllib.parse import urlencode 
from django.urls import reverse
from django.http import HttpResponseRedirect
from .models import Farm


load_dotenv()
weather_api = os.getenv("OPENWEATHER_API_KEY")

logger = logging.getLogger(__name__)


@login_required
def farmer_dashboard(request):
    # Initialize context
    context = {
        'user': request.user,
        'locations': [],
        'selected_location': '',
        'chart_data': '{}',
        'current_weather': None,
        'weather_location': None,
        'farms': Farm.objects.filter(farmer=request.user),
    }

    # Get unique locations for dropdown
    locations = SoilMoistureRecord.objects.values_list('location', flat=True).distinct()
    context['locations'] = locations

    # Default location
    default_location = locations[0] if locations else "Nairobi"
    selected_location = request.GET.get('location', default_location) if request.method == 'GET' else request.POST.get('location', default_location)
    context['selected_location'] = selected_location

    # Get weather data
    if selected_location:
        context['current_weather'] = get_weather_forecast(selected_location)
        context['weather_location'] = selected_location

    # Handle POST request for soil status prediction or farm registration
    if request.method == 'POST':
        if 'farm_name' in request.POST:
            # Farm registration logic (unchanged from original)
            try:
                farm_name = request.POST.get('farm_name')
                farm_size = request.POST.get('farm_size')
                other_farm_name = request.POST.get('other_farm_name')
                farm_description = request.POST.get('farm_description', '')

                if not farm_size or float(farm_size) <= 0:
                    messages.error(request, "Farm size must be a positive number.")
                    return redirect('farmer_dashboard')

                location = other_farm_name if farm_name == 'other' else farm_name
                if not location:
                    messages.error(request, "Please select or specify a valid farm location.")
                    return redirect('farmer_dashboard')

                Farm.objects.create(
                    farmer=request.user,
                    location=location,
                    size=float(farm_size),
                    description=farm_description
                )
                messages.success(request, f"Farm at {location} successfully registered!")
                return redirect('farmer_dashboard')
            except ValueError as e:
                messages.error(request, f"Invalid input: {str(e)}")
                logger.error(f"Farm registration error for farmer {request.user.email}: {str(e)}")
                return redirect('farmer_dashboard')
            except Exception as e:
                messages.error(request, f"Error registering farm: {str(e)}")
                logger.error(f"Farm registration error for farmer {request.user.email}: {str(e)}")
                return redirect('farmer_dashboard')

        # Soil status prediction logic
        try:
            soil_moisture = float(request.POST.get('soil-moisture'))
            temperature = float(request.POST.get('temperature'))
            humidity = float(request.POST.get('humidity'))

            # Query soil moisture records for selected location
            records = SoilMoistureRecord.objects.filter(location=selected_location).order_by('-timestamp')

            # Calculate daily averages for chart
            daily_averages = (
                records.annotate(date=TruncDay('timestamp'))
                .values('date')
                .annotate(
                    avg_moisture=Avg('soil_moisture_percent'),
                    avg_temperature=Avg('temperature_celsius'),
                    avg_humidity=Avg('humidity_percent')
                )
                .order_by('date')
            )

            chart_data = {
                'labels': [record['date'].strftime('%Y-%m-%d') for record in daily_averages],
                'datasets': [
                    {
                        'label': 'Soil Moisture (%)',
                        'data': [round(record['avg_moisture'], 2) for record in daily_averages],
                        'borderColor': '#1E90FF',
                        'backgroundColor': 'rgba(30, 144, 255, 0.2)',
                        'yAxisID': 'y1',
                        'fill': False
                    },
                    {
                        'label': 'Temperature (°C)',
                        'data': [round(record['avg_temperature'], 2) for record in daily_averages],
                        'borderColor': '#FF4500',
                        'backgroundColor': 'rgba(255, 69, 0, 0.2)',
                        'yAxisID': 'y2',
                        'fill': False
                    },
                    {
                        'label': 'Humidity (%)',
                        'data': [round(record['avg_humidity'], 2) for record in daily_averages],
                        'borderColor': '#32CD32',
                        'backgroundColor': 'rgba(50, 205, 50, 0.2)',
                        'yAxisID': 'y1',
                        'fill': False
                    }
                ]
            }

            # Make prediction using the ML model
            prediction_result = predict_soil_status(
                soil_moisture=soil_moisture,
                temperature=temperature,
                humidity=humidity
            )

            # Get irrigation schedule recommendation
            irrigation_schedule = get_irrigation_schedule_recommendation(
                soil_status=prediction_result['status']
            )

            # Store prediction in the database
            SoilMoisturePrediction.objects.create(
                location=selected_location,
                predicted_moisture=soil_moisture,
                current_moisture=soil_moisture,
                temperature=temperature,
                humidity=humidity,
                precipitation=0,
                prediction_for=datetime.now() + timedelta(hours=24),
                status=prediction_result['status']
            )

            context.update({
                'prediction': {
                    'status': prediction_result['status'],
                    'irrigation_recommendation': prediction_result['irrigation_recommendation'],
                    'confidence': prediction_result['confidence'],
                    'method': prediction_result['method'],
                    'input_values': prediction_result['input_values'],
                    'schedule': irrigation_schedule
                },
                'chart_data': json.dumps(chart_data),
            })

            return render(request, 'farmer_dashboard.html', context)

        except ValueError as e:
            messages.error(request, f"Invalid input: {str(e)}")
            logger.error(f"Input validation error for farmer {request.user.email}: {str(e)}")
            return redirect('farmer_dashboard')
        except Exception as e:
            messages.error(request, f"Error processing prediction: {str(e)}")
            logger.error(f"Prediction error for farmer {request.user.email}: {str(e)}")
            return redirect('farmer_dashboard')

    # GET request: Render dashboard with chart data
    records = SoilMoistureRecord.objects.filter(location=selected_location).order_by('-timestamp')

    # Calculate daily averages for chart
    daily_averages = (
        records.annotate(date=TruncDay('timestamp'))
        .values('date')
        .annotate(
            avg_moisture=Avg('soil_moisture_percent'),
            avg_temperature=Avg('temperature_celsius'),
            avg_humidity=Avg('humidity_percent')
        )
        .order_by('date')
    )

    chart_data = {
        'labels': [record['date'].strftime('%Y-%m-%d') for record in daily_averages],
        'datasets': [
            {
                'label': 'Soil Moisture (%)',
                'data': [round(record['avg_moisture'], 2) for record in daily_averages],
                'borderColor': '#1E90FF',
                'backgroundColor': 'rgba(30, 144, 255, 0.2)',
                'yAxisID': 'y1',
                'fill': False
            },
            {
                'label': 'Temperature (°C)',
                'data': [round(record['avg_temperature'], 2) for record in daily_averages],
                'borderColor': '#FF4500',
                'backgroundColor': 'rgba(255, 69, 0, 0.2)',
                'yAxisID': 'y2',
                'fill': False
            },
            {
                'label': 'Humidity (%)',
                'data': [round(record['avg_humidity'], 2) for record in daily_averages],
                'borderColor': '#32CD32',
                'backgroundColor': 'rgba(50, 205, 50, 0.2)',
                'yAxisID': 'y1',
                'fill': False
            }
        ]
    }

    context.update({
        'chart_data': json.dumps(chart_data),
    })

    return render(request, 'farmer_dashboard.html', context)

@login_required
@role_required('farmer')
def delete_farm(request, farm_id):
    try:
        farm = Farm.objects.get(id=farm_id)
        
        if farm.farmer != request.user:
            messages.error(request, "You don't have permission to delete this farm.")
            return redirect('farmer_dashboard')
        
        location = farm.location
        farm.delete()
        messages.success(request, f"Farm at {location} has been successfully deleted.")
        
    except Farm.DoesNotExist:
        messages.error(request, "The farm you tried to delete doesn't exist.")
    except Exception as e:
        messages.error(request, f"Error deleting farm: {str(e)}")
    
    return redirect('farmer_dashboard')