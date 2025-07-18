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
from userAuthentication.models import CustomUser
from urllib.parse import urlencode 
from django.urls import reverse
from django.http import HttpResponseRedirect
from admindashboard.views import get_weather_forecast
import urllib
from techniciandashboard.models import TechnicianLocationAssignment

load_dotenv()
weather_api = os.getenv("OPENWEATHER_API_KEY")

logger = logging.getLogger(__name__)

@login_required
@role_required('technician')
def technician_dashboard(request):
    # Get the logged-in technician
    technician = request.user

    # Get assigned locations
    assigned_locations = list(TechnicianLocationAssignment.objects.filter(
        technician=technician
    ).values_list('location', flat=True).distinct())

    if not assigned_locations:
        messages.warning(request, "You are not assigned to any locations. Please contact your administrator.")
        return render(request, 'technician_dashboard.html', {
            'user': request.user,
            'records': [],
            'recent_records': [],
            'locations': [],
            'chart_data': json.dumps({'labels': [], 'data': []}),
            'average_moisture': None,
            'average_temperature': None,
            'average_humidity': None,
            'active_sensors': 0,
            'total_sensors': 0,
            'current_weather': None,
            'weather_location': None,
            'total_records': 0,
            'last_update': None,
            'last_upload': None,
        })

    # Get filter parameters
    location = request.GET.get('location', '')
    start_date = request.GET.get('start_date', '')
    end_date = request.GET.get('end_date', '')
    show_all = request.GET.get('show_all', 'false').lower() == 'true'

    # If no location is provided, default to the first assigned location
    if not location and assigned_locations:
        location = assigned_locations[0]

    # Query soil moisture records for assigned locations
    records = SoilMoistureRecord.objects.filter(location__in=assigned_locations).order_by('-timestamp')

    # Apply filters
    if location and location in assigned_locations:
        records = records.filter(location=location)
    
    if start_date:
        try:
            start_date_obj = datetime.strptime(start_date, '%Y-%m-%d')
            records = records.filter(timestamp__gte=start_date_obj)
        except ValueError:
            messages.error(request, 'Invalid start date format.')
    
    if end_date:
        try:
            end_date_obj = datetime.strptime(end_date, '%Y-%m-%d')
            records = records.filter(timestamp__lte=end_date_obj)
        except ValueError:
            messages.error(request, 'Invalid end date format.')

    # Get recent records for data ingestion section (last 10 records)
    recent_records = records[:10]

    # Get unique locations for dropdown (only assigned locations)
    locations = assigned_locations

    # Set default location for weather and other data
    default_location = location if location in assigned_locations else locations[0] if locations else None
    
    # Get weather data for default location
    current_weather = None
    if default_location:
        current_weather = get_weather_forecast(default_location)

    # Calculate average soil moisture
    average_moisture = records.aggregate(Avg('soil_moisture_percent'))['soil_moisture_percent__avg']
    average_moisture = round(average_moisture, 2) if average_moisture is not None else None

    # Calculate average temperature
    average_temperature = records.aggregate(Avg('temperature_celsius'))['temperature_celsius__avg']
    average_temperature = round(average_temperature, 2) if average_temperature is not None else None

    # Calculate average humidity
    average_humidity = records.aggregate(Avg('humidity_percent'))['humidity_percent__avg']
    average_humidity = round(average_humidity, 2) if average_humidity is not None else None

    # Calculate sensor statistics
    total_sensors = records.values('sensor_id').distinct().count()
    active_sensors = records.filter(status='active').values('sensor_id').distinct().count()

    # Get database metrics
    total_records = records.count()
    last_update = records.first().timestamp if records.exists() else None

    # Calculate daily averages for moisture trends chart
    daily_averages = (
        records.annotate(date=TruncDay('timestamp'))
        .values('date')
        .annotate(avg_moisture=Avg('soil_moisture_percent'))
        .order_by('date')
    )
    
    chart_data = {
        'labels': [record['date'].strftime('%Y-%m-%d') for record in daily_averages],
        'data': [round(record['avg_moisture'], 2) for record in daily_averages],
    }

    # Get prediction data from session if available
    prediction_data = request.session.get('prediction_data', {})
    if prediction_data:
        # Clear prediction data from session after using it
        del request.session['prediction_data']

    # Limit records to 10 unless show_all is true
    displayed_records = records[:10] if not show_all else records

    context = {
        'user': request.user,
        'records': displayed_records,
        'total_records': records.count(),
        'show_all': show_all,
        'recent_records': recent_records,
        'locations': locations,
        'selected_location': location,
        'start_date': start_date,
        'end_date': end_date,
        'average_moisture': average_moisture,
        'average_temperature': average_temperature,
        'average_humidity': average_humidity,
        'active_sensors': active_sensors,
        'total_sensors': total_sensors,
        'total_records': total_records,
        'last_update': last_update,
        'last_upload': 'N/A',
        'chart_data': json.dumps(chart_data),
        'current_weather': current_weather,
        'weather_location': default_location,
        'prediction': prediction_data.get('prediction'),
        'current_moisture': prediction_data.get('current_moisture'),
        'temperature': prediction_data.get('temperature'),
        'humidity': prediction_data.get('humidity'),
    }
    return render(request, 'technician_dashboard.html', context)


# Weather forecat api
def get_weather_forecast(location="Kampala"):
    """
    Fetch weather forecast for Kampala using OpenWeatherMap API.
    """
    if not weather_api:
        logger.error("OpenWeatherMap API key is not set.")
        return None

    city = "Kampala"  # Always use Kampala for weather forecast

    try:
        # OpenWeatherMap API endpoint for current weather
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={weather_api}&units=metric"
        request = urllib.request.Request(url)
        with urllib.request.urlopen(request, timeout=5) as response:
            if response.getcode() == 200:
                data = json.loads(response.read().decode('utf-8'))
            else:
                logger.error(f"API request failed with status code: {response.getcode()}")
                return None

        # Extract relevant weather information
        weather_data = {
            'temperature': data['main']['temp'],
            'humidity': data['main']['humidity'],
            'description': data['weather'][0]['description'],
            'icon': data['weather'][0]['icon'],
            'wind_speed': data['wind']['speed'],
            'precipitation': data.get('rain', {}).get('1h', 0)  # Rain volume for last hour (mm)
        }
        
        return weather_data
    
    except urllib.error.URLError as e:
        logger.error(f"Failed to fetch weather for {city}: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error fetching weather for {city}: {str(e)}")
        return None


from techniciandashboard.models import TechnicianSoilMoisturePrediction

def get_locations(technician=None):
    """
    Retrieve distinct locations - for technicians, only their assigned locations.
    """
    try:
        if technician and technician.role == 'technician':
            # For technicians, only return their assigned locations
            assigned_locations = TechnicianLocationAssignment.objects.filter(
                technician=technician
            ).values_list('location', flat=True).distinct()
            return sorted(assigned_locations) if assigned_locations else []

    except Exception as e:
        logger.error(f"Error fetching locations: {str(e)}")
        return ['Kampala'] if not technician or technician.role != 'technician' else []
                                            
                                                                                      
                                            
from ml_model.soil_moisture_prediction import load_model, train_model_with_db_data, predict_future_moisture
from admindashboard.models import SoilMoistureRecord
from django.utils import timezone

@login_required
@role_required('technician')
def technician_predict_moisture_view(request):
    """
    Handle both GET and POST requests for soil moisture prediction.
    """
    
    try:
        # If model isn't loaded, try to train it with default data
        model_loaded = load_model()
        if not model_loaded:
            try:
                training_result = train_model_with_db_data()
                if not training_result:
                    messages.error(request, "Model training failed. Please check the logs.")
                messages.info(request, "Model trained successfully with historical data")
            except Exception as train_error:
                messages.error(request, f"Model initialization failed: {str(train_error)}")
                return redirect('technician_predict_moisture_view')
            
        # Get technician's assigned location(s)
        assigned_locations = TechnicianLocationAssignment.objects.filter(
            technician=request.user
        ).values_list('location', flat=True).distinct()
        
        if not assigned_locations.exists():
            messages.error(request, "You are not assigned to any location. Please contact admin.")
            return redirect('technician_dashboard')
            
        location = assigned_locations[0]
            
        if request.method == 'GET':
        # Handle GET requests to display the prediction form
            context = {
                'weather_location': location,
                'current_moisture': None,
                'temperature': None,
                'humidity': None,
                'forecast_table': [],
            }
            return render(request, 'technician_dashboard.html', context)
    
        elif request.method == 'POST':
            # Handle POST requests to process prediction form and generate forecast
            try:
                # Extract form data
                current_moisture = float(request.POST.get('soil_moisture_percent'))
                temperature = float(request.POST.get('temperature'))
                humidity = float(request.POST.get('humidity'))

                # Validate inputs
                if not (0 <= current_moisture <= 100 and -50 <= temperature <= 50 and 0 <= humidity <= 100):
                    messages.error(request, "Invalid input values. Please check your inputs.")
                    return redirect('technician_predict_moisture_view')

                # Get 7-day forecast
                forecast = predict_future_moisture(
                    location=location,
                    current_moisture=current_moisture,
                    temperature=temperature,
                    humidity=humidity,
                    days=7
                )

                # Save predictions to database
                for prediction in forecast:
                    # Ensure timestamp is timezone-aware
                    dt = prediction['datetime']
                    if timezone.is_naive(dt):
                        dt = timezone.make_aware(dt)
                    TechnicianSoilMoisturePrediction.objects.create(
                        location=location,
                        timestamp=dt,
                        current_moisture=current_moisture,
                        temperature=prediction['temperature'],
                        humidity=prediction['humidity'],
                        precipitation=prediction['precipitation'],
                        predicted_category=prediction['predicted_category'],
                        predicted_moisture_value=prediction['predicted_moisture_value'],
                        confidence=prediction['confidence']
                    )

                # Prepare chart data
                chart_data = {
                    'labels': [pred['date'] for pred in forecast],
                    'moisture': [pred['predicted_moisture_value'] for pred in forecast],
                    'temperature': [pred['temperature'] for pred in forecast],
                    'humidity': [pred['humidity'] for pred in forecast]
                }

                 # Prepare forecast table for template
                forecast_table = [
                    {
                        'date': pred['date'],
                        'moisture': round(pred['predicted_moisture_value'], 2),
                        'temperature': round(pred['temperature'], 2),
                        'humidity': round(pred['humidity'], 2)
                    }
                    for pred in forecast
                ]

                # Prepare simplified chart data
                chart_data = {
                    'labels': [pred['date'] for pred in forecast],
                    'datasets': [
                        {
                            'label': 'Predicted Moisture (%)',
                            'data': [round(pred['predicted_moisture_value'], 2) for pred in forecast],
                            'borderColor': '#1E90FF',
                            'backgroundColor': 'rgba(30, 144, 255, 0.2)',
                            'yAxisID': 'y1'
                        },
                        {
                            'label': 'Temperature (°C)',
                            'data': [round(pred['temperature'], 2) for pred in forecast],
                            'borderColor': '#FF4500',
                            'backgroundColor': 'rgba(255, 69, 0, 0.2)',
                            'yAxisID': 'y2'
                        },
                        {
                            'label': 'Humidity (%)',
                            'data': [round(pred['humidity'], 2) for pred in forecast],
                            'borderColor': '#32CD32',
                            'backgroundColor': 'rgba(50, 205, 50, 0.2)',
                            'yAxisID': 'y1'
                        }
                    ]
                }

                context = {
                    'weather_location': location,
                    'current_moisture': current_moisture,
                    'temperature': temperature,
                    'humidity': humidity,
                    'forecast_table': forecast_table,
                    'chart_data_json': json.dumps(chart_data)
                }

                messages.success(request, "Predictions generated successfully!")
                return render(request, 'technician_dashboard.html', context)
            except Exception as e:
                messages.error(request, f"Error generating predictions: {str(e)}")
                return redirect('technician_predict_moisture_view')

    except Exception as e:
        messages.error(request, f"Error generating predictions: {str(e)}")
        return redirect('technician_predict_moisture_view')


#Generating Reports
@login_required
@role_required('technician')
def generate_report(request):
    if request.method == 'POST':
        report_type = request.POST.get('report_type')
        format_type = request.POST.get('format_type')
        start_date = request.POST.get('start_date', '')
        end_date = request.POST.get('end_date', '')

        # Validate inputs
        if report_type not in ['daily', 'weekly', 'monthly']:
            messages.error(request, 'Invalid report type.')
            return redirect('technician_dashboard')
        if format_type not in ['pdf', 'excel']:
            messages.error(request, 'Invalid format type.')
            return redirect('technician_dashboard')

        # Determine date range based on report type
        end_date = datetime.now() if not end_date else datetime.strptime(end_date, '%Y-%m-%d')
        if report_type == 'daily':
            start_date = end_date - timedelta(days=1)
        elif report_type == 'weekly':
            start_date = end_date - timedelta(days=7)
        else:  # monthly
            start_date = end_date - timedelta(days=30)
            
        start_date = start_date.date()
        end_date = end_date.date()

        # Query data
        moisture_records = SoilMoistureRecord.objects.filter(
            timestamp__range=[start_date, end_date]
        ).order_by('timestamp')
        prediction_records = TechnicianSoilMoisturePrediction.objects.filter(
            timestamp__range=[start_date, end_date]
        ).order_by('timestamp')

        # Filter by assigned locations for technicians
        if request.user.role == 'technician':
            assigned_farms = TechnicianLocationAssignment.objects.filter(technician=request.user)
            assigned_locations = assigned_farms.values_list('location', flat=True).distinct()
            moisture_records = moisture_records.filter(location__in=assigned_locations)
            prediction_records = prediction_records.filter(location__in=assigned_locations)
        
        # Aggregate statistics - improved version
        stats = {
            'avg_moisture': moisture_records.aggregate(
                avg=Avg('soil_moisture_percent')
            )['avg'] or 0.0,
            'min_moisture': moisture_records.aggregate(
                min=Min('soil_moisture_percent')
            )['min'] or 0.0,
            'max_moisture': moisture_records.aggregate(
                max=Max('soil_moisture_percent')
            )['max'] or 0.0
        }

        # Convert to float and round
        stats = {
            'avg_moisture': round(float(stats['avg_moisture']), 2),
            'min_moisture': round(float(stats['min_moisture']), 2),
            'max_moisture': round(float(stats['max_moisture']), 2)
        }

        if format_type == 'pdf':
            return generate_pdf_report(request, report_type, moisture_records, prediction_records, stats, start_date, end_date)
        else:  # excel
            return generate_excel_report(request, report_type, moisture_records, prediction_records, stats, start_date, end_date)

    return redirect('technician_dashboard' if request.user.role == 'technician' else 'admin_dashboard')

def generate_pdf_report(request, report_type, moisture_records, prediction_records, stats, start_date, end_date):
    buffer = BytesIO()
    p = canvas.Canvas(buffer, pagesize=letter)
    p.setFont("Helvetica", 12)

    # Title
    p.drawString(100, 750, f"{report_type.capitalize()} Soil Moisture Report")
    p.drawString(100, 730, f"Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

    # Statistics
    p.drawString(100, 700, "Summary Statistics:")
    p.drawString(100, 680, f"Average Moisture: {stats['avg_moisture']:.2f}%")
    p.drawString(100, 660, f"Min Moisture: {stats['min_moisture']:.2f}%")
    p.drawString(100, 640, f"Max Moisture: {stats['max_moisture']:.2f}%")

    # Moisture Records
    y = 600
    p.drawString(100, y, "Soil Moisture Records:")
    y -= 20
    for record in moisture_records[:10]:  # Limit to 10 for brevity
        p.drawString(100, y, f"{record.timestamp.strftime('%Y-%m-%d %H:%M')}: {record.location}, {record.soil_moisture_percent}%")
        y -= 20
        if y < 100:
            p.showPage()
            y = 750

    # Prediction Records
    p.drawString(100, y, "Prediction Records:")
    y -= 20
    for pred in prediction_records[:10]:  # Limit to 10 for brevity
        p.drawString(100, y, f"{pred.timestamp.strftime('%Y-%m-%d %H:%M')}: {pred.location}, Predicted: {pred.predicted_moisture_value}%")
        y -= 20
        if y < 100:
            p.showPage()
            y = 750

    p.showPage()
    p.save()
    buffer.seek(0)
    response = HttpResponse(content_type='application/pdf')
    response['Content-Disposition'] = f'attachment; filename="{report_type}_report_{datetime.now().strftime("%Y%m%d")}.pdf"'
    response.write(buffer.getvalue())
    buffer.close()
    return response

def generate_excel_report(request, report_type, moisture_records, prediction_records, stats, start_date, end_date):
    wb = Workbook()
    ws = wb.active
    ws.title = f"{report_type.capitalize()} Report"

    # Write headers
    ws.append([f"{report_type.capitalize()} Soil Moisture Report"])
    ws.append([f"Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"])
    ws.append([])
    ws.append(["Summary Statistics"])
    ws.append(["Average Moisture", f"{stats['avg_moisture']:.2f}%"])
    ws.append(["Min Moisture", f"{stats['min_moisture']:.2f}%"])
    ws.append(["Max Moisture", f"{stats['max_moisture']:.2f}%"])
    ws.append([])

    # Moisture Records
    ws.append(["Soil Moisture Records"])
    ws.append(["Timestamp", "Location", "Moisture (%)", "Status"])
    for record in moisture_records:
        ws.append([
            record.timestamp.strftime('%Y-%m-%d %H:%M'),
            record.location,
            record.soil_moisture_percent,
            record.status
        ])

    # Prediction Records
    ws.append([])
    ws.append(["Prediction Records"])
    ws.append(["Timestamp", "Location", "Predicted Moisture (%)", "Current Moisture (%)"])
    for pred in prediction_records:
        ws.append([
            pred.timestamp.strftime('%Y-%m-%d %H:%M'),
            pred.location,
            pred.predicted_moisture_value,
            pred.current_moisture
        ])

    buffer = BytesIO()
    wb.save(buffer)
    buffer.seek(0)
    response = HttpResponse(content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
    response['Content-Disposition'] = f'attachment; filename="{report_type}_report_{datetime.now().strftime("%Y%m%d")}.xlsx"'
    response.write(buffer.getvalue())
    buffer.close()
    return response
