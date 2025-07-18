from django.shortcuts import render, redirect
from django.core.mail import send_mail
from django.conf import settings
from userAuthentication.decorators import role_required, roles_required
from django.contrib.auth.decorators import login_required
import logging
import csv
import io
from django.contrib import messages
from datetime import datetime
from ml_model.ml_model import train_model, get_model_metrics, predict_soil_status
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
from django.contrib import messages
from userAuthentication.models import CustomUser
from urllib.parse import urlencode 
from django.urls import reverse
from techniciandashboard.models import TechnicianLocationAssignment
from ml_model.soil_moisture_prediction import train_model_with_db_data, load_model, load_uploaded_model
from admindashboard.models import SoilMoistureRecord



load_dotenv()
weather_api = os.getenv("OPENWEATHER_API_KEY")

logger = logging.getLogger(__name__)


@role_required('admin')
def admin_dashboard(request):
    # Get filter parameters
    location = request.GET.get('location', '')
    start_date = request.GET.get('start_date', '')
    end_date = request.GET.get('end_date', '')
    show_all = request.GET.get('show_all', 'false').lower() == 'true'

    # Query soil moisture records
    records = SoilMoistureRecord.objects.all().order_by('-timestamp')

    # Apply filters
    if location:
        records = records.filter(location=location)
    
    if start_date:
        try:
            start_date = datetime.strptime(start_date, '%Y-%m-%d')
            records = records.filter(timestamp__gte=start_date)
        except ValueError:
            messages.error(request, 'Invalid start date format.')
    
    if end_date:
        try:
            end_date = datetime.strptime(end_date, '%Y-%m-%d')
            records = records.filter(timestamp__lte=end_date)
        except ValueError:
            messages.error(request, 'Invalid end date format.')

    # Get unique locations for dropdown
    locations = SoilMoistureRecord.objects.values_list('location', flat=True).distinct()
    
    # Set default location - use first location if available, otherwise a hardcoded default
    default_location = locations[0] if locations else "Nairobi"
    
    # Get weather data
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

    model_metrics = get_model_metrics()

    technicians = CustomUser.objects.filter(role='technician').select_related()
    farms = TechnicianLocationAssignment.objects.all()
    
    # Limit records to 10 unless show_all is true
    displayed_records = records[:10] if not show_all else records

    context = {
        'user': request.user,
        'records': displayed_records,
        'total_records': records.count(),
        'show_all': show_all,
        'locations': locations,
        'selected_location': location,
        'start_date': start_date,
        'end_date': end_date,
        'model_metrics': model_metrics,
        'average_moisture': average_moisture,
        'average_temperature': average_temperature,
        'average_humidity': average_humidity,
        'chart_data': json.dumps(chart_data),
        'current_weather': current_weather,
        'weather_location': default_location,
        'technicians': technicians,
        'farms': farms,
    }
    return render(request, 'admin_dashboard.html', context)




# Uploading the csv file to the database
def upload_csv(request):
    if request.method == 'POST':
        csv_file = request.FILES.get('csv-upload')
        if not csv_file:
            messages.error(request, 'No file uploaded.')
            return redirect('admin_dashboard')

        if not csv_file.name.endswith('.csv'):
            messages.error(request, 'Please upload a valid CSV file.')
            return redirect('admin_dashboard')

        try:
            # Read CSV file
            csv_data = csv_file.read().decode('utf-8')
            io_string = io.StringIO(csv_data)
            reader = csv.DictReader(io_string)

            # Validate required columns
            required_columns = [
                'record_id', 'sensor_id', 'location', 'soil_moisture_percent',
                'temperature_celsius', 'humidity_percent', 'timestamp',
                'status', 'battery_voltage', 'irrigation_action'
            ]
            if not all(col in reader.fieldnames for col in required_columns):
                messages.error(request, 'CSV file is missing required columns.')
                return redirect('admin_dashboard')

            # Process each row
            for row in reader:
                try:
                    # Convert timestamp to datetime
                    timestamp = datetime.strptime(row['timestamp'], '%Y-%m-%d %H:%M:%S')

                    # Create or update record
                    SoilMoistureRecord.objects.update_or_create(
                        record_id=int(row['record_id']),
                        defaults={
                            'sensor_id': row['sensor_id'],
                            'location': row['location'],
                            'soil_moisture_percent': float(row['soil_moisture_percent']),
                            'temperature_celsius': float(row['temperature_celsius']),
                            'humidity_percent': float(row['humidity_percent']),
                            'timestamp': timestamp,
                            'status': row['status'],
                            'battery_voltage': float(row['battery_voltage']),
                            'irrigation_action': row['irrigation_action']
                        }
                    )
                except (ValueError, KeyError) as e:
                    messages.warning(request, f"Error processing row {row['record_id']}: {str(e)}")
                    continue

            messages.success(request, 'CSV data uploaded successfully!')
            return redirect('admin_dashboard')

        except Exception as e:
            messages.error(request, f'Error processing CSV file: {str(e)}')
            return redirect('admin_dashboard')

    return render(request, 'admin_dashboard')

# Upload a ml model
# Upload a ml model
@login_required
@role_required('admin')
def upload_model(request):
    if request.method == 'POST':
        model_file = request.FILES.get('irrigation-ml-model-upload')
        if not model_file:
            messages.error(request, 'No file uploaded.')
            return redirect('admin_dashboard')
        
        if not model_file.name.endswith(('.pkl', '.h5')):
            messages.error(request, 'Please upload a valid model file (.pkl or .h5).')
            return redirect('admin_dashboard')
        
        try:
            # Save the model file
            model_path = os.path.join(settings.BASE_DIR, 'ml_models', model_file.name)
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            with open(model_path, 'wb') as f:
                for chunk in model_file.chunks():
                    f.write(chunk)
            
            # Optionally retrain the model
            if request.POST.get('irrigation-retrain'):
                _, metrics = train_model()  # Get metrics from retraining
                messages.success(request, f'Model retrained successfully! Accuracy: {metrics["accuracy"]}%')
            else:
                messages.success(request, 'Model uploaded successfully!')
            
            return redirect('admin_dashboard')
        
        except Exception as e:
            messages.error(request, f'Error uploading model: {str(e)}')
            return redirect('admin_dashboard')
        
@login_required
@role_required('admin')
def upload_model_soil_moisture(request):
    if request.method == 'POST':
        model_file = request.FILES.get('soil-moisture-ml-model-upload')
        retrain = request.POST.get('soil-moisture-retrain') == 'on'

        if not model_file:
            messages.error(request, 'No file uploaded.')
            return redirect('admin_dashboard')
        
        if not model_file.name.endswith('.pkl'):
            messages.error(request, 'Please upload a valid model file (.pkl).')
            return redirect('admin_dashboard')
        
        try:
            # Load the uploaded model
            success = load_uploaded_model(model_file)
            if not success:
                messages.error(request, 'Failed to load the uploaded model.')
                return redirect('admin_dashboard')
            
            # Optionally retrain the model
            if retrain:
                metrics = train_model_with_db_data(retrain=True)
                if metrics == 'classifier':
                    accuracy = round(metrics['test_accuracy'] * 100, 2)
                    messages.success(request, f'Model retrained successfully! Test Accuracy: {accuracy}%')
                else:
                    rmse = round(metrics['test_rmse'], 2)
                    r2_score = round(metrics['test_r2'] * 100, 2)
                    messages.success(request, f'Model retrained successfully! Test RMSE: {rmse}, R² Score: {r2_score}%')
            else:
                messages.success(request, 'Model uploaded successfully!')
            
            return redirect('admin_dashboard')
        
        except Exception as e:
            messages.error(request, f'Error processing model: {str(e)}')
            return redirect('admin_dashboard')
    
    return redirect('admin_dashboard')


@login_required
@role_required('admin')
def assign_technician(request):
    if request.method == 'POST':
        technician_id = request.POST.get('technician_id')
        location = request.POST.get('location')

        # Validate that the location exists in SoilMoistureRecord
        if not SoilMoistureRecord.objects.filter(location=location).exists():
            messages.error(request, f"No soil moisture records found for location: {location}")
            logger.error(f"Attempted to assign technician to invalid location: {location}")
            return redirect('admin_dashboard')

        try:
            technician = CustomUser.objects.get(id=technician_id, role='technician')

            # Check if the technician is already assigned to the location
            if TechnicianLocationAssignment.objects.filter(technician=technician, location=location).exists():
                messages.warning(
                    request,
                    f"Technician {technician.get_full_name() or technician.username} is already assigned to {location}."
                )
                logger.warning(f"Attempted to assign already assigned technician {technician.email} to location {location}")
            else:
                # Create new assignment
                TechnicianLocationAssignment.objects.create(
                    technician=technician,
                    location=location
                )
                messages.success(
                    request,
                    f"Technician {technician.get_full_name() or technician.username} assigned to {location} successfully!"
                )
                logger.info(f"Technician {technician.email} assigned to location {location} by {request.user.email}")

        except CustomUser.DoesNotExist:
            messages.error(request, "Invalid technician selected.")
            logger.error(f"Failed to assign technician: Invalid technician ID {technician_id}")
        except Exception as e:
            messages.error(request, f"Error assigning technician: {str(e)}")
            logger.error(f"Error assigning technician: {str(e)}")

        return redirect('admin_dashboard')

    return redirect('admin_dashboard')

@login_required
@role_required('admin')
def unassign_technician(request):
    if request.method == 'POST':
        technician_id = request.POST.get('technician_id')
        location = request.POST.get('location')

        try:
            technician = CustomUser.objects.get(id=technician_id, role='technician')

            # Check if the technician is assigned to the location
            assignment = TechnicianLocationAssignment.objects.filter(
                technician=technician,
                location=location
            ).first()

            if assignment:
                assignment.delete()

                # Check if any other technicians are assigned to this location
                remaining_assignments = TechnicianLocationAssignment.objects.filter(location=location).count()
                if remaining_assignments == 0:
                    # Delete SoilMoistureRecord entries for this location
                    messages.success(
                        request,
                        f"Technician {technician.get_full_name() or technician.username} unassigned from {location} "
                    )
                    logger.info(
                        f"Technician {technician.email} unassigned from location {location} by {request.user.email}. "
                    )
                else:
                    messages.success(
                        request,
                        f"Technician {technician.get_full_name() or technician.username} unassigned from {location}."
                    )
                    logger.info(f"Technician {technician.email} unassigned from location {location} by {request.user.email}")

            else:
                messages.warning(
                    request,
                    f"Technician {technician.get_full_name() or technician.username} is not assigned to {location}."
                )
                logger.warning(f"Attempted to unassign unassigned technician {technician.email} from location {location}")

        except CustomUser.DoesNotExist:
            messages.error(request, "Invalid technician selected.")
            logger.error(f"Failed to unassign technician: Invalid technician ID {technician_id}")
        except Exception as e:
            messages.error(request, f"Error unassigning technician: {str(e)}")
            logger.error(f"Error unassigning technician: {str(e)}")

        return redirect('admin_dashboard')

    return redirect('admin_dashboard')


import urllib.request

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

from admindashboard.models import SoilMoisturePrediction
@login_required
@role_required('admin')
def predict_moisture(request):
    if request.method == 'POST':
        try:
            location = request.POST.get('location')
            current_moisture = float(request.POST.get('soil_moisture_percent'))
            temperature = float(request.POST.get('temperature'))
            humidity = float(request.POST.get('humidity'))
            
            weather_forecast = get_weather_forecast(location)
            
            # Make prediction
            prediction_result = predict_soil_status(
                soil_moisture=current_moisture,
                temperature=temperature,
                humidity=humidity,
                location=location
            )
            
            # Store prediction
            SoilMoisturePrediction.objects.create(
                location=location,
                predicted_moisture=prediction_result['status'],  # Use status or derived value
                current_moisture=current_moisture,
                temperature=temperature,
                humidity=humidity,
                precipitation=weather_forecast.get('rain', {}).get('1h', 0) if weather_forecast else 0,
                prediction_for=datetime.now() + timedelta(hours=24)
            )
            
            # Store prediction in context
            context = {
                'prediction': prediction_result['status'],
                'location': location,
                'current_moisture': current_moisture,
                'temperature': temperature,
                'humidity': humidity,
                'confidence': prediction_result['confidence'],
                'method': prediction_result['method'],
                'irrigation_recommendation': prediction_result['irrigation_recommendation'],
                'input_values': prediction_result['input_values']
            }
            return render(request, 'admindashboard', context)
        
        except ValueError as e:
            return redirect('admin_dashboard')
    
    return render(request, 'admin_dashboard.html')

from techniciandashboard.models import TechnicianSoilMoisturePrediction
#
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
            return redirect('technician_dashboard' if request.user.role == 'technician' else 'admin_dashboard')
        if format_type not in ['pdf', 'excel']:
            messages.error(request, 'Invalid format type.')
            return redirect('technician_dashboard' if request.user.role == 'technician' else 'admin_dashboard')

        # Determine date range based on report type
        end_date = datetime.now() if not end_date else datetime.strptime(end_date, '%Y-%m-%d')
        if report_type == 'daily':
            start_date = end_date - timedelta(days=1)
        elif report_type == 'weekly':
            start_date = end_date - timedelta(days=7)
        else:  # monthly
            start_date = end_date - timedelta(days=30)

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
