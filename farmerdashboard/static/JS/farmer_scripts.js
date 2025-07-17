// soil_moisture_app/static/js/admin_scripts.js
document.addEventListener('DOMContentLoaded', function() {
    // Sidebar Toggle for Mobile
    const sidebarToggle = document.getElementById('sidebarToggle');
    if (sidebarToggle) {
        sidebarToggle.addEventListener('click', function() {
            document.querySelector('.sidebar').classList.toggle('active');
        });
    }

    // Chart.js for Moisture Trends
    const moistureCtx = document.getElementById('moistureChart');
    if (moistureCtx) {
        // Get chart data from Django template variable
        const chartDataElement = document.getElementById('chartData');
        let chartData = null;
        
        if (chartDataElement) {
            try {
                chartData = JSON.parse(chartDataElement.textContent);
            } catch (e) {
                console.error('Error parsing chart data:', e);
            }
        }

        if (chartData && chartData.labels && chartData.data) {
            new Chart(moistureCtx.getContext('2d'), {
                type: 'line',
                data: {
                    labels: chartData.labels,
                    datasets: [{
                        label: 'Average Soil Moisture (%)',
                        data: chartData.data,
                        borderColor: '#3498db',
                        backgroundColor: 'rgba(52, 152, 219, 0.1)',
                        borderWidth: 2,
                        fill: true,
                        tension: 0.4,
                        pointBackgroundColor: '#3498db',
                        pointBorderColor: '#ffffff',
                        pointRadius: 5,
                        pointHoverRadius: 7
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        title: {
                            display: true,
                            text: 'Daily Average Soil Moisture Trends'
                        },
                        legend: {
                            display: true,
                            position: 'top'
                        }
                    },
                    scales: {
                        x: {
                            title: {
                                display: true,
                                text: 'Date'
                            },
                            grid: {
                                display: true,
                                color: 'rgba(0,0,0,0.1)'
                            }
                        },
                        y: {
                            beginAtZero: true,
                            max: 100,
                            title: {
                                display: true,
                                text: 'Moisture (%)'
                            },
                            grid: {
                                display: true,
                                color: 'rgba(0,0,0,0.1)'
                            },
                            ticks: {
                                callback: function(value) {
                                    return value + '%';
                                }
                            }
                        }
                    },
                    interaction: {
                        intersect: false,
                        mode: 'index'
                    },
                    hover: {
                        animationDuration: 200
                    }
                }
            });
        } else {
            // Display message when no data is available
            const canvas = moistureCtx;
            const ctx = canvas.getContext('2d');
            ctx.font = '16px Arial';
            ctx.fillStyle = '#666';
            ctx.textAlign = 'center';
            ctx.fillText('No data available for chart', canvas.width / 2, canvas.height / 2);
        }
    }

    // Chart.js for Prediction Chart
    const predictionCtx = document.getElementById('predictionChart');
    if (predictionCtx) {
        const predictionDataElement = document.getElementById('predictionData');
        let predictions = null;
        
        if (predictionDataElement) {
            try {
                predictions = JSON.parse(predictionDataElement.textContent);
            } catch (e) {
                console.error('Error parsing prediction data:', e);
            }
        }

        if (predictions && predictions.length > 0) {
            new Chart(predictionCtx.getContext('2d'), {
                type: 'line',
                data: {
                    labels: predictions.map(p => new Date(p.timestamp).toLocaleString()),
                    datasets: [{
                        label: 'Predicted Moisture (%)',
                        data: predictions.map(p => p.predicted_moisture),
                        borderColor: 'rgba(75, 192, 192, 1)',
                        backgroundColor: 'rgba(75, 192, 192, 0.1)',
                        borderWidth: 2,
                        fill: false,
                        tension: 0.4,
                        pointRadius: 4
                    }, {
                        label: 'Input Moisture (%)',
                        data: predictions.map(p => p.input_moisture),
                        borderColor: 'rgba(255, 99, 132, 1)',
                        backgroundColor: 'rgba(255, 99, 132, 0.1)',
                        borderWidth: 2,
                        fill: false,
                        tension: 0.4,
                        pointRadius: 4
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        title: {
                            display: true,
                            text: 'Moisture Predictions vs Input Values'
                        }
                    },
                    scales: {
                        x: {
                            title: {
                                display: true,
                                text: 'Timestamp'
                            }
                        },
                        y: {
                            title: {
                                display: true,
                                text: 'Moisture (%)'
                            },
                            ticks: {
                                callback: function(value) {
                                    return value + '%';
                                }
                            }
                        }
                    }
                }
            });
        }
    }

    // Handle section navigation
    const navLinks = document.querySelectorAll('.nav-link');
    const sections = document.querySelectorAll('.content-section');

    navLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            const targetId = this.getAttribute('href').substring(1);
            
            // Hide all sections
            sections.forEach(section => {
                section.style.display = 'none';
            });
            
            // Show target section
            const targetSection = document.getElementById(targetId);
            if (targetSection) {
                targetSection.style.display = 'block';
            }
            
            // Update active nav link
            navLinks.forEach(navLink => {
                navLink.classList.remove('active');
            });
            this.classList.add('active');
        });
    });

    // Show dashboard section by default
    const dashboardSection = document.getElementById('dashboard');
    if (dashboardSection) {
        dashboardSection.style.display = 'block';
    }
});

function updateChart(location) {
    const canvas = document.getElementById('moistureChart');
    if (!canvas) {
        console.error('Canvas element with ID "moistureChart" not found.');
        return;
    }

    const loadingIndicator = document.createElement('div');
    loadingIndicator.id = 'loading';
    loadingIndicator.textContent = 'Loading...';
    canvas.parentNode.appendChild(loadingIndicator);

    if (!location) {
        if (moistureChart) {
            moistureChart.destroy();
            moistureChart = null;
        }
        canvas.style.display = 'none';
        loadingIndicator.remove();
        return;
    }

    canvas.style.display = 'block';

    fetch(`/api/soil-moisture/?location=${encodeURIComponent(location)}`, {
        method: 'GET',
        headers: {
            'X-Requested-With': 'XMLHttpRequest'
        }
    })
        .then(response => {
            loadingIndicator.remove();
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            if (data.error) {
                console.error('Error from server:', data.error);
                alert('No data available for this location.');
                if (moistureChart) {
                    moistureChart.destroy();
                    moistureChart = null;
                }
                canvas.style.display = 'none';
                return;
            }

            if (moistureChart) {
                moistureChart.destroy();
            }

            const ctx = canvas.getContext('2d');
            moistureChart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: data.labels,
                    datasets: [{
                        label: `Soil Moisture (%) - ${location}`,
                        data: data.data,
                        borderColor: 'rgba(75, 192, 192, 1)',
                        backgroundColor: 'rgba(75, 192, 192, 0.2)',
                        borderWidth: 2,
                        tension: 0.1,
                        fill: true
                    }]
                },
                options: {
                    responsive: true,
                    plugins: {
                        legend: {
                            position: 'top',
                        },
                        tooltip: {
                            mode: 'index',
                            intersect: false,
                        }
                    },
                    scales: {
                        y: {
                            beginAtZero: false,
                            title: {
                                display: true,
                                text: 'Soil Moisture (%)'
                            }
                        },
                        x: {
                            title: {
                                display: true,
                                text: 'Date'
                            }
                        }
                    }
                }
            });
        })
        .catch(error => {
            loadingIndicator.remove();
            console.error('Error fetching data:', error);
            alert('Failed to load moisture data. Please try again.');
            canvas.style.display = 'none';
        });
}


// User Profile Popover
document.addEventListener('DOMContentLoaded', function() {
    const userIcon = document.querySelector('.dashboard-header .bi-person');
    const popover = document.getElementById('userPopover');
    const closePopover = document.getElementById('closePopover');
    const backdrop = document.createElement('div');
    backdrop.className = 'popover-backdrop';
    document.body.appendChild(backdrop);

    // Toggle popover
    userIcon.addEventListener('click', function(e) {
        e.stopPropagation();
        popover.classList.toggle('show');
        backdrop.classList.toggle('show');
    });

    // Close popover when clicking close button
    closePopover.addEventListener('click', function() {
        popover.classList.remove('show');
        backdrop.classList.remove('show');
    });

    // Close popover when clicking backdrop
    backdrop.addEventListener('click', function() {
        popover.classList.remove('show');
        backdrop.classList.remove('show');
    });

    // Close popover when clicking outside
    document.addEventListener('click', function(e) {
        if (!popover.contains(e.target) && e.target !== userIcon) {
            popover.classList.remove('show');
            backdrop.classList.remove('show');
        }
    });
});