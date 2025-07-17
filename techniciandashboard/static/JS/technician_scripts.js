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

// Add this JavaScript to your Django template
// Make sure Chart.js is loaded before this script

document.addEventListener('DOMContentLoaded', function() {
    let chart = null;

    function createMoistureChart() {
        const ctx = document.getElementById('soilMoistureChart').getContext('2d');
        
        // Destroy existing chart if it exists
        if (chart) {
            chart.destroy();
        }
        
        // Get chart data from the Django template
        const chartDataScript = document.getElementById('chartData');
        if (!chartDataScript) {
            console.log('No chart data available');
            return;
        }
        
        let chartData;
        try {
            chartData = JSON.parse(chartDataScript.textContent);
        } catch (e) {
            console.error('Error parsing chart data:', e);
            return;
        }
        
        // Create the chart with the data from Django
        chart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: chartData.labels,
                datasets: chartData.datasets
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: '7-Day Soil Moisture Forecast'
                    },
                    legend: {
                        display: true,
                        position: 'top'
                    }
                },
                scales: {
                    x: {
                        display: true,
                        title: {
                            display: true,
                            text: 'Date'
                        }
                    },
                    y1: {
                        type: 'linear',
                        display: true,
                        position: 'left',
                        title: {
                            display: true,
                            text: 'Moisture & Humidity (%)'
                        },
                        min: 0,
                        max: 100
                    },
                    y2: {
                        type: 'linear',
                        display: true,
                        position: 'right',
                        title: {
                            display: true,
                            text: 'Temperature (°C)'
                        },
                        grid: {
                            drawOnChartArea: false,
                        },
                    }
                },
                interaction: {
                    mode: 'index',
                    intersect: false,
                }
            }
        });
    }

    // Initialize chart when page loads
    createMoistureChart();

    // Re-create chart when form is submitted (if needed)
    const predictionForm = document.getElementById('predictionForm');
    if (predictionForm) {
        predictionForm.addEventListener('submit', function() {
            // Chart will be recreated when the page reloads with new data
            setTimeout(createMoistureChart, 100);
        });
    }
});

