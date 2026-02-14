# 🌾 FarmAI - Precision Agriculture Platform

> **Thesis Topic:** Configuring an Open-Source AI Model for Real-World Farm Analytics  
> **Supervisors:** Hassan Jaleel, Abubakr Muhammad (LUMS)  
> **Student:** Yahya Habib

---

## 📋 Project Overview

FarmAI is an open-source precision agriculture framework designed for Pakistani farms. It integrates satellite remote sensing, drone imagery, IoT sensor data, and machine learning analytics into a unified farm management dashboard.

### Key Features
- 🛰️ **Satellite Imagery** - Microsoft Planetary Computer integration (Sentinel-2, all 13 bands)
- 🚁 **Drone Support** - Upload and process drone-captured field images
- 📡 **IoT Integration** - ThingsBoard connectivity with LUMS IoT (icarus.lums.edu.pk)
- 🤖 **AI Analytics** - NDVI/NDWI indices, irrigation classification, GHG emissions
- 📊 **Interactive Dashboard** - Real-time analysis and visualization

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Dashboard (Bootstrap 5 + Leaflet)                │
├─────────────────────────────────────────────────────────────────────┤
│                   Django REST Framework APIs                        │
├───────────────┬───────────────┬───────────────┬────────────────────┤
│     Core      │      IoT      │    Imagery    │     Analytics      │
│  Farm/Field   │  ThingsBoard  │   Sentinel-2  │    ML Models       │
│   Devices     │    MQTT       │   NDVI/NDWI   │  GHG Calculator    │
├───────────────┴───────────────┴───────────────┴────────────────────┤
│         PostgreSQL + PostGIS    │    MinIO    │    Redis           │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Features

### 1. Manual Data Management
- CSV import/export for farms, fields, devices, assets
- Django Admin with interactive GIS maps (Google/ESRI/OSM layers)
- Field boundary validation (50km proximity to farm)

### 2. IoT Data Integration
- ThingsBoard REST API client (LUMS IoT compatible)
- MQTT ingestion for real-time sensor data
- Flexible JSON sensor storage (temperature, humidity, soil moisture)
- Time-series visualization with Chart.js

### 3. Satellite & Drone Imagery
- Microsoft Planetary Computer Sentinel-2 download
- All 13 Sentinel-2 bands (10m, 20m, 60m resolution)
- NDVI/NDWI heatmap generation and visualization
- MinIO S3-compatible object storage
- Drone image upload and processing

### 4. AI Analytics
| Feature | Implementation |
|---------|----------------|
| NDVI/NDWI Indices | Rasterio-based band calculations |
| Irrigation Classification | RandomForest ML model (82% accuracy) |
| GHG Emissions | IPCC Tier 1/2 compliant calculator |
| Crop Cycle Detection | Savitzky-Golay + peak detection |
| Harvest Prediction | Growth stage tracking |
| Sensor Placement | K-means clustering optimization |

### 5. Dashboard
- Farm/field selector with date range
- Real-time NDVI/NDWI analysis
- Satellite imagery gallery with band switching
- IoT time-series charts
- Settings modal for farm/field management

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- PostgreSQL 14+ with PostGIS
- Docker & Docker Compose
- GDAL/GEOS/PROJ libraries

### Installation

```bash
# Clone repository
git clone https://github.com/yahyahabib/farmai.git
cd farmai

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Start infrastructure (PostgreSQL, MinIO, Redis)
docker compose up -d

# Run migrations
python manage.py migrate

# Create superuser
python manage.py createsuperuser

# Train ML model (optional - pre-trained model included)
python manage.py shell < scripts/train_model.py

# Start development server
python manage.py runserver 0.0.0.0:8001
```

### Access Points
- **Dashboard:** http://localhost:8001/analytics/dashboard/
- **Admin:** http://localhost:8001/admin/
- **API:** http://localhost:8001/api/
- **MinIO Console:** http://localhost:9001/

---

## 📁 Project Structure

```
farmai/
├── core/               # Farm, Field, Device, Asset models
│   ├── models.py       # PostGIS geometry fields
│   ├── admin.py        # GIS widgets with Google Maps
│   └── management/     # CSV import/export commands
├── iot/                # IoT integration
│   ├── thingsboard_client.py  # REST API client
│   ├── mqtt_ingest.py  # MQTT listener
│   └── models.py       # SensorReading, WeatherData
├── imagery/            # Satellite & drone processing
│   ├── planetary_downloader.py  # Sentinel-2 download
│   ├── ndvi_processor.py        # NDVI/NDWI calculation
│   └── geoai_processor.py       # Advanced GeoAI
├── analytics/          # ML & analytics
│   ├── ml_models.py    # Irrigation classifier, weed detection
│   ├── ghg_calculator.py  # Carbon footprint
│   ├── time_series.py  # Crop cycle detection
│   └── tests.py        # Unit tests (20 tests)
├── farm_system/        # Django project settings
├── templates/          # Admin GIS widget templates
├── models/             # Trained ML models (.pkl)
└── media/              # Uploaded files
```

---

## 🔧 Configuration

### Environment Variables
```bash
# Django
export DJANGO_SECRET_KEY="your-secret-key"
export DJANGO_DEBUG=True

# Database
export POSTGRES_HOST=localhost
export POSTGRES_PORT=5432
export POSTGRES_DB=farmdb
export POSTGRES_USER=farm_user
export POSTGRES_PASSWORD=farm_pass

# MinIO
export MINIO_ENDPOINT=localhost:9000
export MINIO_ACCESS_KEY=minioadmin
export MINIO_SECRET_KEY=minioadmin

# Farm defaults (Pakistan/LUMS)
export FARMAI_DEFAULT_LAT=31.4697
export FARMAI_DEFAULT_LON=74.4101
export FARMAI_DEFAULT_ZOOM=12
```

---

## 📊 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/farms/` | GET, POST | Farm CRUD |
| `/api/fields/` | GET, POST | Field boundaries (GeoJSON) |
| `/api/analytics/run/` | POST | Run NDVI/NDWI analysis |
| `/api/satellite-images/` | GET | List stored imagery |
| `/api/iot/sync/` | POST | Sync ThingsBoard data |
| `/api/iot/readings/` | GET | Sensor readings |

---

## 🧪 Testing

```bash
# Run all analytics tests
python manage.py test analytics.tests -v 2

# Run specific test class
python manage.py test analytics.tests.GHGCalculatorTests

# Coverage report
coverage run manage.py test
coverage report
```

---

## ML Models

### Irrigation Classifier
- **Algorithm:** RandomForest (100 trees)
- **Features:** NDVI stats, NDWI, soil moisture, temperature, precipitation
- **Classes:** irrigated, partially_irrigated, rainfed
- **Accuracy:** 82%
- **Model file:** `models/irrigation_classifier.pkl`

### Training
```python
from analytics.ml_models import IrrigationClassifier
classifier = IrrigationClassifier()
metrics = classifier.train(X, y, save_model=True)
```

---

## Pakistan-Specific Features

- Default coordinates: LUMS, Lahore (31.47°N, 74.41°E)
- LUMS IoT integration: icarus.lums.edu.pk
- GHG emission factors calibrated for Pakistan
- Crop parameters for wheat, rice, cotton, sugarcane, maize

---

## License

MIT License - Open Source

---

## Contributors

- **Yahya Habib** - Developer
- **Hassan Jaleel** - Supervisor
- **Abubakr Muhammad** - Supervisor

---

## References

- [Microsoft Planetary Computer](https://planetarycomputer.microsoft.com/)
- [ThingsBoard IoT Platform](https://thingsboard.io/)
- [IPCC GHG Guidelines](https://www.ipcc-nggip.iges.or.jp/)