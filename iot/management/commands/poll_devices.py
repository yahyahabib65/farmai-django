import requests
import time
import os
from datetime import datetime, timezone
from django.core.management.base import BaseCommand
from core.models import Device
from iot.models import SensorReading

class Command(BaseCommand):
    help = 'Auto-logins and polls Thingsboard Telemetry'

    def handle(self, *args, **options):
        # --- CONFIGURATION ---
        TB_HOST = os.getenv("TB_HOST", "http://icarus.lums.edu.pk")
        TB_USER = os.getenv("TB_USER", "24280001@lums.edu.pk")
        TB_PASS = os.getenv("TB_PASS", "LUMS12345")
        # ---------------------

        self.stdout.write("1. Authenticating with Thingsboard...")
        
        # Step 1: Get JWT Token programmatically
        auth_url = f"{TB_HOST}/api/auth/login"
        try:
            auth_resp = requests.post(auth_url, json={"username": TB_USER, "password": TB_PASS})
            if auth_resp.status_code != 200:
                self.stdout.write(self.style.ERROR("Authentication Failed! Check username/password."))
                return
            
            token = auth_resp.json()['token']
            headers = {
                'Content-Type': 'application/json',
                'X-Authorization': f'Bearer {token}'
            }
            self.stdout.write(self.style.SUCCESS("Authentication Successful."))
            
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Connection Error: {e}"))
            return

        # Step 2: Poll Devices
        self.stdout.write("2. Polling Devices...")
        
        for device in Device.objects.all():
            try:
                # Time window: Last 10 minutes
                now = int(time.time() * 1000)
                start_ts = now - (10 * 60 * 1000)

                # The Endpoint you selected
                # FIX: Added moisture and soil_moisture to keys so API actually returns them
                url = (
                    f"{TB_HOST}/api/plugins/telemetry/DEVICE/{device.thingsboard_id}/values/timeseries"
                    f"?keys=temperature,moisture,soil_moisture"
                    f"&startTs={start_ts}&endTs={now}"
                )

                response = requests.get(url, headers=headers)

                if response.status_code == 200:
                    data = response.json()
                    
                    # Logic to find latest value
                    temp = 0.0
                    moist = 0.0
                    timestamp = datetime.now(timezone.utc)
                    found_data = False

                    # Extract Temperature
                    if 'temperature' in data and data['temperature']:
                        latest = data['temperature'][0]
                        temp = float(latest['value'])
                        # Use the timestamp from the data, not current time
                        timestamp = datetime.fromtimestamp(latest['ts']/1000, tz=timezone.utc)
                        found_data = True

                    # Extract Moisture (handling common key variations)
                    if 'moisture' in data and data['moisture']:
                        moist = float(data['moisture'][0]['value'])
                        found_data = True
                    elif 'soil_moisture' in data and data['soil_moisture']:
                        moist = float(data['soil_moisture'][0]['value'])
                        found_data = True

                    # Save to DB
                    if found_data:
                        SensorReading.objects.update_or_create(
                            device=device,
                            timestamp=timestamp,
                            defaults={'temperature': temp, 'moisture': moist}
                        )
                        self.stdout.write(f"Updated {device.name}: {temp}C, {moist}% Moisture")
                    else:
                        self.stdout.write(self.style.WARNING(f"No telemetry data found for {device.name}"))
                
                else:
                    self.stdout.write(self.style.WARNING(f"Device {device.name} not found or no data (Code {response.status_code})"))

            except Exception as e:
                self.stdout.write(self.style.ERROR(f"Error on {device.name}: {e}"))