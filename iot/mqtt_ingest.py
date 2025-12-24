import paho.mqtt.client as mqtt
import json
import time
from core.models import Device, SensorReading
from django.utils.timezone import now

# MQTT Configuration
BROKER = "icarus.lums.edu.pk"
PORT = 1883
USERNAME = "YOUR_DEVICE_TOKEN"
PASSWORD = ""

# Enhanced Callback when connected to MQTT broker
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Connected successfully to MQTT broker")
        client.subscribe("v1/devices/me/telemetry")
    else:
        print(f"Connection failed with code {rc}")

# Enhanced Callback when a message is received
def on_message(client, userdata, msg):
    try:
        payload = json.loads(msg.payload)
        device = Device.objects.filter(thingsboard_id=payload.get('device_id')).first()
        if device:
            SensorReading.objects.create(
                device=device,
                timestamp=now(),
                temperature=payload.get('temperature'),
                moisture=payload.get('moisture')
            )
            print(f"Data saved for device {device.name}")
        else:
            print("Device not found")
    except json.JSONDecodeError:
        print("Invalid JSON received")
    except Exception as e:
        print("Error processing message: ", e)

# Enhanced MQTT client initialization with reconnection logic
while True:
    try:
        client = mqtt.Client()
        client.username_pw_set(USERNAME, PASSWORD)
        client.on_connect = on_connect
        client.on_message = on_message

        print("Attempting to connect to MQTT broker...")
        client.connect(BROKER, PORT, 60)
        client.loop_forever()
    except Exception as e:
        print("Connection error: ", e)
        print("Reconnecting in 5 seconds...")
        time.sleep(5)