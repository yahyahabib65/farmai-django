import math

class KalmanFilter:
    """
    Simple 1D Kalman Filter for sensor data smoothing.
    """
    def __init__(self, R=10.0, Q=0.1, A=1.0, B=0.0, C=1.0):
        self.R = R  # Measurement Noise (Higher = smoother/slower)
        self.Q = Q  # Process Noise (Higher = follows spikes faster)
        self.A = A
        self.B = B
        self.C = C
        self.cov = float('nan')
        self.x = float('nan')

    def filter(self, measurement):
        measurement = float(measurement)
        if math.isnan(self.x):
            self.x = (1 / self.C) * measurement
            self.cov = (1 / self.C) * self.R * (1 / self.C)
            return measurement
        
        # Prediction
        pred_x = (self.A * self.x)
        pred_cov = ((self.A * self.cov) * self.A) + self.Q

        # Update
        K = pred_cov * self.C * (1 / ((self.C * pred_cov * self.C) + self.R))
        self.x = pred_x + K * (measurement - (self.C * pred_x))
        self.cov = pred_cov - (K * self.C * pred_cov)
        
        return self.x

def normalize_sensor_data(data_list, value_key='value', key_groups=['device_id', 'sensor_type']):
    """
    Applies Kalman filter to a list of dictionaries, respecting sensor groups.
    Adds 'normalized_value' key to each dictionary.
    """
    # 1. Group data so we don't mix different sensors
    groups = {}
    for item in data_list:
        # Create a unique key for this sensor stream (e.g., Device 1's Temperature)
        group_key = tuple(item.get(k) for k in key_groups)
        if group_key not in groups:
            groups[group_key] = []
        groups[group_key].append(item)

    # 2. Apply Filter per group
    processed_list = []
    
    for _, items in groups.items():
        # Ensure time order for filtering
        items.sort(key=lambda x: x.get('timestamp', ''))
        
        # Initialize filter for this specific sensor stream
        kf = KalmanFilter(R=15.0, Q=0.1) # R=15 handles noisy agricultural sensors well
        
        for item in items:
            raw_val = item.get(value_key)
            if raw_val is not None:
                item['normalized_value'] = round(kf.filter(raw_val), 2)
            else:
                item['normalized_value'] = None
            processed_list.append(item)
            
    # 3. Return combined list (caller can resort if needed)
    return processed_list
