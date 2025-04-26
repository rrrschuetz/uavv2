#!/usr/bin/env python3
"""
Full-Workflow:
1) 2D-Kalibrierung (X/Y) mit sample_limit oder STRG+C
2) Laden der Kalibrierungsdaten
3) Heading-Berechnung mit optionaler Tilt-Compensation
4) Debug-Ausgaben, um ggf. Achsen-Vorzeichen oder -Vertauschung zu korrigieren
"""

import time
import json
import math
import board
import qmc5883l as qmc5883

# --- Globale Pitch/Roll (in Grad), bitte hier mit Deinen Werten füllen bzw. dynamisch setzen ---
Gpitch = -2.197265625
Groll  =   5.8392333984375

# --- Sensor initialisieren ---
i2c = board.I2C()
qmc = qmc5883.QMC5883L(i2c)
qmc.output_data_rate = qmc5883.OUTPUT_DATA_RATE_200

# --- Cache für Kalibrierungsdaten ---
Gcalibration_cache = None

def calibrate_2d_magnetometer(sample_limit=None, delay=0.05):
    """
    Führt 2D-Kalibrierung (nur X/Y) durch und speichert offsets/scales in JSON.
    """
    print("Starte 2D-Magnetometer-Kalibrierung (X/Y). Drehe ruhig 360° um Z.")
    print("STRG+C beendet.")
    samples = []
    try:
        while True:
            x, y, z = qmc.magnetic
            samples.append((x, y, z))
            print(f"Samples: {len(samples)}", end="\r")
            if sample_limit and len(samples) >= sample_limit:
                break
            time.sleep(delay)
    except KeyboardInterrupt:
        print("\nKalibrierung abgebrochen.")

    if not samples:
        print("Keine Daten gesammelt.")
        return

    xs, ys, zs = zip(*samples)
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    min_z, max_z = min(zs), max(zs)  # nur Vollständigkeit

    # Hard-Iron Offsets
    offset_x = (max_x + min_x) / 2.0
    offset_y = (max_y + min_y) / 2.0
    offset_z = 0.0  # ignorieren

    # Radien & Soft-Iron Skalen
    radius_x = (max_x - min_x) / 2.0
    radius_y = (max_y - min_y) / 2.0
    avg_r    = (radius_x + radius_y) / 2.0
    scale_x = avg_r / radius_x if radius_x else 1.0
    scale_y = avg_r / radius_y if radius_y else 1.0
    scale_z = 1.0

    data = {
        "offsets": [offset_x, offset_y, offset_z],
        "scales":  [scale_x,  scale_y,  scale_z],
        "min":     [min_x,     min_y,     min_z],
        "max":     [max_x,     max_y,     max_z],
    }

    with open("compass_calibration.json", "w") as f:
        json.dump(data, f, indent=4)
    print(f"\nKalibriert: Offsets={data['offsets']}, Scales={data['scales']}")

def load_compass_calibration(filename="compass_calibration.json"):
    """Lädt Offsets und Scales einmalig und cached sie global."""
    global Gcalibration_cache
    if Gcalibration_cache is None:
        try:
            with open(filename, "r") as f:
                d = json.load(f)
            Gcalibration_cache = (d["offsets"], d["scales"])
            print("Calibration data loaded.")
        except Exception as e:
            print("Error loading calibration:", e)
            Gcalibration_cache = ([0,0,0], [1,1,1])
    return Gcalibration_cache

def calibrate_magnetometer_data(raw, offsets, scales):
    """
    Wendet (raw - offset) * scale an.  <-- Wichtig: Multiplikation!
    """
    return tuple((r - o) * s for r, o, s in zip(raw, offsets, scales))

def tilt_compensate(x, y, z, pitch_rad, roll_rad):
    """
    Standard-Tilt-Compensation (NED-Konvention).
    """
    Xh = x * math.cos(pitch_rad) + z * math.sin(pitch_rad)
    Yh = (x * math.sin(roll_rad) * math.sin(pitch_rad)
         + y * math.cos(roll_rad)
         - z * math.sin(roll_rad) * math.cos(pitch_rad))
    return Xh, Yh

def vector_2_degrees(x, y):
    """(X,Y) → 0–360°"""
    hdg = math.atan2(y, x)
    return (math.degrees(hdg) + 360) % 360

def get_magnetometer_heading():
    offsets, scales = load_compass_calibration()
    raw_x, raw_y, raw_z = qmc.magnetic
    # 1) Offset + Scale
    mx = (raw_x - offsets[0]) * scales[0]
    my = (raw_y - offsets[1]) * scales[1]
    # 2) keine Tilt-Compensation
    xh, yh = mx, my
    # 3) Heading auspfiffen (je nach Achsenlage Variante wählen)
    heading_raw = math.atan2(yh, xh)
    heading_cal = (math.degrees(heading_raw) + 360) % 360
    # 4) Nullpunkt-Offset (z.B. 12° Nordabweichung)
    H0 = 12.0  # ersetzen durch deinen Messwert für echte Nord-Ausrichtung
    return (heading_cal - H0 + 360) % 360


if __name__ == "__main__":
    # Schritt 1: Kalibrierung durchführen (einmalig!)
    # calibrate_2d_magnetometer(sample_limit=1000, delay=0.05)

    # Schritt 2: Endlosschleife mit Heading und Debug-Ausgabe
    while True:
        hdg = get_magnetometer_heading(debug=True)
        # hier ggf. Deklination hinzufügen:
        # hdg_true = (hdg - 2.0) % 360
        time.sleep(0.5)
