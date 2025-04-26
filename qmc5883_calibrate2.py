#!/usr/bin/env python3
"""
2D-Magnetometer-Kalibrierung für ein waagerecht bleibendes, autonomes Fahrzeug.

Sammelt Magnetometer-Rohdaten, berechnet Offsets und Skalen (nur X/Y),
und speichert die Kalibrierungsdaten in compass_calibration.json.
"""

import time
import json
import board
import qmc5883l as qmc5883

from math import atan2, degrees

# --- Sensor initialisieren ---
i2c = board.I2C()
qmc = qmc5883.QMC5883L(i2c)
qmc.output_data_rate = qmc5883.OUTPUT_DATA_RATE_200

def calibrate_2d_magnetometer(sample_limit=None, delay=0.05):
    """
    Kalibriert den Magnetometer auf X/Y-Ebene:
      - sample_limit: maximale Anzahl Samples (None = bis STRG+C)
      - delay: Pause zwischen Messungen in Sekunden
    """
    print("Starte 2D-Magnetometer-Kalibrierung (nur X/Y).")
    print("Drehe das Fahrzeug ruhig 360° um die Hochachse (Yaw).")
    print("Drücke STRG+C, um zu beenden.\n")

    samples = []
    try:
        while True:
            x_raw, y_raw, z_raw = qmc.magnetic
            samples.append((x_raw, y_raw, z_raw))
            print(f"Samples gesammelt: {len(samples)}", end="\r")

            if sample_limit and len(samples) >= sample_limit:
                break
            time.sleep(delay)
    except KeyboardInterrupt:
        print("\nKalibrierung abgebrochen.")

    if not samples:
        print("Keine Daten gesammelt. Beende.")
        return

    # Daten trennen
    xs, ys, zs = zip(*samples)

    # Min/Max X/Y
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    # Min/Max Z (nur zur Vollständigkeit)
    min_z, max_z = min(zs), max(zs)

    # Offsets (Hard-Iron)
    offset_x = (max_x + min_x) / 2.0
    offset_y = (max_y + min_y) / 2.0
    # Z-Achse ignorieren (Offset=0)
    offset_z = 0.0

    # Radien X/Y
    radius_x = (max_x - min_x) / 2.0
    radius_y = (max_y - min_y) / 2.0
    avg_radius = (radius_x + radius_y) / 2.0

    # Skalenfaktoren X/Y (Soft-Iron)
    scale_x = avg_radius / radius_x if radius_x else 1.0
    scale_y = avg_radius / radius_y if radius_y else 1.0
    # Z-Achse unverändert lassen
    scale_z = 1.0

    # Kalibrierungsdaten zusammenstellen
    calibration_data = {
        "offsets": [offset_x, offset_y, offset_z],
        "scales":  [scale_x,  scale_y,  scale_z],
        "min":     [min_x,     min_y,     min_z],
        "max":     [max_x,     max_y,     max_z],
    }

    # Als JSON speichern
    filename = "compass_calibration.json"
    try:
        with open(filename, "w") as f:
            json.dump(calibration_data, f, indent=4)
        print(f"\nKalibrierung abgeschlossen. Daten in '{filename}' gespeichert.")
        print(f"Offsets: {calibration_data['offsets']}")
        print(f"Skalen:  {calibration_data['scales']}")
    except Exception as e:
        print(f"Fehler beim Speichern: {e}")

if __name__ == "__main__":
    # Beispiel: bis 1000 Samples oder bis STRG+C
    calibrate_2d_magnetometer(sample_limit=1000, delay=0.05)
