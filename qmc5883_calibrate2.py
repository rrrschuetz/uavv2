import time
import json
import board
import qmc5883l as qmc5883

# initialize I²C and sensor
i2c = board.I2C()
qmc = qmc5883.QMC5883L(i2c)
qmc.output_data_rate = qmc5883.OUTPUT_DATA_RATE_200

def calibrate_magnetometer(sample_limit=None, delay=0.05):
    """
    Kalibriert den QMC5883L:
    - Sammle Roh-Magnetometerdaten in einer Liste, bis sample_limit erreicht ist
      oder der Benutzer STRG+C drückt.
    - Berechne die Min/Max für jede Achse, daraus Hard-Iron-Offsets.
    - Berechne die “Radii” und normiere so, dass alle Achsen den gleichen Radius haben
      (Soft-Iron-Korrektur).
    - Speichere offsets und scale-Faktoren in JSON.
    """
    print("Starte Magnetometer-Kalibrierung.")
    print("Dreh den Sensor langsam in alle Richtungen.")
    print("Drücke STRG+C oder warte auf sample limit, um zu beenden.\n")

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
        print("\nKalibrierung manuell abgebrochen.")

    if not samples:
        print("Keine Daten gesammelt – Kalibrierung abgebrochen.")
        return

    # Transponieren und Min/Max berechnen
    xs, ys, zs = map(list, zip(*samples))
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    min_z, max_z = min(zs), max(zs)

    # Hard-iron Offsets
    offset_x = (max_x + min_x) / 2.0
    offset_y = (max_y + min_y) / 2.0
    offset_z = (max_z + min_z) / 2.0

    # “Radii” je Achse und mittlerer Radius
    radius_x = (max_x - min_x) / 2.0
    radius_y = (max_y - min_y) / 2.0
    radius_z = (max_z - min_z) / 2.0
    avg_radius = (radius_x + radius_y + radius_z) / 3.0

    # Soft-iron scale-Faktoren
    scale_x = avg_radius / radius_x if radius_x else 1.0
    scale_y = avg_radius / radius_y if radius_y else 1.0
    scale_z = avg_radius / radius_z if radius_z else 1.0

    calibration_data = {
        "offsets": [offset_x, offset_y, offset_z],
        "scales": [scale_x, scale_y, scale_z],
        "min": [min_x, min_y, min_z],
        "max": [max_x, max_y, max_z]
    }

    # JSON speichern
    filename = "compass_calibration.json"
    try:
        with open(filename, "w") as f:
            json.dump(calibration_data, f, indent=4)
        print(f"\nKalibrierungsdaten in '{filename}' gespeichert.")
        print(f"Offsets: {[offset_x, offset_y, offset_z]}")
        print(f"Scale-Faktoren: {[scale_x, scale_y, scale_z]}")
    except Exception as e:
        print(f"Fehler beim Speichern: {e}")

if __name__ == "__main__":
    # z.B. max. 2000 Samples oder bis STRG+C
    calibrate_magnetometer(sample_limit=2000, delay=0.05)
