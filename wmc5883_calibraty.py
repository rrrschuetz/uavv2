import math
import time
import json

# Annahme: qmc5883 ist bereits importiert und initialisiert.
import qmc5883


# Hier stehen auch deine existierenden Funktionen:
def vector_2_degrees(x, y):
    angle = math.degrees(math.atan2(y, x))
    if angle < 0:
        angle += 360
    return angle


def tilt_compensate(mag_x, mag_y, mag_z, pitch, roll):
    mag_x_comp = mag_x * math.cos(pitch) + mag_z * math.sin(pitch)
    mag_y_comp = (mag_x * math.sin(roll) * math.sin(pitch)
                  + mag_y * math.cos(roll)
                  - mag_z * math.sin(roll) * math.cos(pitch))
    return mag_x_comp, mag_y_comp


# Funktionen zum Laden der Kalibrierungsdaten und Anwenden der Korrektur
def load_compass_calibration(filename="compass_calibration.json"):
    """
    Lädt Kalibrierungsdaten aus der JSON-Datei.
    Liefert:
        offsets: Liste mit Offsets für X, Y, Z
        scales: Liste mit Skalenfaktoren für X, Y, Z
    """
    try:
        with open(filename, "r") as f:
            data = json.load(f)
        offsets = data.get("offsets", [0, 0, 0])
        scales = data.get("scales", [1, 1, 1])
        return offsets, scales
    except Exception as e:
        print(f"Fehler beim Laden der Kalibrierungsdaten: {e}")
        return [0, 0, 0], [1, 1, 1]


def calibrate_magnetometer_data(raw_data, offsets, scales):
    """
    Wendet die Kalibrierungsdaten auf die Rohwerte an.

    Args:
        raw_data: Tupel (x, y, z) mit Rohwerten.
        offsets: Liste der Offsets.
        scales: Liste der Skalenfaktoren.

    Returns:
        Tupel mit kalibrierten Werten.
    """
    calibrated = []
    for raw, offset, scale in zip(raw_data, offsets, scales):
        if scale != 0:
            calibrated_val = (raw - offset) / scale
        else:
            calibrated_val = raw - offset
        calibrated.append(calibrated_val)
    return tuple(calibrated)


# Beispiel: Integration in die Funktion zur Bestimmung des Magnetometer-Headings
# Hier wird angenommen, dass Gpitch und Groll (in Grad) global definiert sind.
Gpitch = 0.0  # Beispielwert, ersetze durch deine tatsächlichen Daten
Groll = 0.0  # Beispielwert


def get_magnetometer_heading():
    retries = 10  # Anzahl der Wiederholungsversuche
    # Kalibrierungsdaten laden
    offsets, scales = load_compass_calibration("compass_calibration.json")

    for attempt in range(retries):
        try:
            # Lese Rohwerte vom Magnetometer
            raw_data = qmc5883.magnetic  # Liefert (mag_x, mag_y, mag_z)
            # Wende die Kalibrierung an
            mag_x, mag_y, mag_z = calibrate_magnetometer_data(raw_data, offsets, scales)
            # Optional: Hier kannst du die kalibrierten Werte ausgeben
            # print(f"Kalibrierte Rohwerte: X={mag_x}, Y={mag_y}, Z={mag_z}")

            # Führe Neigungskompensation anhand von Pitch und Roll durch
            mag_x_comp, mag_y_comp = tilt_compensate(
                mag_x, mag_y, mag_z,
                math.radians(Gpitch), math.radians(Groll)
            )
            # Berechne den Kompasswinkel aus den kompensierten Werten
            mag_heading = vector_2_degrees(mag_x_comp, mag_y_comp)
            return mag_heading
        except OSError as e:
            # Fehler beim Lesen – kurze Wartezeit und erneuter Versuch
            time.sleep(0.5)
    return 0


# Beispielhafter Aufruf
if __name__ == "__main__":
    heading = get_magnetometer_heading()
    print(f"Magnetometer Heading: {heading:.2f}°")
