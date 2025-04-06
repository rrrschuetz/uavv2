#!/usr/bin/env python3
import time
import json
import math
import board
import qmc5883l as qmc5883

i2c = board.I2C()
qmc = qmc5883.QMC5883L(i2c)
qmc.output_data_rate = (qmc5883.OUTPUT_DATA_RATE_200)

def calibrate_magnetometer():
    """
    Kalibriert den QMC5883, indem für jede Achse die minimalen und maximalen
    Rohwerte erfasst werden. Mit STRG+C wird die Kalibrierung beendet und die
    Daten (Offsets und Skalen) werden in einer JSON-Datei gespeichert.
    """
    print("Starte Magnetometer-Kalibrierung.")
    print("Drehe den Sensor in alle Richtungen.")
    print("Drücke STRG+C, um die Kalibrierung zu beenden und die Daten zu speichern.\n")

    # Initialisiere die Minimal- und Maximalwerte für X, Y, Z als None
    mag_min = [None, None, None]
    mag_max = [None, None, None]

    try:
        while True:
            # Lese die Rohwerte vom Magnetometer
            mag_x, mag_y, mag_z = qmc5883.magnetic  # Ersetze diesen Aufruf falls anders
            # Aktualisiere die Min-/Max-Werte für jede Achse
            for i, val in enumerate((mag_x, mag_y, mag_z)):
                if mag_min[i] is None or val < mag_min[i]:
                    mag_min[i] = val
                if mag_max[i] is None or val > mag_max[i]:
                    mag_max[i] = val
            # Gib aktuelle Werte aus (diese Ausgabe aktualisiert sich in derselben Zeile)
            print(f"Raw: X={mag_x}, Y={mag_y}, Z={mag_z} | Min: {mag_min} | Max: {mag_max}", end="\r")
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nKalibrierung beendet. Berechne Kalibrierungsdaten...")

    # Berechne Offset und Skalenfaktor für jede Achse:
    # Offset = (min + max) / 2  -> Hard-Iron-Korrektur
    # Skalenfaktor = (max - min) / 2  -> Normierung des Messbereichs
    offsets = []
    scales = []
    for i in range(3):
        if mag_min[i] is None or mag_max[i] is None:
            offsets.append(0)
            scales.append(1)
        else:
            offset = (mag_max[i] + mag_min[i]) / 2.0
            scale = (mag_max[i] - mag_min[i]) / 2.0
            offsets.append(offset)
            scales.append(scale)

    calibration_data = {
        "offsets": offsets,
        "scales": scales,
        "min": mag_min,
        "max": mag_max
    }

    filename = "compass_calibration.json"
    try:
        with open(filename, "w") as f:
            json.dump(calibration_data, f, indent=4)
        print(f"\nKalibrierungsdaten wurden in '{filename}' gespeichert.")
        print(f"Offsets: {offsets}")
        print(f"Skalierungsfaktoren: {scales}")
    except Exception as e:
        print(f"Fehler beim Speichern der Kalibrierungsdaten: {e}")


if __name__ == "__main__":
    calibrate_magnetometer()
