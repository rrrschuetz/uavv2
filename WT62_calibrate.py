#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Kompasskalibrierung für WT62

Dieses Skript liest fortlaufend Magnetometer-Daten vom WT62 (erwartet 11-Byte-Pakete, wobei
das erste Byte 0x55 und das zweite Byte 0x59 ist) und berechnet für jede Achse den minimalen
und maximalen Messwert. Daraus werden Offset- und Skalierungswerte ermittelt und in der
Datei "compass_calibration.json" abgelegt.

Zum Starten des Kalibrierungsvorgangs:
    1. Starte das Skript.
    2. Drehe den Sensor in alle Richtungen, sodass möglichst viele Messwerte erfasst werden.
    3. Drücke Strg+C, um die Kalibrierung zu beenden.
"""

import serial
import time
import json

# Konfiguriere hier deinen seriellen Port, Baudrate und Timeout
SERIAL_PORT = "/dev/ttyUSB0"  # passe diesen Wert an dein System an (z.B. "COM3" unter Windows)
BAUD_RATE = 115200
TIMEOUT = 1


def parse_wt61_data(packet):
    """
    Parst ein 11-Byte-Paket vom WT62.
    Erwartet:
      - Byte 0: Startbyte (0x55)
      - Byte 1: Datentyp (hier 0x59 für Magnetometer)
      - Bytes 2-7: Rohdaten für X, Y, Z (je 2 Byte, little-endian, vorzeichenbehaftet)
      - Die restlichen Bytes werden hier ignoriert.
    """
    data_type = packet[1]
    if data_type == 0x59:
        # Extrahiere je 2 Byte für X, Y, Z und interpretiere sie als vorzeichenbehaftete Integer
        x = int.from_bytes(packet[2:4], byteorder='little', signed=True)
        y = int.from_bytes(packet[4:6], byteorder='little', signed=True)
        z = int.from_bytes(packet[6:8], byteorder='little', signed=True)
        return data_type, (x, y, z)
    return data_type, (0, 0, 0)


def main():
    print("Starte Kompasskalibrierung. Bitte drehen Sie den Sensor in alle Richtungen.")
    print("Drücken Sie Strg+C, um die Kalibrierung abzuschließen und die Daten zu speichern.\n")

    # Initialisiere Listen für Minimum und Maximum der drei Achsen
    mag_min = [None, None, None]
    mag_max = [None, None, None]

    buff = bytearray()  # Buffer für eingehende serielle Daten

    try:
        with serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=TIMEOUT) as ser:
            ser.reset_input_buffer()  # Alte Daten verwerfen
            while True:
                if ser.in_waiting:
                    data = ser.read(ser.in_waiting)
                    buff.extend(data)

                    # Solange genügend Bytes für ein komplettes Paket vorhanden sind:
                    while len(buff) >= 11:
                        # Überprüfe, ob ein gültiges Paket vorliegt (Startbyte 0x55 und Datentyp 0x59)
                        if buff[0] == 0x55 and buff[1] == 0x59:
                            packet = buff[:11]
                            buff = buff[11:]  # Entferne das verarbeitete Paket aus dem Buffer

                            # Parse das Magnetometer-Paket
                            _, (mx, my, mz) = parse_wt61_data(packet)

                            # Aktualisiere Min- und Max-Werte für jede Achse
                            for i, val in enumerate((mx, my, mz)):
                                if mag_min[i] is None or val < mag_min[i]:
                                    mag_min[i] = val
                                if mag_max[i] is None or val > mag_max[i]:
                                    mag_max[i] = val

                            print(f"Magnetometer Rohwerte: X={mx}, Y={my}, Z={mz}")
                        else:
                            # Entferne ein Byte, falls kein gültiger Start gefunden wurde
                            buff.pop(0)
                else:
                    time.sleep(0.01)
    except KeyboardInterrupt:
        print("\nKalibrierung beendet. Berechne Kalibrierungsdaten...")

    # Berechne Offset und Skalierung für jede Achse
    offsets = []
    scales = []
    for i in range(3):
        if mag_min[i] is None or mag_max[i] is None:
            offsets.append(0)
            scales.append(1)
        else:
            # Offset: Mittelwert aus min und max
            offset = (mag_max[i] + mag_min[i]) / 2.0
            # Skalenfaktor: Halbe Spannweite (dies kann später zur Normalisierung genutzt werden)
            scale = (mag_max[i] - mag_min[i]) / 2.0
            offsets.append(offset)
            scales.append(scale)

    calibration_data = {
        "offsets": offsets,
        "scales": scales,
        "min": mag_min,
        "max": mag_max
    }

    # Speichere die Kalibrierungsdaten in einer JSON-Datei
    calibration_filename = "compass_calibration.json"
    try:
        with open(calibration_filename, "w") as f:
            json.dump(calibration_data, f, indent=4)
        print(f"\nKalibrierungsdaten wurden in '{calibration_filename}' gespeichert.")
        print("Berechnete Offsets:", offsets)
        print("Berechnete Skalenfaktoren:", scales)
    except Exception as e:
        print("Fehler beim Speichern der Kalibrierungsdaten:", e)


if __name__ == "__main__":
    main()
