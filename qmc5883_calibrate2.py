#!/usr/bin/env python3
import time, json, math, board, qmc5883l as qmc5883

def calibrate_2d_magnetometer(sample_limit=2000, delay=0.02):
    print("Kalibrierung starten: waagerecht drehen, 360° um Z.")
    samples = []
    try:
        while len(samples) < sample_limit:
            x, y, _ = qmc.magnetic
            samples.append((x, y))
            print(f"Samples: {len(samples)}/{sample_limit}", end="\r")
            time.sleep(delay)
    except KeyboardInterrupt:
        pass

    if not samples:
        print("Keine Daten.")
        return

    xs, ys = zip(*samples)
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    # 1) Hard-Iron Offsets
    offset_x = (max_x + min_x) / 2.0
    offset_y = (max_y + min_y) / 2.0

    # 2) Punkte zentrieren
    xs1 = [x - offset_x for x in xs]
    ys1 = [y - offset_y for y in ys]

    # 3) Ellipsen-Halbachsen
    Rx = max(abs(x) for x in xs1)
    Ry = max(abs(y) for y in ys1)

    # 4) Mittlerer Radius
    Rmean = sum(math.hypot(x, y) for x, y in zip(xs1, ys1)) / len(xs1)

    # 5) Soft-Iron Skalen
    scale_x = Rmean / Rx if Rx else 1.0
    scale_y = Rmean / Ry if Ry else 1.0

    data = {
        "offsets": [offset_x, offset_y, 0.0],
        "scales":  [scale_x,  scale_y,  1.0],
        "min":     [min_x,    min_y,    None],
        "max":     [max_x,    max_y,    None],
    }

    with open("compass_calibration.json", "w") as f:
        json.dump(data, f, indent=4)
    print("\nFertig:", data)

if __name__ == "__main__":
    # Sensor initialisieren
    i2c = board.I2C()
    qmc = qmc5883.QMC5883L(i2c)
    qmc.output_data_rate = qmc5883.OUTPUT_DATA_RATE_200

    calibrate_2d_magnetometer()
