#!/usr/bin/env python3
import time, json, math, board, qmc5883l as qmc5883
import numpy as np

# I²C und Sensor
i2c = board.I2C()
qmc = qmc5883.QMC5883L(i2c)
qmc.output_data_rate = qmc5883.OUTPUT_DATA_RATE_200

def calibrate_2d_ellipse(sample_limit=2000, delay=0.02):
    """
    1) Sammle sample_limit X/Y-Paare bei konstanter 360°-Drehung.
    2) Fitte Ellipse A x² + B xy + C y² + D x + E y + F = 0.
    3) Leite Offsets (cx,cy) und Skalen so ab, dass die Ellipse zur Einheitsscheibe wird.
    4) Speichere offsets und scales in JSON.
    """
    data = []
    print("Drehe 360° um Z – sammle Daten…")
    for i in range(sample_limit):
        x, y, _ = qmc.magnetic
        data.append((x, y))
        time.sleep(delay)
    data = np.array(data)
    x = data[:,0]; y = data[:,1]

    # --------------- Ellipse-Fit (Least Squares) ---------------
    # Wir lösen [x², x*y, y², x, y, 1] · p = 0 mit Constraint p[5] = -1
    D = np.vstack([ x*x, x*y, y*y, x, y, np.ones_like(x) ]).T
    # Wir minimieren ||D·p|| unter p[5] = -1 → p = argmin ||D[:,0:5]·q - D[:,5]||, q = p[0:5]
    # Also: D5 = D[:,5], M = D[:,0:5]
    M = D[:,:5]; D5 = D[:,5]
    q, *_ = np.linalg.lstsq(M, -D5, rcond=None)
    A, B, C, D_, E_ = q
    F_ = 1.0

    # --------------- Ellipsenparameter in Offset/Rotation/Skala ---------------
    # Berechne Ellipsenzentrum:
    denom = B*B - 4*A*C
    cx = (2*C*D_ - B*E_) / denom
    cy = (2*A*E_ - B*D_) / denom

    # Translation der Punkte ins Zentrum
    xt = x - cx; yt = y - cy
    # Rotationswinkel der Ellipsen-Hauptachse
    theta = 0.5 * math.atan2(B, A - C)

    # Rotationsmatrix
    c0, s0 = math.cos(theta), math.sin(theta)
    R = np.array([[c0, s0],[-s0, c0]])

    # Koordinaten in Rotated Frame
    rot = R.dot(np.vstack([xt, yt]))
    xr, yr = rot[0], rot[1]

    # Halbachsenlängen (Ellipse-Radien)
    # Aus Ellipsengleichung: (xr/a)² + (yr/b)² = 1 → a = sqrt(λ1), b = sqrt(λ2)
    # λi sind Eigenwerte von Transformationsmatrix
    # Hier vereinfacht: suche max |xr|, max |yr|
    a = np.max(np.abs(xr))
    b = np.max(np.abs(yr))

    # Soft-Iron Skalen auf Kreis
    scale_x = 1.0 / a
    scale_y = 1.0 / b

    # Speichere JSON
    calib = {
        "offsets": [cx, cy, 0.0],
        "scales":  [scale_x, scale_y, 1.0],
        "ellipse": {"center": [cx, cy], "theta": theta, "axes": [a, b]}
    }
    with open("compass_calibration.json", "w") as f:
        json.dump(calib, f, indent=4)
    print("Ellipse-Kalibrierung fertig:", calib)

if __name__ == "__main__":
    calibrate_2d_ellipse()
