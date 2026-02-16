import requests
import numpy as np


def get_elevation_data():
    """
    Отримує дані висот для фіксованого маршруту (Заросляк - Говерла)
    використовуючи Open-Elevation API, як вказано в лабораторній[cite: 92].
    """
    # Координати з PDF [cite: 93, 101-102]
    locations = (
        "48.164214,24.536044|48.164983,24.534836|48.165605,24.534068|"
        "48.166228,24.532915|48.166777,24.531927|48.167326,24.530884|"
        "48.167011,24.530061|48.166053,24.528039|48.166655,24.526064|"
        "48.166497,24.523574|48.166128,24.520214|48.165416,24.517170|"
        "48.164546,24.514640|48.163412,24.512980|48.162331,24.511715|"
        "48.162015,24.509462|48.162147,24.506932|48.161751,24.504244|"
        "48.161197,24.501793|48.160580,24.500537|48.160250,24.500106"
    )

    url = f"https://api.open-elevation.com/api/v1/lookup?locations={locations}"

    try:
        response = requests.get(url)
        data = response.json()
        return data['results']
    except Exception as e:
        print(f"Помилка API: {e}")
        # Повертаємо пустий список або тестові дані, якщо API не відповідає
        return []


def haversine(lat1, lon1, lat2, lon2):
    """
    Обчислює відстань між двома точками за координатами (формула гаверсинуса).
    Реалізація згідно [cite: 121-129].
    """
    R = 6371000  # Радіус Землі в метрах

    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)

    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    # [cite: 129] return 2*R*arctan...
    return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def process_coordinates(results):
    """Обробляє 'results' з API і повертає масиви відстаней (X) та висот (Y)."""
    n = len(results)
    coords = [(p["latitude"], p["longitude"]) for p in results]
    elevations = [p["elevation"] for p in results]

    distances = [0.0]
    for i in range(1, n):
        d = haversine(*coords[i - 1], *coords[i])
        distances.append(distances[-1] + d)  # Кумулятивна відстань [cite: 133]

    return np.array(distances), np.array(elevations)