import requests
import numpy as np
import matplotlib.pyplot as plt

locations_list = [
    "48.164214,24.536044", "48.164983,24.534836", "48.165605,24.534068",
    "48.166228,24.532915", "48.166777,24.531927", "48.167326,24.530884",
    "48.167011,24.530061", "48.166053,24.528039", "48.166655,24.526064",
    "48.166497,24.523574", "48.166128,24.520214", "48.165416,24.517170",
    "48.164546,24.514640", "48.163412,24.512980", "48.162331,24.511715",
    "48.162015,24.509462", "48.162147,24.506932", "48.161751,24.504244",
    "48.161197,24.501793", "48.160580,24.500537", "48.160250,24.500106"
]

def get_elevation_data(locations):
    url = "https://api.open-elevation.com/api/v1/lookup?locations=" + "|".join(locations)
    try:
        print("Запит до API...")
        response = requests.get(url, timeout=10)
        data = response.json()
        return data["results"]
    except Exception:
        print("Помилка API. Використовую заглушку.")
        return [{'latitude': 0, 'longitude': 0, 'elevation': 1000 + i * 10} for i in range(len(locations))]

results = get_elevation_data(locations_list)

def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

lats = [p['latitude'] for p in results]
lons = [p['longitude'] for p in results]
elevations = [p['elevation'] for p in results]

distances = [0.0]
for i in range(1, len(results)):
    d = haversine(lats[i - 1], lons[i - 1], lats[i], lons[i])
    distances.append(distances[-1] + d)

def compute_splines(x_nodes, y_nodes):
    n = len(x_nodes) - 1
    h = np.diff(x_nodes)
    a = y_nodes[:-1]

    alpha = np.zeros(n)
    beta = np.zeros(n)

    for i in range(1, n):
        A_i = h[i - 1]
        C_diag = 2 * (h[i - 1] + h[i])
        B_i = h[i]
        F_i = 3 * ((y_nodes[i + 1] - y_nodes[i]) / h[i] - (y_nodes[i] - y_nodes[i - 1]) / h[i - 1])

        denom = A_i * alpha[i - 1] + C_diag
        alpha[i] = -B_i / denom
        beta[i] = (F_i - A_i * beta[i - 1]) / denom

    c = np.zeros(n + 1)
    c[n] = 0
    for i in range(n - 1, 0, -1):
        c[i] = alpha[i] * c[i + 1] + beta[i]

    b = np.zeros(n)
    d = np.zeros(n)
    for i in range(n):
        d[i] = (c[i + 1] - c[i]) / (3 * h[i])
        b[i] = (y_nodes[i + 1] - y_nodes[i]) / h[i] - (h[i] * (c[i + 1] + 2 * c[i])) / 3

    return a, b, c[:-1], d

def print_coefficients(x_full, y_full):
    a, b, c, d = compute_splines(np.array(x_full), np.array(y_full))
    print("\n--- Коефіцієнти сплайнів (для повного набору) ---")
    print(f"{'i':<3} | {'a':<10} | {'b':<10} | {'c':<10} | {'d':<10}")
    print("-" * 55)
    for i in range(len(a)):
        print(f"{i:<3} | {a[i]:<10.2f} | {b[i]:<10.2f} | {c[i]:<10.4f} | {d[i]:<10.6f}")

def plot_interpolation(x_full, y_full, num_nodes, color):
    indices = np.linspace(0, len(x_full) - 1, num_nodes, dtype=int)
    x_nodes = np.array(x_full)[indices]
    y_nodes = np.array(y_full)[indices]

    a, b, c, d = compute_splines(x_nodes, y_nodes)

    x_smooth = []
    y_smooth = []

    for i in range(len(x_nodes) - 1):
        xs = np.linspace(x_nodes[i], x_nodes[i + 1], 20)
        for val in xs:
            dx = val - x_nodes[i]
            S_val = a[i] + b[i] * dx + c[i] * (dx ** 2) + d[i] * (dx ** 3)
            x_smooth.append(val)
            y_smooth.append(S_val)

    plt.plot(x_smooth, y_smooth, label=f'Сплайн ({num_nodes} вузлів)', linewidth=2, color=color, alpha=0.8)

    if num_nodes == 10:
        plt.scatter(x_nodes, y_nodes, s=40, marker='x', color=color, zorder=5)

print_coefficients(distances, elevations)

plt.figure(figsize=(12, 8))
plt.plot(distances, elevations, 'o--', color='black', alpha=0.3, label='Еталонні дані (Всі точки)', zorder=0)

plot_interpolation(distances, elevations, 10, 'red')
plot_interpolation(distances, elevations, 15, 'green')
plot_interpolation(distances, elevations, 20, 'blue')

plt.title("Інтерполяція профілю Говерли")
plt.xlabel("Відстань (м)")
plt.ylabel("Висота (м)")
plt.legend()
plt.grid(True)
print("\nГрафік виведено на екран.")
plt.show()

total_ascent = sum(max(elevations[i] - elevations[i - 1], 0) for i in range(1, len(elevations)))
grads = np.gradient(elevations, distances) * 100
energy_kcal = (80 * 9.81 * total_ascent) / 4184

print(f"\n--- Підсумки ---")
print(f"Набір висоти: {total_ascent:.2f} м")
print(f"Макс. крутизна: {np.max(grads):.2f}%")
print(f"Витрачена енергія (80кг): {energy_kcal:.2f} ккал")

def calculate_spline_value(x_val, x_nodes, a, b, c, d):
    for i in range(len(x_nodes) - 1):
        if x_nodes[i] <= x_val <= x_nodes[i + 1]:
            dx = x_val - x_nodes[i]
            return a[i] + b[i] * dx + c[i] * (dx ** 2) + d[i] * (dx ** 3)
    return a[-1]

def plot_error(x_full, y_full, num_nodes, color):
    indices = np.linspace(0, len(x_full) - 1, num_nodes, dtype=int)
    x_nodes = np.array(x_full)[indices]
    y_nodes = np.array(y_full)[indices]

    a, b, c, d = compute_splines(x_nodes, y_nodes)

    errors = []
    for i in range(len(x_full)):
        y_approx = calculate_spline_value(x_full[i], x_nodes, a, b, c, d)
        error = abs(y_full[i] - y_approx)
        errors.append(error)

    plt.plot(x_full, errors, label=f'Похибка ({num_nodes} вузлів)', color=color, marker='o', markersize=4)

plt.figure(figsize=(12, 6))
plot_error(distances, elevations, 10, 'red')
plot_error(distances, elevations, 15, 'green')
plot_error(distances, elevations, 20, 'blue')

plt.title("Графік абсолютної похибки інтерполяції ε = |y - S(x)|")
plt.xlabel("Відстань (м)")
plt.ylabel("Похибка (м)")
plt.legend()
plt.grid(True)
plt.show()