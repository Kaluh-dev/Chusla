import requests
import numpy as np
import matplotlib.pyplot as plt


# --- 1. ОТРИМАННЯ ДАНИХ (Як вказано в методичці) ---
def get_data():
    # Координати з PDF (сторінка 5)
    locs = "48.164214,24.536044|48.164983,24.534836|48.165605,24.534068|" \
           "48.166228,24.532915|48.166777,24.531927|48.167326,24.530884|" \
           "48.167011,24.530061|48.166053,24.528039|48.166655,24.526064|" \
           "48.166497,24.523574|48.166128,24.520214|48.165416,24.517170|" \
           "48.164546,24.514640|48.163412,24.512980|48.162331,24.511715|" \
           "48.162015,24.509462|48.162147,24.506932|48.161751,24.504244|" \
           "48.161197,24.501793|48.160580,24.500537|48.160250,24.500106"

    url = f"https://api.open-elevation.com/api/v1/lookup?locations={locs}"
    print("Завантаження даних висот...")
    try:
        data = requests.get(url).json()['results']
        return data
    except:
        print("Помилка API! Перевірте інтернет.")
        return []


# Формула відстані (Haversine) з методички (стор. 6)
def get_dist_elev(results):
    R = 6371000
    dists = [0]
    elevs = [results[0]['elevation']]

    for i in range(1, len(results)):
        lat1, lon1 = np.radians(results[i - 1]['latitude']), np.radians(results[i - 1]['longitude'])
        lat2, lon2 = np.radians(results[i]['latitude']), np.radians(results[i]['longitude'])

        dphi = lat2 - lat1
        dlam = lon2 - lon1
        a = np.sin(dphi / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlam / 2) ** 2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        d = R * c

        dists.append(dists[-1] + d)
        elevs.append(results[i]['elevation'])

    return np.array(dists), np.array(elevs)


# --- 2. МАТЕМАТИКА СПЛАЙНІВ (Метод прогонки) ---
class Spline:
    def __init__(self, x, y):
        self.x = np.array(x)
        self.y = np.array(y)
        self.n = len(x)
        self.h = np.diff(self.x)
        self.a = self.y[:-1]  # Коефіцієнт a (стор. 2)
        self.b, self.c, self.d = np.zeros(self.n), np.zeros(self.n), np.zeros(self.n)
        self.solve()

    def solve(self):
        # Метод прогонки (Thomas Algorithm) - стор. 3-4
        n = self.n - 1
        alpha = np.zeros(n)
        beta = np.zeros(n)

        # Прямий хід
        A = self.h[1:-1]
        B = 2 * (self.h[:-1] + self.h[1:])
        C = self.h[1:]
        F = 3 * ((self.y[2:] - self.y[1:-1]) / self.h[1:] - (self.y[1:-1] - self.y[:-2]) / self.h[:-1])

        # c[0] = 0 (вільний сплайн)
        curr_alpha = -C[0] / B[0]
        curr_beta = F[0] / B[0]
        alpha[0], beta[0] = curr_alpha, curr_beta

        for i in range(1, len(B)):
            denom = B[i] + A[i - 1] * curr_alpha
            curr_alpha = -C[i] / denom if i < len(C) else 0
            curr_beta = (F[i] - A[i - 1] * curr_beta) / denom
            alpha[i], beta[i] = curr_alpha, curr_beta

        # Зворотний хід (знаходимо c)
        c_res = np.zeros(n + 1)
        c_res[-1] = 0
        c_res[1:-1][::-1]

        for i in range(len(B) - 1, -1, -1):
            c_res[i + 1] = alpha[i] * c_res[i + 2] + beta[i]

        self.c = c_res

        # Знаходимо d та b (стор. 2)
        for i in range(n):
            self.d[i] = (self.c[i + 1] - self.c[i]) / (3 * self.h[i])
            self.b[i] = (self.y[i + 1] - self.y[i]) / self.h[i] - self.h[i] * (self.c[i + 1] + 2 * self.c[i]) / 3

    def calc(self, x_val):
        for i in range(self.n - 1):
            if self.x[i] <= x_val <= self.x[i + 1]:
                dx = x_val - self.x[i]
                return self.a[i] + self.b[i] * dx + self.c[i] * dx ** 2 + self.d[i] * dx ** 3
        return self.y[-1]


# --- 3. ЗАПУСК ---
data = get_data()
if not data: exit()

x, y = get_dist_elev(data)
spline = Spline(x, y)

# Вивід коефіцієнтів у консоль (Завдання 8-9)
print(f"\n{'i':<3} | {'a':<10} | {'b':<10} | {'c':<10} | {'d':<10}")
print("-" * 50)
for i in range(len(x) - 1):
    print(f"{i:<3} | {spline.a[i]:<10.2f} | {spline.b[i]:<10.2f} | {spline.c[i]:<10.4f} | {spline.d[i]:<10.6f}")

# Графік
x_new = np.linspace(x[0], x[-1], 500)
y_new = [spline.calc(xi) for xi in x_new]

plt.figure(figsize=(10, 6))
plt.plot(x, y, 'ro', label='GPS Точки')
plt.plot(x_new, y_new, 'b-', label='Кубічний сплайн')
plt.title("Профіль висоти: Заросляк - Говерла")
plt.xlabel("Відстань (м)")
plt.ylabel("Висота (м)")
plt.legend()
plt.grid(True)
plt.show()