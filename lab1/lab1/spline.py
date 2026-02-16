import numpy as np


class CubicSpline:
    def __init__(self, x, y):
        self.x = np.array(x)
        self.y = np.array(y)
        self.n = len(x)

        # Коефіцієнти сплайна: a, b, c, d
        self.a = np.zeros(self.n)
        self.b = np.zeros(self.n)
        self.c = np.zeros(self.n)
        self.d = np.zeros(self.n)
        self.h = np.diff(self.x)  # крок сітки h_i [cite: 16]

        self.build_spline()

    def solve_tridiagonal_system(self):
        """
        Розв'язує систему лінійних рівнянь методом прогонки (Thomas algorithm).
        Відповідає пункту 7 [cite: 141] та теорії [cite: 50-73].
        """
        n = self.n
        # Матриця системи має розмір (n-2)x(n-2) для c_1...c_{n-1}
        # (в індексації 0..n-1 це c[1]...c[n-2], оскільки c[0]=0 і c[n-1]=0 для вільного сплайна)

        alpha = np.zeros(n)
        beta = np.zeros(n)

        # Прямий хід прогонки [cite: 53-64]
        # Будуємо систему для c_i.
        # Рівняння: h_{i-1}c_{i-1} + 2(h_{i-1}+h_i)c_i + h_i c_{i+1} = ... [cite: 43]

        # Для зручності реалізації в Python використовуємо тимчасові масиви A, B, C, F для методу прогонки
        # A[i]*x[i-1] + B[i]*x[i] + C[i]*x[i+1] = F[i]

        size = n - 2  # Кількість невідомих c (від c[1] до c[n-2])
        if size <= 0: return

        # Ініціалізація прогоночних коефіцієнтів
        p_alpha = np.zeros(size)
        p_beta = np.zeros(size)

        # Формування правих частин рівнянь [cite: 43]
        def rhs(i):
            return 3 * ((self.y[i + 1] - self.y[i]) / self.h[i] - (self.y[i] - self.y[i - 1]) / self.h[i - 1])

        # Перший крок прямого ходу (для i=1 в глобальному, 0 в локальному)
        # 2(h0 + h1)c1 + h1c2 = rhs(1)
        A0 = 0
        B0 = 2 * (self.h[0] + self.h[1])
        C0 = self.h[1]
        F0 = rhs(1)

        p_alpha[0] = -C0 / B0
        p_beta[0] = F0 / B0

        # Основний цикл прямого ходу
        for i in range(1, size):
            global_i = i + 1
            A = self.h[global_i - 1]
            B = 2 * (self.h[global_i - 1] + self.h[global_i])
            C = self.h[global_i]
            F = rhs(global_i)

            denom = B + A * p_alpha[i - 1]
            if i < size - 1:
                p_alpha[i] = -C / denom
            p_beta[i] = (F - A * p_beta[i - 1]) / denom

        # Зворотний хід прогонки [cite: 66-69]
        self.c[size] = p_beta[size - 1]  # Це c[n-2]
        for i in range(size - 2, -1, -1):
            self.c[i + 1] = p_alpha[i] * self.c[i + 2] + p_beta[i]

        # Граничні умови c[0] = 0, c[n-1] = 0 (вільний сплайн) [cite: 30, 45]
        self.c[0] = 0
        self.c[-1] = 0

    def build_spline(self):
        """Обчислює всі коефіцієнти сплайна a, b, c, d."""
        # 1. Знаходимо a_i = y_i (у PDF a_i = y_{i-1}, але для зручності індексації беремо y_i як початок інтервалу)
        # В коді: S_i(x) визначено на [x_i, x_{i+1}].
        # Відповідно до [cite: 36] a_i = y_i (якщо індексувати з 0)
        self.a = self.y[:-1]

        # 2. Знаходимо c_i методом прогонки
        self.solve_tridiagonal_system()

        # 3. Знаходимо d_i та b_i [cite: 37, 38]
        # Зауваження: у PDF формули прив'язані до x_{i-1}. Тут адаптовано до стандартної 0-based індексації Python
        # S_i(x) = a_i + b_i(x-x_i) + c_i(x-x_i)^2 + d_i(x-x_i)^3

        for i in range(self.n - 1):
            # d_i = (c_{i+1} - c_i) / 3h_i [cite: 37]
            self.d[i] = (self.c[i + 1] - self.c[i]) / (3 * self.h[i])

            # b_i = (y_{i+1} - y_i)/h_i - h_i(c_{i+1} + 2c_i)/3 [cite: 38]
            self.b[i] = (self.y[i + 1] - self.y[i]) / self.h[i] - (self.h[i] * (self.c[i + 1] + 2 * self.c[i])) / 3

    def interpolate(self, x_val):
        """Повертає значення сплайна в точці x_val."""
        # Знайти відповідний інтервал
        if x_val < self.x[0] or x_val > self.x[-1]:
            return None  # За межами діапазону

        # Знаходимо індекс i, такий що x[i] <= x_val < x[i+1]
        for i in range(self.n - 1):
            if self.x[i] <= x_val <= self.x[i + 1]:
                dx = x_val - self.x[i]
                # Формула сплайна [cite: 11]
                return self.a[i] + self.b[i] * dx + self.c[i] * (dx ** 2) + self.d[i] * (dx ** 3)
        return self.y[-1]

    def print_coefficients(self):
        """Виводить коефіцієнти у консоль [cite: 141-144]."""
        print("\nКоефіцієнти сплайнів:")
        print(f"{'i':<3} | {'a':<10} | {'b':<10} | {'c':<10} | {'d':<10}")
        print("-" * 50)
        for i in range(self.n - 1):
            print(f"{i:<3} | {self.a[i]:<10.4f} | {self.b[i]:<10.4f} | {self.c[i]:<10.4f} | {self.d[i]:<10.6f}")