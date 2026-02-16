import numpy as np
import matplotlib.pyplot as plt
from utils import get_elevation_data, process_coordinates
from spline import CubicSpline


def analyze_route(distances, elevations, spline_x, spline_y):
    """Виконує аналіз маршруту (додаткове завдання)."""
    print("\n--- Аналіз маршруту ---")

    # 1. Загальна довжина
    total_dist = distances[-1]
    print(f"Загальна довжина маршруту: {total_dist:.2f} м ")

    # 2. Набір/спуск (початкові вузли)
    total_ascent = sum(max(elevations[i] - elevations[i - 1], 0) for i in range(1, len(elevations)))
    total_descent = sum(max(elevations[i - 1] - elevations[i], 0) for i in range(1, len(elevations)))
    print(f"Сумарний набір висоти: {total_ascent:.2f} м ")
    print(f"Сумарний спуск: {total_descent:.2f} м ")

    # 3. Градієнт (на згладженому сплайні)
    grads = np.gradient(spline_y, spline_x) * 100  # у відсотках
    print(f"Макс. підйом: {np.max(grads):.2f}%")
    print(f"Макс. спуск: {np.min(grads):.2f}%")
    print(f"Середній градієнт: {np.mean(np.abs(grads)):.2f}% [cite: 169]")

    # 4. Енергія
    mass = 80  # кг
    g = 9.81
    energy_j = mass * g * total_ascent
    energy_kcal = energy_j / 4184
    print(f"Механічна робота: {energy_j / 1000:.2f} кДж")
    print(f"Енергія: {energy_kcal:.2f} ккал [cite: 180]")


def main():
    # 1. Отримання даних
    print("Отримання даних з Open-Elevation API...")
    results = get_elevation_data()

    if not results:
        print("Не вдалося отримати дані.")
        return

    # 2. Табуляція [cite: 104-114]
    print(f"Кількість отриманих вузлів: {len(results)}")

    # Обробка координат (переведення в метри)
    x_full, y_full = process_coordinates(results)

    # Виведення таблиці [cite: 136]
    print("\nТабуляція (Відстань, Висота):")
    for d, h in zip(x_full, y_full):
        print(f"{d:10.2f} | {h:8.2f}")

    # 3. Дослідження впливу кількості вузлів [cite: 145-146]
    # Набір точок має 21 точку. Спробуємо взяти підмножини.
    # (Лабораторна просить 10, 15, 20).
    subsets = [10, 15, 20]

    plt.figure(figsize=(12, 8))

    # Малюємо оригінальні точки
    plt.scatter(x_full, y_full, color='red', label='Original GPS Nodes', zorder=5)

    colors = ['green', 'blue', 'orange']

    for i, num_nodes in enumerate(subsets):
        if num_nodes > len(x_full):
            print(f"\nУвага: Запитано {num_nodes} вузлів, але доступно лише {len(x_full)}.")
            continue

        # Вибираємо рівномірно розподілені індекси
        indices = np.linspace(0, len(x_full) - 1, num_nodes, dtype=int)
        x_subset = x_full[indices]
        y_subset = y_full[indices]

        print(f"\n--- Розрахунок сплайна для {num_nodes} вузлів ---")
        spline = CubicSpline(x_subset, y_subset)

        # Виведення коефіцієнтів для першого варіанту (наприклад)
        if num_nodes == 20:
            spline.print_coefficients()

        # Генерація точок для плавного графіка
        x_smooth = np.linspace(x_subset[0], x_subset[-1], 500)
        y_smooth = [spline.interpolate(val) for val in x_smooth]

        plt.plot(x_smooth, y_smooth, label=f'Spline ({num_nodes} nodes)', color=colors[i], linestyle='--')

        # Для повного набору (20 вузлів) робимо аналіз
        if num_nodes == 20:
            analyze_route(x_full, y_full, x_smooth, y_smooth)

    plt.title("Інтерполяція профілю висоти (Заросляк - Говерла)")
    plt.xlabel("Кумулятивна відстань (м)")
    plt.ylabel("Висота (м)")
    plt.legend()
    plt.grid(True)
    plt.savefig('elevation_profile.png')
    print("\nГрафік збережено у файл 'elevation_profile.png'")
    plt.show()


if __name__ == "__main__":
    main()