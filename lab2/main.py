import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ─────────────────────────────────────────────
# 1. зчитування даних з CSV
# ─────────────────────────────────────────────

def read_data(filename):
    x, y = [], []
    with open(filename, 'r', newline='') as file:
        reader = csv.DictReader(file)
        for row in reader:
            x.append(float(row['n']))
            y.append(float(row['t']))
    return x, y


# ─────────────────────────────────────────────
# 2. функції інтерполяції Ньютона
# ─────────────────────────────────────────────

def omega(x_val, nodes):
    """ω_k(x) = ∏(x - x_i) для i = 0..k-1"""
    result = 1.0
    for xi in nodes:
        result *= (x_val - xi)
    return result


def divided_diff(x_nodes, f_nodes):
    """Таблиця розділених різниць. dd[i][j] = f[x_i,...,x_{i+j}]"""
    n = len(x_nodes)
    dd = [[0.0] * n for _ in range(n)]
    for i in range(n):
        dd[i][0] = f_nodes[i]
    for j in range(1, n):
        for i in range(n - j):
            dd[i][j] = (dd[i+1][j-1] - dd[i][j-1]) / (x_nodes[i+j] - x_nodes[i])
    return dd


def newton_value(x_val, x_nodes, dd):
    """N_n(x) = f_0 + Σ dd[0][k] * ω_k(x)"""
    n = len(x_nodes)
    result = dd[0][0]
    for k in range(1, n):
        result += dd[0][k] * omega(x_val, x_nodes[:k])
    return result


def newton_array(x_arr, x_nodes, dd):
    """Обчислення N(x) для масиву значень"""
    return np.array([newton_value(xi, x_nodes, dd) for xi in x_arr])


# ─────────────────────────────────────────────
# 3. метод Лагранжа
# ─────────────────────────────────────────────

def lagrange_value(x_val, x_nodes, f_nodes):
    """L_n(x) = Σ f_i * ∏_{j≠i} (x-x_j)/(x_i-x_j)"""
    n = len(x_nodes)
    result = 0.0
    for i in range(n):
        term = f_nodes[i]
        for j in range(n):
            if j != i:
                term *= (x_val - x_nodes[j]) / (x_nodes[i] - x_nodes[j])
        result += term
    return result


def lagrange_array(x_arr, x_nodes, f_nodes):
    return np.array([lagrange_value(xi, x_nodes, f_nodes) for xi in x_arr])


# ─────────────────────────────────────────────
# 4. табуляція
# ─────────────────────────────────────────────

def tabulate(a, b, n, x_nodes, dd):
    """Табуляція на [a,b] з кроком h=(b-a)/(20n)."""
    h = (b - a) / (20 * n)
    xs = np.arange(a, b + h * 0.5, h)
    N_vals  = newton_array(xs, x_nodes, dd)
    om_vals = np.array([omega(xi, x_nodes) for xi in xs])
    return xs, N_vals, om_vals


# ─────────────────────────────────────────────
# 5. друк таблиці розділених різниць
# ─────────────────────────────────────────────

def print_dd_table(x_nodes, f_nodes, dd):
    n = len(x_nodes)
    print("\n" + "="*72)
    print("ТАБЛИЦЯ РОЗДІЛЕНИХ РІЗНИЦЬ")
    print("="*72)
    header = f"{'i':>3} {'x_i':>7} {'f[x_i]':>10}"
    for j in range(1, n):
        header += f"  {'порядок '+str(j):>14}"
    print(header)
    print("-"*72)
    for i in range(n):
        row = f"{i:>3} {x_nodes[i]:>7.0f} {dd[i][0]:>10.4f}"
        for j in range(1, n - i):
            row += f"  {dd[i][j]:>14.8e}"
        print(row)
    print("="*72)


# ─────────────────────────────────────────────
# 6. головна програма
# ─────────────────────────────────────────────

def main():

    # Вхідні дані (Варіант 1)
    x_all = [1000.0, 2000.0, 4000.0, 8000.0, 16000.0]
    t_all = [3.0,    5.0,    11.0,   28.0,   85.0]

    # Запис CSV
    data_file = "data.csv"
    with open(data_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['n', 't'])
        writer.writeheader()
        for xi, ti in zip(x_all, t_all):
            writer.writerow({'n': xi, 't': ti})

    # Зчитування
    x_nodes, f_nodes = read_data(data_file)
    n = len(x_nodes)

    print("="*72)
    print("  ЛАБОРАТОРНА РОБОТА — ІНТЕРПОЛЯЦІЯ МНОГОЧЛЕНОМ НЬЮТОНА")
    print("  Варіант 1: Прогнозування часу виконання алгоритму")
    print("="*72)
    print("\nВузли інтерполяції:")
    for xi, fi in zip(x_nodes, f_nodes):
        print(f"  n={xi:.0f}  ->  t={fi} мс")

    # Таблиця розділених різниць
    dd = divided_diff(x_nodes, f_nodes)
    print_dd_table(x_nodes, f_nodes, dd)

    # Прогноз для n=6000 обома методами
    n_pred = 6000.0
    t_newton  = newton_value(n_pred, x_nodes, dd)
    t_lagrange = lagrange_value(n_pred, x_nodes, f_nodes)
    print(f"\nПрогноз Ньютона  N_4({int(n_pred)}) = {t_newton:.6f} мс")
    print(f"Прогноз Лагранжа L_4({int(n_pred)}) = {t_lagrange:.6f} мс")
    print(f"Різниця між методами: {abs(t_newton - t_lagrange):.2e} мс")

    a, b = x_nodes[0], x_nodes[-1]   # 1000 .. 16000
    x_fine = np.linspace(a, b, 600)
    N_fine = newton_array(x_fine, x_nodes, dd)
    L_fine = lagrange_array(x_fine, x_nodes, f_nodes)

    # ─────────────────────────────────────────
    # ГРАФІКИ — аркуш 1: основні
    # ─────────────────────────────────────────
    fig1, axes1 = plt.subplots(2, 2, figsize=(14, 10))
    fig1.suptitle("Інтерполяційний многочлен Ньютона — основні графіки\n"
                  "Варіант 1: Прогнозування часу виконання алгоритму",
                  fontsize=12, fontweight='bold')

    # ── графік 1: N(x), L(x), точки, прогноз ──
    ax = axes1[0, 0]
    ax.plot(x_fine, N_fine, 'b-', linewidth=2.0, label='$N_4(x)$ — Ньютон')
    ax.plot(x_fine, L_fine, 'r--', linewidth=1.5, label='$L_4(x)$ — Лагранж', alpha=0.7)
    ax.scatter(x_nodes, f_nodes, color='red', s=90, zorder=5, label='Вузли')
    ax.axvline(n_pred, color='green', linestyle=':', alpha=0.8)
    ax.scatter([n_pred], [t_newton], color='green', s=140, zorder=6,
               marker='*', label=f'Прогноз n=6000: {t_newton:.2f} мс')
    ax.set_xlabel('Розмір вхідних даних n')
    ax.set_ylabel('Час виконання t (мс)')
    ax.set_title('Інтерполяція: Ньютон vs Лагранж')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # ── графік 2: похибка ε(x) = |N(x) - L(x)| (різниця методів) ──
    ax = axes1[0, 1]
    eps_methods = np.abs(N_fine - L_fine)
    ax.plot(x_fine, eps_methods, 'purple', linewidth=1.5)
    ax.set_xlabel('n')
    ax.set_ylabel('|N(x) − L(x)|')
    ax.set_title('Різниця між методами Ньютона і Лагранжа\n(ε ≈ 0 → методи еквівалентні)')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('symlog', linthresh=1e-12)

    # ── графік 3: вузловий поліном ω_n(x) ──
    ax = axes1[1, 0]
    xs_tab, N_tab, om_tab = tabulate(a, b, n, x_nodes, dd)
    ax.plot(xs_tab, om_tab, 'm-', linewidth=1.5, label='$\\omega_5(x)$')
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax.scatter(x_nodes, [0]*len(x_nodes), color='red', s=70, zorder=5,
               label='Вузли (нулі ω)')
    ax.set_xlabel('n'); ax.set_ylabel('$\\omega_5(x)$')
    ax.set_title('Вузловий поліном $\\omega_n(x)$')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # ── графік 4: похибка ε(x) на табличних точках ──
    # "справжні" значення = самі вузлові точки, перевіряємо що N(x_i)=f(x_i)
    ax = axes1[1, 1]
    # для наочності: беремо підмножини і дивимось похибку відносно повного N_4
    colors_k = ['orange', 'purple', 'green']
    for idx, k in enumerate([2, 3, 4]):
        xk = x_nodes[:k]
        fk = f_nodes[:k]
        ddk = divided_diff(xk, fk)
        N_k = newton_array(x_fine[:400], xk, ddk)
        N_ref = newton_array(x_fine[:400], x_nodes, dd)
        eps_k = np.abs(N_ref - N_k)
        ax.plot(x_fine[:400], eps_k, color=colors_k[idx], linewidth=1.4,
                label=f'N_{k-1} vs N_4, max={np.max(eps_k):.2f}')
    ax.set_xlabel('n'); ax.set_ylabel('|ε(x)|')
    ax.set_title('Похибка ε(x) при неповній кількості вузлів\n(відносно N₄)')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    fig1.tight_layout()
    fig1.savefig('lab_graphs_1_main.png', dpi=150, bbox_inches='tight')
    plt.close(fig1)

    # ─────────────────────────────────────────
    # ГРАФІКИ — аркуш 2: дослідницька частина
    # ─────────────────────────────────────────
    fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))
    fig2.suptitle("Дослідницька частина\nВаріант 1: Прогнозування часу виконання алгоритму",
                  fontsize=12, fontweight='bold')

    # ── дослідження 1а: фіксований інтервал, різна кількість вузлів n=5,10,20 ──
    ax = axes2[0, 0]
    node_counts = [5, 10, 20]
    colors_ni = ['blue', 'orange', 'green']
    x_t = np.linspace(a, b, 400)
    N_ref = newton_array(x_t, x_nodes, dd)
    for idx, ni in enumerate(node_counts):
        x_ni = list(np.linspace(a, b, ni))
        f_ni = [newton_value(xi, x_nodes, dd) for xi in x_ni]
        dd_ni = divided_diff(x_ni, f_ni)
        N_ni = newton_array(x_t, x_ni, dd_ni)
        h_i = (b - a) / (ni - 1)
        eps = np.abs(N_ref - N_ni)
        ax.plot(x_t, eps, color=colors_ni[idx], linewidth=1.5,
                label=f'n={ni}, h={h_i:.0f}, max|ε|={np.max(eps):.2e}')
    ax.set_xlabel('n'); ax.set_ylabel('|ε(x)|')
    ax.set_title('Дослідження 1: вплив кроку\n(фіксований інтервал [1000,16000])')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    ax.set_yscale('symlog', linthresh=1e-10)

    # ── дослідження 1б: фіксований крок h, змінний інтервал b=a+h*n ──
    ax = axes2[0, 1]
    h_fixed = 3000.0
    colors_n = ['blue', 'orange', 'green']
    for idx, ni in enumerate([5, 10, 20]):
        b_i = a + h_fixed * (ni - 1)
        x_ni = list(np.linspace(a, b_i, ni))
        f_ni = [newton_value(xi, x_nodes, dd) for xi in x_ni if xi <= b]
        # обрізаємо до діапазону даних
        x_ni_clip = [xi for xi in x_ni if xi <= b]
        f_ni_clip = [newton_value(xi, x_nodes, dd) for xi in x_ni_clip]
        if len(x_ni_clip) < 2:
            continue
        dd_ni = divided_diff(x_ni_clip, f_ni_clip)
        x_range = np.linspace(x_ni_clip[0], x_ni_clip[-1], 300)
        N_ni = newton_array(x_range, x_ni_clip, dd_ni)
        ax.plot(x_range, N_ni, color=colors_n[idx], linewidth=1.5,
                label=f'n={ni}, b={x_ni_clip[-1]:.0f}')
    ax.scatter(x_nodes, f_nodes, color='red', s=60, zorder=5, label='Вузли')
    ax.set_xlabel('n'); ax.set_ylabel('t (мс)')
    ax.set_title(f'Дослідження 2: змінний інтервал\n(h={h_fixed:.0f}, n=5,10,20)')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # ── дослідження 3: ефект Рунге (класична функція) ──
    ax = axes2[1, 0]
    runge_f = lambda x: 1.0 / (1.0 + x**2)
    a_r, b_r = -5.0, 5.0
    x_r = np.linspace(a_r, b_r, 500)
    ax.plot(x_r, runge_f(x_r), 'k-', linewidth=2.5, label='f(x)=1/(1+x²)', zorder=5)
    colors_r = plt.cm.plasma(np.linspace(0.1, 0.85, 5))
    for idx, ni in enumerate([3, 5, 7, 9, 11]):
        x_ri = list(np.linspace(a_r, b_r, ni))
        f_ri = [runge_f(xi) for xi in x_ri]
        dd_ri = divided_diff(x_ri, f_ri)
        N_ri = newton_array(x_r, x_ri, dd_ri)
        ax.plot(x_r, N_ri, color=colors_r[idx], linewidth=1.4, alpha=0.9,
                label=f'N_{ni-1}, n={ni}')
    ax.set_ylim(-0.5, 1.8)
    ax.set_xlabel('x'); ax.set_ylabel('y')
    ax.set_title('Аналіз ефекту Рунге\n(f(x) = 1/(1+x²))')
    ax.legend(fontsize=8, loc='upper center'); ax.grid(True, alpha=0.3)

    # ── дослідження 4: Ньютон vs Лагранж — час виконання (швидкість) ──
    import time
    ax = axes2[1, 1]
    test_sizes = list(range(3, 21))
    times_newton   = []
    times_lagrange = []
    x_test = np.linspace(a, b, 200)
    for ni in test_sizes:
        x_ni = list(np.linspace(a, b, ni))
        f_ni = [newton_value(xi, x_nodes, dd) for xi in x_ni]
        dd_ni = divided_diff(x_ni, f_ni)

        t0 = time.perf_counter()
        for _ in range(20):
            newton_array(x_test, x_ni, dd_ni)
        times_newton.append((time.perf_counter() - t0) / 20 * 1000)

        t0 = time.perf_counter()
        for _ in range(20):
            lagrange_array(x_test, x_ni, f_ni)
        times_lagrange.append((time.perf_counter() - t0) / 20 * 1000)

    ax.plot(test_sizes, times_newton,   'b-o', linewidth=1.5, markersize=5,
            label='Ньютон')
    ax.plot(test_sizes, times_lagrange, 'r-s', linewidth=1.5, markersize=5,
            label='Лагранж')
    ax.set_xlabel('Кількість вузлів n')
    ax.set_ylabel('Час обчислення (мс)')
    ax.set_title('Порівняння методів Ньютона і Лагранжа\n(час обчислення 200 точок)')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    fig2.tight_layout()
    fig2.savefig('lab_graphs_2_research.png', dpi=150, bbox_inches='tight')
    plt.close(fig2)

    print("\nГрафіки збережено:")
    print("  lab_graphs_1_main.png     — основні графіки")
    print("  lab_graphs_2_research.png — дослідницька частина")
    print(f"\nПРОГНОЗ: N_4(6000) = {t_newton:.4f} мс  |  L_4(6000) = {t_lagrange:.4f} мс")

    return t_newton


if __name__ == "__main__":
    main()