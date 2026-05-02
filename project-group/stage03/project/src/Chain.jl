# # Модуль Chain.jl
# 
# Модуль для моделирования одномерной цепочки связанных частиц.
# Реализует гармонический и ангармонический (FPU) случаи.

module Chain

using Plots
using DataFrames
using LinearAlgebra

# Экспортируемые функции
export init_positions, init_velocities
export compute_accelerations_harmonic, compute_accelerations_anharmonic
export velocity_verlet, simulate
export compute_mode_energies, compute_frequencies
export plot_trajectory, plot_energies

# 1. Инициализация системы

"""
    init_positions(N, mode, amplitude; d=1.0)

Инициализирует смещения частиц в виде стоячей волны (синус).
"""
function init_positions(N, mode, amplitude; d=1.0)
    L = (N + 1) * d
    y = zeros(N)
    for i in 1:N
        x = i * d
        y[i] = amplitude * sin(mode * π * x / L)
    end
    return y
end

"""
    init_velocities(N)

Инициализирует нулевые скорости.
"""
function init_velocities(N)
    return zeros(N)
end

# 2. Расчёт ускорений

"""
    compute_accelerations_harmonic(y; k=1.0, m=1.0)

Вычисляет ускорения для гармонической цепочки.
Сила F_i = k * (y_{i+1} - 2y_i + y_{i-1})
"""
function compute_accelerations_harmonic(y, k=1.0, m=1.0)
    N = length(y)
    a = zeros(N)
    
    for i in 1:N
        # Левая граница (i=1) — стенка: y0 = 0
        left = i > 1 ? y[i-1] : 0.0
        # Правая граница (i=N) — стенка: y_{N+1} = 0
        right = i < N ? y[i+1] : 0.0
        
        a[i] = (k / m) * (right - 2*y[i] + left)
    end
    
    return a
end

"""
    compute_accelerations_anharmonic(y; k=1.0, m=1.0, alpha=0.25, d=1.0)

Вычисляет ускорения для ангармонической цепочки (FPU).
Сила F = -k * x * (1 - alpha * x / d)
"""
function compute_accelerations_anharmonic(y, k=1.0, m=1.0, alpha=0.25, d=1.0)
    N = length(y)
    a = zeros(N)
    
    for i in 1:N
        # Деформация левой пружины
        left_delta = i > 1 ? y[i] - y[i-1] : y[i]
        # Деформация правой пружины
        right_delta = i < N ? y[i+1] - y[i] : -y[i]
        
        # Сила левой пружины (с ангармонической поправкой)
        F_left = k * left_delta
        if abs(left_delta) < d  # чтобы избежать проблем при больших деформациях
            F_left = F_left * (1 - alpha * left_delta / d)
        end
        
        # Сила правой пружины (с ангармонической поправкой)
        F_right = k * right_delta
        if abs(right_delta) < d
            F_right = F_right * (1 - alpha * right_delta / d)
        end
        
        # Суммарная сила (со знаком)
        F_total = -F_left + F_right
        
        a[i] = F_total / m
    end
    
    return a
end

# 3. Численное интегрирование (скоростной метод Верле)

"""
    velocity_verlet(y, v, a, dt, compute_accel_func, params...)

Один шаг метода Верле.
"""
function velocity_verlet(y, v, a, dt, compute_accel_func, params...)
    N = length(y)
    
    # 1. Обновление позиций
    y_new = y + v * dt + 0.5 * a * dt^2
    
    # 2. Вычисление новых ускорений
    a_new = compute_accel_func(y_new, params...)
    
    # 3. Обновление скоростей
    v_new = v + 0.5 * (a + a_new) * dt
    
    return y_new, v_new, a_new
end

"""
    simulate(N, mode, amplitude, tmax, dt, compute_accel_func, params...)

Полная симуляция.
"""
function simulate(N, mode, amplitude, tmax, dt, compute_accel_func, params...)
    # Инициализация
    y = init_positions(N, mode, amplitude)
    v = init_velocities(N)
    a = compute_accel_func(y, params...)
    
    # Массивы для сохранения результатов
    times = 0.0:dt:tmax
    n_steps = length(times)
    y_history = zeros(N, n_steps)
    v_history = zeros(N, n_steps)
    
    y_history[:, 1] = y
    v_history[:, 1] = v
    
    # Основной цикл
    for step in 2:n_steps
        y, v, a = velocity_verlet(y, v, a, dt, compute_accel_func, params...)
        y_history[:, step] = y
        v_history[:, step] = v
    end
    
    return times, y_history, v_history
end

# 4. Спектральный анализ (ДПФ)

"""
    compute_frequencies(N; d=1.0, k=1.0, m=1.0)

Вычисляет теоретические частоты для гармонической цепочки.
"""
function compute_frequencies(N; d=1.0, k=1.0, m=1.0)
    ω = zeros(N)
    ω0 = sqrt(k / m)
    for l in 1:N
        ω[l] = 2ω0 * sin(l * π / (2 * (N + 1)))
    end
    return ω
end

"""
    compute_mode_energies(y, v, ω)

Вычисляет энергию каждой моды по смещениям и скоростям.
"""
function compute_mode_energies(y, v, ω)
    N = length(y)
    E = zeros(N)
    d = 1.0
    L = (N + 1) * d
    
    for l in 1:N
        # Коэффициент b_l (синусное разложение)
        b = 0.0
        db_dt = 0.0
        
        for j in 1:N
            x = j * d
            phase = l * π * x / L
            b += y[j] * sin(phase)
            db_dt += v[j] * sin(phase)
        end
        
        b = sqrt(2 / (N + 1)) * b
        db_dt = sqrt(2 / (N + 1)) * db_dt
        
        # Энергия моды (как у гармонического осциллятора)
        E[l] = 0.5 * (db_dt^2 + ω[l]^2 * b^2)
    end
    
    return E
end

# 5. Визуализация

"""
    plot_trajectory(y_history, times, i; title="")

График смещения частицы i от времени.
"""
function plot_trajectory(y_history, times, i; title="")
    p = plot(times, y_history[i, :], 
             xlabel="Time", ylabel="Displacement",
             title=isempty(title) ? "Particle $i trajectory" : title,
             linewidth=2, legend=false)
    return p
end

"""
    plot_energies(E_history, times, ω; N_modes=5)

График энергий первых N_modes мод.
"""
function plot_energies(E_history, times, ω; N_modes=5)
    p = plot()
    for l in 1:min(N_modes, length(ω))
        plot!(p, times, E_history[l, :], 
              label="Mode $l (ω=$(round(ω[l], digits=4)))",
              linewidth=1.5)
    end
    plot!(p, xlabel="Time", ylabel="Energy", title="Mode energies", legend=:topright)
    return p
end

end # module Chain
