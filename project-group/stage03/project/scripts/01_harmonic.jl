# # Верификация гармонической цепочки
# 
# Проверка:
# 1. Сохранение полной энергии
# 2. Частота колебаний соответствует теоретической
# 3. Отсутствие перекрёстных мод

using DrWatson
@quickactivate "project"

include(srcdir("Chain.jl"))
using .Chain

using Plots, DataFrames, CSV

# Параметры

N = 31                    # число частиц
mode = 1                  # возбуждаемая мода
amplitude = 0.01          # малая амплитуда
tmax = 100.0              # время симуляции
dt = 0.01                 # шаг интегрирования

# Параметры для гармонического случая
k = 1.0
m = 1.0
alpha = 0.0               # ангармонизм выключен

println("="^60)
println("ВЕРИФИКАЦИЯ ГАРМОНИЧЕСКОЙ ЦЕПОЧКИ")
println("="^60)
println("Параметры: N=$N, mode=$mode, A=$amplitude, tmax=$tmax, dt=$dt")
println()

# Симуляция

println("Запуск симуляции...")
times, y_history, v_history = Chain.simulate(
    N, mode, amplitude, tmax, dt,
    Chain.compute_accelerations_harmonic, k, m
)

println("Готово. Шагов: $(length(times))")
println()

# 1. Сохранение полной энергии

println("1. Проверка сохранения энергии...")

# Кинетическая энергия
E_kin = zeros(length(times))
for step in 1:length(times)
    E_kin[step] = 0.5 * m * sum(v_history[:, step].^2)
end

# Потенциальная энергия
E_pot = zeros(length(times))
for step in 1:length(times)
    y = y_history[:, step]
    pot = 0.0
    # Энергия левой пружины (стенка - первая частица)
    pot += 0.5 * k * (y[1] - 0)^2
    # Энергия правой пружины (N-я частица - стенка)
    pot += 0.5 * k * (0 - y[N])^2
    # Энергия внутренних пружин
    for i in 1:N-1
        pot += 0.5 * k * (y[i+1] - y[i])^2
    end
    E_pot[step] = pot
end

E_total = E_kin + E_pot

# Относительное отклонение
E_initial = E_total[1]
E_rel_error = abs.(E_total .- E_initial) / E_initial

println("   Начальная энергия: $(round(E_initial, digits=6))")
println("   Макс. отклонение: $(round(maximum(E_rel_error)*100, digits=8))%")
println("   Отклонение в конце: $(round(E_rel_error[end]*100, digits=8))%")
println()

# График сохранения энергии
p_energy = plot(times, [E_kin E_pot E_total],
    label=["Kinetic" "Potential" "Total"],
    xlabel="Time", ylabel="Energy",
    title="Energy conservation (harmonic chain)",
    linewidth=2)
savefig(plotsdir("harmonic_energy.png"))

# 2. Проверка частоты колебаний

println("2. Проверка частоты колебаний...")

# Берём центральную частицу
center = div(N, 2) + 1
y_center = y_history[center, :]

# Находим периоды
peaks = []
for step in 2:length(times)-1
    if y_center[step] > y_center[step-1] && y_center[step] > y_center[step+1]
        push!(peaks, (times[step], y_center[step]))
    end
end

if length(peaks) >= 2
    T_meas = peaks[2][1] - peaks[1][1]
else
    T_meas = NaN
end

# Теоретическая частота
omega_theor = compute_frequencies(N; k=k, m=m)
omega_meas = 2 * pi / T_meas

println("   Теоретическая частота для моды $mode: $(round(omega_theor[mode], digits=6))")
println("   Измеренная частота (центральная частица): $(round(omega_meas, digits=6))")
if !isnan(omega_meas)
    println("   Отклонение: $(round(abs(omega_meas - omega_theor[mode])/omega_theor[mode]*100, digits=6))%")
end
println()

# График траектории
p_traj = plot_trajectory(y_history, times, center,
    title="Particle $center trajectory (mode $mode)")
savefig(plotsdir("harmonic_trajectory.png"))

# 3. Проверка отсутствия перекрёстных мод

println("3. Анализ энергий мод...")

omega = compute_frequencies(N; k=k, m=m)

# Энергии мод на каждом шаге
E_modes = zeros(N, length(times))
for step in 1:length(times)
    E_modes[:, step] = compute_mode_energies(
        y_history[:, step], v_history[:, step], omega
    )
end

# Нормируем на энергию первой моды в начальный момент
E1_initial = E_modes[1, 1]
E_modes_norm = E_modes ./ E1_initial

println("   Энергия первой моды в начале: $(round(E1_initial, digits=6))")
println()
println("   Энергии высших мод (нормированные на E1):")
for l in 2:5
    max_E = maximum(E_modes_norm[l, :])
    println("     Мода $l: макс. = $(round(max_E, digits=10))")
end
println()

# График энергий мод
p_modes = plot_energies(E_modes_norm, times, omega, N_modes=5)
savefig(plotsdir("harmonic_mode_energies.png"))

# Сохранение результатов

df = DataFrame(
    time = times,
    E_kin = E_kin,
    E_pot = E_pot,
    E_total = E_total,
    y_center = y_center
)
CSV.write(datadir("harmonic_results.csv"), df)

# Сохраняем энергии мод
df_modes = DataFrame(time = times)
for l in 1:5
    df_modes[!, "mode_$l"] = E_modes[l, :]
end
CSV.write(datadir("harmonic_modes.csv"), df_modes)

println("="^60)
println("РЕЗУЛЬТАТЫ СОХРАНЕНЫ")
println("="^60)
println("Графики: plots/harmonic_*.png")
println("Данные:  data/harmonic_results.csv")
println("         data/harmonic_modes.csv")
