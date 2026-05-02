# # Ангармоническая цепочка (задача Ферми–Пасты–Улама)
# 
# Исследование перераспределения энергии между модами
# при включении нелинейности.

using DrWatson
@quickactivate "project"

include(srcdir("Chain.jl"))
using .Chain

using Plots, DataFrames, CSV

# ============================================================ #
# Параметры
# ============================================================ #

N = 31                    # число частиц
mode = 1                  # возбуждаемая мода
amplitude = 0.5           # бОльшая амплитуда (для ангармонизма)
tmax = 500.0              # большое время для наблюдения перераспределения
dt = 0.01                 # шаг интегрирования

# Параметры ангармонической цепочки
k = 1.0
m = 1.0
d = 1.0
alpha = 0.25              # параметр ангармонизма (FPU)

println("="^60)
println("ЗАДАЧА ФЕРМИ-ПАСТЫ-УЛАМА (FPU)")
println("="^60)
println("Параметры: N=$N, mode=$mode, A=$amplitude")
println("           tmax=$tmax, dt=$dt, alpha=$alpha")
println()

# ============================================================ #
# Симуляция
# ============================================================ #

println("Запуск симуляции...")
times, y_history, v_history = simulate(
    N, mode, amplitude, tmax, dt,
    compute_accelerations_anharmonic, k, m, alpha, d
)
println("Готово. Шагов: $(length(times))")
println()

# ============================================================ #
# Полная энергия (для контроля)
# ============================================================ #

println("1. Сохранение полной энергии...")

# Кинетическая энергия
E_kin = zeros(length(times))
for step in 1:length(times)
    E_kin[step] = 0.5 * m * sum(v_history[:, step].^2)
end

# Потенциальная энергия (с ангармонической поправкой)
E_pot = zeros(length(times))
for step in 1:length(times)
    y = y_history[:, step]
    pot = 0.0
    
    # Левая пружина (стенка - первая частица)
    delta = y[1]
    pot += 0.5 * k * delta^2 - (k * alpha / (3*d)) * delta^3
    
    # Правая пружина (N-я частица - стенка)
    delta = -y[N]
    pot += 0.5 * k * delta^2 - (k * alpha / (3*d)) * delta^3
    
    # Внутренние пружины
    for i in 1:N-1
        delta = y[i+1] - y[i]
        pot += 0.5 * k * delta^2 - (k * alpha / (3*d)) * delta^3
    end
    
    E_pot[step] = pot
end

E_total = E_kin + E_pot

# Относительное отклонение
E_initial = E_total[1]
E_rel_error = abs.(E_total .- E_initial) / E_initial

println("   Начальная энергия: $(round(E_initial, digits=6))")
println("   Макс. отклонение: $(round(maximum(E_rel_error)*100, digits=6))%")
println()

# График сохранения энергии
p_energy = plot(times, [E_kin E_pot E_total],
    label=["Kinetic" "Potential" "Total"],
    xlabel="Time", ylabel="Energy",
    title="Energy conservation (FPU chain, α=$alpha)",
    linewidth=2)
savefig(plotsdir("fpu_energy.png"))

# ============================================================ #
# Анализ энергий мод
# ============================================================ #

println("2. Анализ перераспределения энергии между модами...")

ω = compute_frequencies(N; k=k, m=m)

# Энергии мод на каждом шаге
E_modes = zeros(N, length(times))
for step in 1:length(times)
    E_modes[:, step] = compute_mode_energies(
        y_history[:, step], v_history[:, step], ω
    )
end

# Нормируем на полную энергию (или на энергию первой моды в начале)
E_total_sys = sum(E_modes[:, 1])  # полная энергия по суммам мод
E_modes_norm = E_modes ./ E_total_sys

println("   Полная энергия системы (сумма по модам): $(round(E_total_sys, digits=6))")
println()
println("   Максимальные доли энергии в модах:")
for l in 1:5
    max_frac = maximum(E_modes_norm[l, :])
    println("     Мода $l: макс. = $(round(max_frac*100, digits=4))%")
end
println()

# График эволюции энергий мод
p_modes = plot()
for l in 1:5
    plot!(p_modes, times, E_modes_norm[l, :],
          label="Mode $l (ω=$(round(ω[l], digits=4)))",
          linewidth=1.5)
end
plot!(p_modes, xlabel="Time", ylabel="Energy fraction",
      title="FPU: Mode energy redistribution (α=$alpha)",
      legend=:topright)
savefig(plotsdir("fpu_mode_energies.png"))

# ============================================================ #
# Траектории нескольких частиц
# ============================================================ #

println("3. Визуализация траекторий...")

# Центральная частица
center = div(N, 2) + 1
p_center = plot(times, y_history[center, :],
    xlabel="Time", ylabel="Displacement",
    title="Particle $center trajectory (FPU, α=$alpha)",
    linewidth=1.5, legend=false)
savefig(plotsdir("fpu_trajectory_center.png"))

# Несколько частиц
p_several = plot()
for i in [1, div(N,4)+1, center, div(3N,4)+1, N]
    plot!(p_several, times, y_history[i, :],
          label="Particle $i", linewidth=1.5)
end
plot!(p_several, xlabel="Time", ylabel="Displacement",
      title="FPU: Trajectories of several particles",
      legend=:topright)
savefig(plotsdir("fpu_trajectories.png"))

# ============================================================ #
# Зависимость от α (параметрическое исследование)
# ============================================================ #

println("4. Параметрическое исследование (влияние α)...")

alphas = [0.05, 0.1, 0.25, 0.5]
alpha_results = []

for α_test in alphas
    println("   α = $α_test...")
    
    _, y_hist, v_hist = simulate(
        N, mode, amplitude, 200.0, dt,
        compute_accelerations_anharmonic, k, m, α_test, d
    )
    
    # Энергии мод в конце
    ω_local = compute_frequencies(N; k=k, m=m)
    E_modes_local = compute_mode_energies(y_hist[:, end], v_hist[:, end], ω_local)
    E_total_local = sum(E_modes_local)
    
    push!(alpha_results, (α=α_test, E_mode1=E_modes_local[1]/E_total_local))
end

# График зависимости от α
p_alpha = plot([r.α for r in alpha_results], [r.E_mode1 for r in alpha_results],
    marker=:circle, linewidth=2, markersize=6,
    xlabel="α (anharmonicity parameter)",
    ylabel="Fraction of energy in mode 1 (final)",
    title="FPU: Energy remaining in mode 1 vs α")
savefig(plotsdir("fpu_vs_alpha.png"))

# ============================================================ #
# Сохранение результатов
# ============================================================ #

# Основной DataFrame
df = DataFrame(
    time = times,
    E_kin = E_kin,
    E_pot = E_pot,
    E_total = E_total,
    y_center = y_history[center, :]
)
CSV.write(datadir("fpu_results.csv"), df)

# Энергии мод
df_modes = DataFrame(time = times)
for l in 1:5
    df_modes[!, "mode_$l"] = E_modes_norm[l, :]
end
CSV.write(datadir("fpu_modes.csv"), df_modes)

# Параметрическое исследование
df_alpha = DataFrame(α=[r.α for r in alpha_results], 
                     E_mode1_fraction=[r.E_mode1 for r in alpha_results])
CSV.write(datadir("fpu_alpha_scan.csv"), df_alpha)

println("="^60)
println("РЕЗУЛЬТАТЫ СОХРАНЕНЫ")
println("="^60)
println("Графики: plots/fpu_*.png")
println("Данные:  data/fpu_results.csv")
println("         data/fpu_modes.csv")
println("         data/fpu_alpha_scan.csv")
