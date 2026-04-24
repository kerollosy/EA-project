import csv
import json
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from deap import base, creator, tools


F = 0.5
CR = 0.9
NUM_INTERSECTIONS = 2
MIN_GREEN = 10
MAX_GREEN = 120
SERVICE_RATE = 2
CONGESTION_WEIGHT = 120

POPULATION_SIZE = 30
SIM_HORIZON = 150
NUM_GENERATIONS = 50
NUM_RUNS = 30
OUTPUT_DIR = "de_final_outputs"


toolbox = base.Toolbox()
if not hasattr(creator, "FitnessMin"):
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
if not hasattr(creator, "Individual"):
    creator.create("Individual", np.ndarray, fitness=creator.FitnessMin)


def createIndividual():
    return creator.Individual(np.random.uniform(MIN_GREEN, MAX_GREEN, NUM_INTERSECTIONS * 2))


toolbox.register("individualCreator", createIndividual)
toolbox.register("populationCreator", tools.initRepeat, list, toolbox.individualCreator)


def simulate_traffic(time_config, traffic_stream):
    if traffic_stream is None or len(traffic_stream) == 0:
        return {
            "total_wait": float("inf"),
            "avg_queue": float("inf"),
            "objective": float("inf"),
        }

    queue_ns = [0] * NUM_INTERSECTIONS
    queue_ew = [0] * NUM_INTERSECTIONS
    total_wait = 0.0
    queue_accumulator = 0.0

    for current_time in range(len(traffic_stream)):
        arrivals_per_intersection = traffic_stream[current_time]

        for i in range(NUM_INTERSECTIONS):
            green_ns = time_config[2 * i]
            green_ew = time_config[2 * i + 1]

            if green_ns <= 0 or green_ew <= 0:
                return {
                    "total_wait": float("inf"),
                    "avg_queue": float("inf"),
                    "objective": float("inf"),
                }

            arrivals_ns, arrivals_ew = arrivals_per_intersection[i]
            queue_ns[i] += arrivals_ns
            queue_ew[i] += arrivals_ew

            total_cycle = green_ns + green_ew
            time_in_cycle = current_time % int(total_cycle)

            if time_in_cycle < green_ns:
                queue_ns[i] = max(0, queue_ns[i] - SERVICE_RATE)
            else:
                queue_ew[i] = max(0, queue_ew[i] - SERVICE_RATE)

            current_total_queue = queue_ns[i] + queue_ew[i]
            total_wait += current_total_queue
            queue_accumulator += current_total_queue

    avg_queue = queue_accumulator / (len(traffic_stream) * NUM_INTERSECTIONS)
    objective = total_wait + CONGESTION_WEIGHT * avg_queue
    return {"total_wait": float(total_wait), "avg_queue": float(avg_queue), "objective": float(objective)}


def evaluate(individual, traffic_stream):
    results = simulate_traffic(individual, traffic_stream)
    return (results["objective"],)


toolbox.register("evaluate", evaluate)


def generate_traffic_stream(sim_time):
    traffic_stream = []
    for _ in range(sim_time):
        arrivals_snapshot = []
        for _ in range(NUM_INTERSECTIONS):
            # Strictly fair demand model: NS and EW get identical arrivals.
            base_arrival = random.randint(0, 3)
            arrivals_ns = base_arrival
            arrivals_ew = base_arrival
            arrivals_snapshot.append((arrivals_ns, arrivals_ew))
        traffic_stream.append(arrivals_snapshot)
    return traffic_stream


def load_or_create_seeds(filename="seeds.json", num_runs=30):
    try:
        with open(filename, "r", encoding="utf-8") as f:
            return json.load(f)["seeds"]
    except FileNotFoundError as e:
        print(f"Warning: {filename} not found. Generating new seeds. {e}")
        seeds = [random.randint(1, 10000) for _ in range(num_runs)]

        with open(filename, "w", encoding="utf-8") as f:
            json.dump({"seeds": seeds}, f, indent=2)

        print(f"Generated and saved new {filename}")
        return seeds


def mutation(a, b, c, scale_factor, low, up):
    mutant = c + scale_factor * (b - a)
    mutant = np.clip(mutant, low, up)
    return creator.Individual(mutant)


def crossOver(target, mutant, crossover_rate):
    # Binomial crossover with forced mutant gene to avoid cloning target.
    mask = np.random.rand(len(target)) < crossover_rate
    j_rand = np.random.randint(0, len(target))
    mask[j_rand] = True
    trial = np.where(mask, mutant, target)
    return creator.Individual(trial)


def selectedIndices(pop_size, current_idx):
    indices = list(range(pop_size))
    indices.remove(current_idx)
    return random.sample(indices, 3)


def plot_optimization_results(run_histories, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid", context="talk")

    generations = np.arange(NUM_GENERATIONS)
    best_curves = np.array([run["best_curve"] for run in run_histories], dtype=float)
    avg_curves = np.array([run["avg_curve"] for run in run_histories], dtype=float)
    improvement_curves = np.array([run["improvement_curve"] for run in run_histories], dtype=float)

    mean_best = best_curves.mean(axis=0)
    std_best = best_curves.std(axis=0)
    mean_avg = avg_curves.mean(axis=0)
    mean_improvement = improvement_curves.mean(axis=0)
    std_improvement = improvement_curves.std(axis=0)

    baseline_values = np.array([run["baseline_objective"] for run in run_histories], dtype=float)
    final_best_values = np.array([run["final_best"] for run in run_histories], dtype=float)
    baseline_mean = baseline_values.mean()

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    axes[0].plot(generations, mean_best, label="Mean best-so-far objective", color="#1f77b4", linewidth=2.5)
    axes[0].fill_between(generations, mean_best - std_best, mean_best + std_best, color="#1f77b4", alpha=0.2)
    axes[0].plot(generations, mean_avg, label="Mean population objective", color="#2ca02c", linewidth=2.0, alpha=0.9)
    axes[0].axhline(baseline_mean, linestyle="--", color="#d62728", linewidth=2.0, label="Mean baseline objective")
    axes[0].set_title("DE Objective Over Generations")
    axes[0].set_xlabel("Generation")
    axes[0].set_ylabel("Objective (lower is better)")
    axes[0].legend()

    axes[1].plot(generations, mean_improvement, label="Mean improvement vs baseline", color="#ff7f0e", linewidth=2.5)
    axes[1].fill_between(
        generations,
        mean_improvement - std_improvement,
        mean_improvement + std_improvement,
        color="#ff7f0e",
        alpha=0.2,
    )
    axes[1].axhline(0.0, linestyle="--", color="black", linewidth=1.2)
    axes[1].set_title("Improvement vs Baseline")
    axes[1].set_xlabel("Generation")
    axes[1].set_ylabel("Improvement (%)")
    axes[1].legend()

    fig.tight_layout()
    progress_plot_path = os.path.join(output_dir, "de_progress_vs_baseline.png")
    fig.savefig(progress_plot_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(14, 7))
    run_indices = np.arange(1, len(run_histories) + 1)
    width = 0.42
    ax2.bar(run_indices - width / 2, baseline_values, width=width, label="Baseline objective", color="#d62728", alpha=0.85)
    ax2.bar(run_indices + width / 2, final_best_values, width=width, label="DE final best objective", color="#1f77b4", alpha=0.9)
    ax2.set_title("Per-Run Baseline vs DE Final Best")
    ax2.set_xlabel("Run index")
    ax2.set_ylabel("Objective (lower is better)")
    ax2.legend()

    fig2.tight_layout()
    comparison_plot_path = os.path.join(output_dir, "baseline_vs_de_per_run.png")
    fig2.savefig(comparison_plot_path, dpi=180, bbox_inches="tight")
    plt.close(fig2)

    return progress_plot_path, comparison_plot_path


def save_run_summaries(run_histories, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "de_run_summary.csv")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "run_index",
                "seed",
                "baseline_objective",
                "final_best",
                "improvement_percent",
                "best_timings",
            ]
        )
        for run in run_histories:
            baseline = float(run["baseline_objective"])
            final_best = float(run["final_best"])
            improvement = ((baseline - final_best) / baseline) * 100.0 if baseline != 0 else 0.0
            best_timings = "[" + ", ".join(f"{v:.2f}" for v in run["best_solution"]) + "]"
            writer.writerow(
                [
                    run["run_index"],
                    run["seed"],
                    f"{baseline:.6f}",
                    f"{final_best:.6f}",
                    f"{improvement:.4f}",
                    best_timings,
                ]
            )

    return csv_path


if __name__ == "__main__":
    seeds = load_or_create_seeds("seeds.json", num_runs=NUM_RUNS)
    run_histories = []

    for run_idx, seed_val in enumerate(seeds):
        random.seed(seed_val)
        np.random.seed(seed_val)

        population = toolbox.populationCreator(n=POPULATION_SIZE)
        traffic_stream = generate_traffic_stream(SIM_HORIZON)

        print(f"\n--- RUN {run_idx + 1} (Seed {seed_val}) ---")

        for individual in population:
            individual.fitness.values = toolbox.evaluate(individual, traffic_stream)

        best_overall = None
        for individual in population:
            if best_overall is None or individual.fitness.values[0] < best_overall.fitness.values[0]:
                best_overall = creator.Individual(individual)
                best_overall.fitness.values = individual.fitness.values

        best_curve = []
        avg_curve = []
        best_so_far = float("inf")

        for gen in range(NUM_GENERATIONS):
            for i in range(len(population)):
                a_idx, b_idx, c_idx = selectedIndices(len(population), i)
                mutant = mutation(population[a_idx], population[b_idx], population[c_idx], F, MIN_GREEN, MAX_GREEN)

                trial = crossOver(population[i], mutant, CR)
                trial[:] = np.clip(trial, MIN_GREEN, MAX_GREEN)
                trial.fitness.values = toolbox.evaluate(trial, traffic_stream)

                if trial.fitness.values[0] < population[i].fitness.values[0]:
                    population[i] = trial

                if best_overall is None or population[i].fitness.values[0] < best_overall.fitness.values[0]:
                    best_overall = creator.Individual(population[i])
                    best_overall.fitness.values = population[i].fitness.values

            generation_fitness = np.array([individual.fitness.values[0] for individual in population], dtype=float)
            generation_min = float(np.min(generation_fitness))
            generation_avg = float(np.mean(generation_fitness))
            best_so_far = min(best_so_far, generation_min)

            best_curve.append(best_so_far)
            avg_curve.append(generation_avg)

            if gen % 10 == 0 or gen == NUM_GENERATIONS - 1:
                print(
                    f"Gen {gen:02d} | Best so far: {best_so_far:.2f} | "
                    f"Generation avg: {generation_avg:.2f}"
                )

        baseline_timings = np.array([60.0, 60.0] * NUM_INTERSECTIONS, dtype=float)
        baseline_res = simulate_traffic(baseline_timings, traffic_stream)
        print(
            f"Baseline -> Wait: {baseline_res['total_wait']:.2f}, "
            f"Avg Queue: {baseline_res['avg_queue']:.2f}, "
            f"Objective: {baseline_res['objective']:.2f}"
        )

        baseline_obj = float(baseline_res["objective"])
        improvement_curve = [((baseline_obj - value) / baseline_obj) * 100.0 for value in best_curve]

        run_histories.append(
            {
                "run_index": run_idx + 1,
                "seed": seed_val,
                "best_curve": best_curve,
                "avg_curve": avg_curve,
                "improvement_curve": improvement_curve,
                "baseline_objective": baseline_obj,
                "final_best": float(best_overall.fitness.values[0]),
                "best_solution": np.array(best_overall, dtype=float).tolist(),
            }
        )

        print("-- Best Individual = ", np.round(best_overall, 2))
        print("-- Best Fitness    = ", best_overall.fitness.values[0])

    progress_plot_path, comparison_plot_path = plot_optimization_results(run_histories, OUTPUT_DIR)
    csv_summary_path = save_run_summaries(run_histories, OUTPUT_DIR)

    print(f"\nSaved plot: {progress_plot_path}")
    print(f"Saved plot: {comparison_plot_path}")
    print(f"Saved summary CSV: {csv_summary_path}")
