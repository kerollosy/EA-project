import csv
import json
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from deap import base, creator, tools

# Hyperparameters
USE_LINEAR_INERTIA = True
W_MAX = 0.9
W_MIN = 0.4
W_FIXED = 0.7

C1 = 2.0
C2 = 2.0

NUM_INTERSECTIONS = 2
MIN_GREEN = 10
MAX_GREEN = 120
V_MAX = 10
SERVICE_RATE = 2  # cars/sec during green
# لو حصل زحمة كبيرة، ممكن نزيد الوزن ده عشان نركز أكتر على تقليل الزحمة بدل الانتظار
CONGESTION_WEIGHT = 120
POPULATION_SIZE = 30
SIM_HORIZON = 150
NUM_GENERATIONS = 50
NUM_RUNS = 30
OUTPUT_DIR = "pso_outputs"

# Constrained Optimisation
# Constraint: 10 <= Green Signal Duration <= 120
# An individual is the time of green signal for NS and EW for each intersection.
# Constraint is handled by clipping the values to the valid range after each update.
# np.clip(self.time, a_min=MIN_GREEN, a_max=MAX_GREEN)

# Resources:
# https://www.sciencedirect.com/science/article/pii/S0096300315014630
# https://www.scirp.org/journal/paperinformation?paperid=70955
# https://www.researchgate.net/publication/287022845
# https://idus.us.es/server/api/core/bitstreams/7544aab6-f0db-493c-bd26-1e36f140302c/content
# https://link.springer.com/chapter/10.1007/BFb0040810

toolbox = base.Toolbox()

if not hasattr(creator, "FitnessMin"):
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
if not hasattr(creator, "Particle"):
    creator.create(
        "Particle", np.ndarray,
        fitness=creator.FitnessMin, speed=None, best=None,
        smin=-V_MAX, smax=V_MAX,
    )


def createParticle():
    particle = creator.Particle(np.random.uniform(MIN_GREEN, MAX_GREEN, NUM_INTERSECTIONS * 2))
    particle.speed = np.random.uniform(-V_MAX, V_MAX, NUM_INTERSECTIONS * 2)
    return particle

toolbox.register("particleCreator", createParticle)
toolbox.register("populationCreator", tools.initRepeat, list, toolbox.particleCreator)



def get_inertia(generation, total_generations):
    if not USE_LINEAR_INERTIA:
        return W_FIXED
    if total_generations <= 1:
        return W_MAX
    fraction = generation / (total_generations - 1)
    return W_MAX - (W_MAX - W_MIN) * fraction


def updateParticle(particle, global_best, inertia_weight):
    r1 = np.random.random(particle.size)
    r2 = np.random.random(particle.size)

    # # calculate local and global speed updates:
    # localSpeedUpdate = localUpdateFactor * (particle.best - particle)
    # globalSpeedUpdate = globalUpdateFactor * (global_best - particle)

    # calculate updated speed (inertia + cognitive + social)
    particle.speed = (
        inertia_weight * particle.speed
        + C1 * r1 * (particle.best - particle)  # Cognitive (local)
        + C2 * r2 * (global_best - particle)  # Social (global)
    )

    # enforce limits on the updated speed:
    particle.speed = np.clip(particle.speed, -V_MAX, V_MAX)

    # replace particle position with old-position + speed:
    particle[:] = particle + particle.speed
    repairParticle(particle)

def repairParticle(particle):
    # Ensure the green signal durations are within the valid range
    particle[:] = np.clip(particle, MIN_GREEN, MAX_GREEN)

toolbox.register("update", updateParticle)

def simulate_traffic(time, traffic_stream):
    if traffic_stream is None or len(traffic_stream) == 0:
        return {
            'total_wait': float('inf'),
            'avg_queue': float('inf'),
            'objective': float('inf')
        }

    queue_ns = [0] * NUM_INTERSECTIONS
    queue_ew = [0] * NUM_INTERSECTIONS
    total_wait = 0.0
    queue_accumulator = 0.0

    for current_time in range(len(traffic_stream)):
        arrivals_per_intersection = traffic_stream[current_time]

        for i in range(NUM_INTERSECTIONS):
            green_ns = time[2 * i]
            green_ew = time[2 * i + 1]

            if green_ns <= 0 or green_ew <= 0:
                return {
                    'total_wait': float('inf'),
                    'avg_queue': float('inf'),
                    'objective': float('inf')
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

    return {
        'total_wait': float(total_wait),
        'avg_queue': float(avg_queue),
        'objective': float(objective)
    }

def evaluate(particle, traffic_stream):
    results = simulate_traffic(particle, traffic_stream)
    return (results['objective'],)

toolbox.register("evaluate", evaluate)

def generate_traffic_stream(sim_time):
    traffic_stream = []
    for _ in range(sim_time):
        arrivals_snapshot = []
        for intersection_idx in range(NUM_INTERSECTIONS):
            # Strictly fair demand model: NS and EW get identical arrivals.
            base_arrival = random.randint(0, 3)
            arrivals_ns = base_arrival
            arrivals_ew = base_arrival
            arrivals_snapshot.append((arrivals_ns, arrivals_ew))
        traffic_stream.append(arrivals_snapshot)
    return traffic_stream

def load_or_create_seeds(filename='seeds.json', num_runs=30):
    try:
        with open(filename, 'r') as f:
            return json.load(f)["seeds"]
    except FileNotFoundError as e:
        print(f"Warning: {filename} not found. Generating new seeds. {e}")
        seeds = [random.randint(1, 10000) for _ in range(num_runs)]

        with open(filename, 'w') as f:
            json.dump({'seeds': seeds}, f, indent=2)
        
        print(f"Generated and saved new {filename}")
        return seeds


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
    axes[0].set_title("PSO Objective Over Generations")
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
    axes[1].set_title("Improvement Over Baseline")
    axes[1].set_xlabel("Generation")
    axes[1].set_ylabel("Improvement (%)")
    axes[1].legend()

    fig.tight_layout()
    progress_plot_path = os.path.join(output_dir, "pso_progress_vs_baseline.png")
    fig.savefig(progress_plot_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(14, 7))
    run_indices = np.arange(1, len(run_histories) + 1)
    width = 0.42
    ax2.bar(run_indices - width / 2, baseline_values, width=width, label="Baseline objective", color="#d62728", alpha=0.85)
    ax2.bar(run_indices + width / 2, final_best_values, width=width, label="PSO final best objective", color="#1f77b4", alpha=0.9)
    ax2.set_title("Per-Run Baseline vs PSO Final Best")
    ax2.set_xlabel("Run index")
    ax2.set_ylabel("Objective (lower is better)")
    ax2.legend()

    fig2.tight_layout()
    comparison_plot_path = os.path.join(output_dir, "baseline_vs_pso_per_run.png")
    fig2.savefig(comparison_plot_path, dpi=180, bbox_inches="tight")
    plt.close(fig2)

    return progress_plot_path, comparison_plot_path

def save_run_summaries(run_histories, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "pso_run_summary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "run_index", "seed", "inertia_scheme",
            "baseline_objective", "final_best",
            "improvement_percent", "best_timings",
        ])
scheme = "linear_decreasing" if USE_LINEAR_INERTIA else f"fixed_W={W_FIXED}"
        for run in run_histories:
            baseline = float(run["baseline_objective"])
            final_best = float(run["final_best"])
            improvement = ((baseline - final_best) / baseline) * 100.0 if baseline else 0.0
            best_timings = "[" + ", ".join(f"{v:.2f}" for v in run["best_solution"]) + "]"
            writer.writerow([
                run["run_index"], run["seed"], scheme,
                f"{baseline:.6f}", f"{final_best:.6f}",
                f"{improvement:.4f}", best_timings,
            ])
    return csv_path

if __name__ == "__main__":
    seeds = load_or_create_seeds('seeds.json', num_runs=NUM_RUNS)
    run_histories = []

    for run_idx, seed_val in enumerate(seeds):
        best = None

        random.seed(seed_val)
        np.random.seed(seed_val)

        # call populationCreator with n=30 to create a population of 30 particles:
        population = toolbox.populationCreator(n=POPULATION_SIZE)

        traffic_stream = generate_traffic_stream(SIM_HORIZON)

        print(f"\n--- RUN {run_idx + 1} (Seed {seed_val}) ---")

        best_curve = []
        avg_curve = []
        best_so_far = float('inf')

        for generation in range(NUM_GENERATIONS):

            for particle in population:
                # find the fitness of the particle:
                particle.fitness.values = toolbox.evaluate(particle, traffic_stream)

                # particle best needs to be updated:
                if particle.best is None or particle.best.size == 0 or particle.best.fitness < particle.fitness:
                    particle.best = creator.Particle(particle)
                    particle.best.fitness.values = particle.fitness.values

                # global best needs to be updated:
                if best is None or best.size == 0 or best.fitness < particle.fitness:
                    best = creator.Particle(particle)
                    best.fitness.values = particle.fitness.values

            generation_fitness = np.array([particle.fitness.values[0] for particle in population], dtype=float)
            generation_min = float(np.min(generation_fitness))
            generation_avg = float(np.mean(generation_fitness))
            best_so_far = min(best_so_far, generation_min)

            best_curve.append(best_so_far)
            avg_curve.append(generation_avg)

            current_w = get_inertia(generation, NUM_GENERATIONS)
            for particle in population:
                toolbox.update(particle, best, current_w)

            if generation % 10 == 0 or generation == NUM_GENERATIONS - 1:
                print(
                    f"Gen {generation:02d} | W={current_w:.3f} | "
                    f"Best so far: {best_so_far:.2f} | Generation avg: {generation_avg:.2f}"
                )
    
        baseline_timings = np.array([60.0, 60.0] * NUM_INTERSECTIONS, dtype=float)
        baseline_metrics = simulate_traffic(baseline_timings, traffic_stream)
        print(
            f"Baseline -> Wait: {baseline_metrics['total_wait']:.2f}, "
            f"Avg Queue: {baseline_metrics['avg_queue']:.2f}, "
            f"Objective: {baseline_metrics['objective']:.2f}"
        )

        baseline_objective = float(baseline_metrics['objective'])
        improvement_curve = [
((baseline_objective - v) / baseline_objective) * 100.0 if baseline_objective else 0.0
for v in best_curve
]
        run_histories.append({
                "run_index": run_idx + 1,
                "seed": seed_val,
                "best_curve": best_curve,
                "avg_curve": avg_curve,
                "improvement_curve": improvement_curve,
                "baseline_objective": baseline_objective,
                "final_best": float(best.fitness.values[0]),
                "best_solution": np.array(best, dtype=float).tolist(),
            })

        # print info for best solution found:
        print("-- Best Particle = ", np.round(best, 2))
        print("-- Best Fitness  = ", best.fitness.values[0])

    progress_path, comparison_path = plot_optimization_results(run_histories, OUTPUT_DIR)
    csv_path = save_run_summaries(run_histories, OUTPUT_DIR)
    print(f"\nSaved plot: {progress_path}")
    print(f"Saved plot: {comparison_path}")
    print(f"Saved summary CSV: {csv_path}")
