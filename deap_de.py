import csv
import json
import os
import random
from collections import defaultdict
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

NUM_POPULATIONS = 4 
POP_SIZE_PER_SUBPOP = 10 
SIM_HORIZON = 150
NUM_GENERATIONS = 50
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
    queue_ns = [0] * NUM_INTERSECTIONS
    queue_ew = [0] * NUM_INTERSECTIONS
    total_wait = 0.0
    queue_accumulator = 0.0

    for current_time in range(len(traffic_stream)):
        arrivals_per_intersection = traffic_stream[current_time]
        for i in range(NUM_INTERSECTIONS):
            green_ns = max(1, time_config[2 * i])
            green_ew = max(1, time_config[2 * i + 1])
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
    return {'total_wait': float(total_wait), 'avg_queue': float(avg_queue), 'objective': float(objective)}

def evaluate(individual, traffic_stream):
    res = simulate_traffic(individual, traffic_stream)
    return (res['objective'],)

toolbox.register("evaluate", evaluate)



def mutation(a, b, c, F, low, up):
    """ حساب المتجه الطافر """
    mutant = c + F * (b - a)
    mutant = np.clip(mutant, low, up)
    return creator.Individual(mutant)

def crossOver(target, mutant, CR):
    """ عملية التقاطع بين الهدف والمتجه الطافر """
    mask = np.random.rand(len(target)) < CR
    trial = np.where(mask, mutant, target)
    return creator.Individual(trial)

def selectedIndices(pop_size, current_idx):
    """ اختيار 3 مؤشرات عشوائية مختلفة """
    indices = list(range(pop_size))
    indices.remove(current_idx)
    return random.sample(indices, 3)


def plot_optimization_results(run_histories, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid", context="talk")
    generations = np.arange(NUM_GENERATIONS)
    best_curves = np.array([run["best_curve"] for run in run_histories], dtype=float)
    improvement_curves = np.array([run["improvement_curve"] for run in run_histories], dtype=float)
    mean_best = best_curves.mean(axis=0)
    std_best = best_curves.std(axis=0)
    mean_improvement = improvement_curves.mean(axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    axes[0].plot(generations, mean_best, label="DE Mean Best Objective", color="#1f77b4", lw=2.5)
    axes[0].fill_between(generations, mean_best - std_best, mean_best + std_best, color="#1f77b4", alpha=0.2)
    axes[0].set_title("DE Objective (4 Populations)")
    axes[0].set_xlabel("Generation")
    axes[0].set_ylabel("Objective")
    axes[0].legend()

    axes[1].plot(generations, mean_improvement, label="Improvement %", color="#ff7f0e", lw=2.5)
    axes[1].axhline(0.0, linestyle="--", color="black")
    axes[1].set_title("Improvement vs Baseline")
    axes[1].set_xlabel("Generation")
    axes[1].set_ylabel("Improvement (%)")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "de_advanced_results.png"))
    plt.close()


if __name__ == "__main__":
    traffic_stream = [[(random.randint(0, 3), random.randint(0, 3)) for _ in range(NUM_INTERSECTIONS)] for _ in range(SIM_HORIZON)]
    run_histories = []

    for run_idx in range(1):
      
        subpopulations = [toolbox.populationCreator(n=POP_SIZE_PER_SUBPOP) for _ in range(NUM_POPULATIONS)]
        
       
        for subpop in subpopulations:
            for ind in subpop:
                ind.fitness.values = toolbox.evaluate(ind, traffic_stream)

        best_overall = None
        best_curve = []

        for gen in range(NUM_GENERATIONS):
            for subpop in subpopulations:
                for i in range(len(subpop)):
                   
                    a_idx, b_idx, c_idx = selectedIndices(len(subpop), i)
                    
                   
                    mutant = mutation(subpop[a_idx], subpop[b_idx], subpop[c_idx], F, MIN_GREEN, MAX_GREEN)
                    
                 
                    trial = crossOver(subpop[i], mutant, CR)
                    trial.fitness.values = toolbox.evaluate(trial, traffic_stream)
                  
                    if trial.fitness.values[0] < subpop[i].fitness.values[0]:
                        subpop[i] = trial

                   
                    if best_overall is None or subpop[i].fitness.values[0] < best_overall.fitness.values[0]:
                        best_overall = creator.Individual(subpop[i])
                        best_overall.fitness.values = subpop[i].fitness.values
            
            best_curve.append(best_overall.fitness.values[0])
            if gen % 10 == 0: print(f"Gen {gen} | Best Objective: {best_overall.fitness.values[0]:.2f}")

        baseline_res = simulate_traffic([60, 60] * NUM_INTERSECTIONS, traffic_stream)
        baseline_obj = baseline_res['objective']
        improvement_curve = [((baseline_obj - v) / baseline_obj) * 100 for v in best_curve]

        run_histories.append({
            "best_curve": best_curve,
            "improvement_curve": improvement_curve,
            "baseline_objective": baseline_obj,
            "final_best": best_overall.fitness.values[0]
        })

        print("\n" + "="*40)
        print(f"RUN {run_idx+1} FINISHED")
        print(f"Baseline Objective: {baseline_obj:.2f}")
        print(f"Optimized Best Individual: ", np.round(best_overall, 2))
        print(f"Optimized Best Fitness: ", best_overall.fitness.values[0])
        print("="*40)

    plot_optimization_results(run_histories, OUTPUT_DIR)
    print(f"\nResults and Plots saved in {OUTPUT_DIR}/")
    print(f"\nResults and Plots saved in {OUTPUT_DIR}/")