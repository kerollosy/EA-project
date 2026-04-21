import random
import numpy as np
import matplotlib.pyplot as plt
import csv
import json
from collections import defaultdict

C1 = 2.0
C2 = 2.0

NUM_INTERSECTIONS = 2
MIN_GREEN = 10
MAX_GREEN = 120
V_MAX = 10 
SERVICE_RATE = 2  # cars/sec during green
# لو حصل زحمة كبيرة، ممكن نزيد الوزن ده عشان نركز أكتر على تقليل الزحمة بدل الانتظار
CONGESTION_WEIGHT = 120

# Constrained Optimisation
# Constraint: 10 <= Green Signal Duration <= 120
# An individual is the time of green signal for NS and EW for each intersection.
# Constraint is handled by clipping the values to the valid range after each update.
# np.clip(self.time, a_min=MIN_GREEN, a_max=MAX_GREEN)

# Resources:
# https://www.sciencedirect.com/science/article/pii/S0096300315014630
# https://www.scirp.org/journal/paperinformation?paperid=70955
# https://www.researchgate.net/publication/287022845_Traffic_signal_control_based_on_particle_swarm_optimization
# https://idus.us.es/server/api/core/bitstreams/7544aab6-f0db-493c-bd26-1e36f140302c/content
# https://link.springer.com/chapter/10.1007/BFb0040810


class Swarm:
    def __init__(self, num_particles, traffic_stream=None):
        self.global_best = None
        self.global_best_fit = float('inf')
        self.particles = []
        self.traffic_stream = traffic_stream
        self.fitness_history = []

        for _ in range(num_particles):
            particle = Particle(traffic_stream=traffic_stream)
            self.particles.append(particle)

            if particle.fitness < self.global_best_fit:
                self.global_best_fit = particle.fitness
                self.global_best = particle.time.copy()

    def optimize(self, iterations):
        w = 0.9 
        for _ in range(iterations):
            for particle in self.particles:
                particle.update(self.global_best, w, self.traffic_stream) 
                
                if particle.fitness < self.global_best_fit:
                    self.global_best_fit = particle.fitness
                    self.global_best = particle.time.copy()
            
            self.fitness_history.append(self.global_best_fit)
            w = max(0.4, w - 0.02)

class Particle:
    def __init__(self, traffic_stream=None):
        timing = []
        velocity = []
        for _ in range(NUM_INTERSECTIONS):
            ns = random.uniform(MIN_GREEN, MAX_GREEN)
            ew = random.uniform(MIN_GREEN, MAX_GREEN)
            vns = random.uniform(-V_MAX, V_MAX)
            vew = random.uniform(-V_MAX, V_MAX)
            timing.extend((ns, ew))
            velocity.extend((vns, vew))

        self.time = np.array(timing, dtype=float)
        self.v = np.array(velocity, dtype=float)
        self.traffic_stream = traffic_stream

        self.fitness = self.evaluate(self.time, self.traffic_stream)
        self.pbest = self.time.copy()
        self.pbest_fit = self.fitness

    def update(self, global_best, w, traffic_stream=None):
        r1 = np.random.random(len(self.time))
        r2 = np.random.random(len(self.time))

        self.v = w * self.v + C1 * r1 * (self.pbest - self.time) + C2 * r2 * (global_best - self.time)
        self.v = np.clip(self.v, a_min=-V_MAX, a_max=V_MAX)

        self.time += self.v

        mutation_rate = 0.10  # 10% chance to mutate
        if random.random() < mutation_rate:
            # Pick one signal (NS or EW) to mutate at random
            gene_index = random.randint(0, len(self.time) - 1)
            
            # Add or subtract a random amount of time (between -15 and 15 seconds)
            mutation_shift = random.uniform(-15, 15)
            self.time[gene_index] += mutation_shift

        self.time = np.clip(self.time, a_min=MIN_GREEN, a_max=MAX_GREEN)

        self.fitness = self.evaluate(self.time, traffic_stream)
        if self.fitness < self.pbest_fit:
            self.pbest = self.time.copy()
            self.pbest_fit = self.fitness

    def evaluate(self, time, traffic_stream=None):
        metrics = self.simulate(time, traffic_stream)
        return metrics['objective']
    
    def simulate(self, time, traffic_stream=None):
        if traffic_stream is None or len(traffic_stream) == 0:
            return {
                'total_wait': float('inf'),
                'avg_queue': float('inf'),
                'objective': float('inf')
            }

        queue_NS = [0] * NUM_INTERSECTIONS
        queue_EW = [0] * NUM_INTERSECTIONS
        total_wait = 0
        queue_accumulator = 0

        for current_time in range(len(traffic_stream)):
            arrivals_per_intersection = traffic_stream[current_time]

            for i in range(NUM_INTERSECTIONS):
                # Time array: [NS_green_duration, EW_green_duration]
                green_NS_duration = time[2 * i]
                green_EW_duration = time[2 * i + 1]

                if green_NS_duration <= 0 or green_EW_duration <= 0:
                    return {
                        'total_wait': float('inf'),
                        'avg_queue': float('inf'),
                        'objective': float('inf')
                    }

                arrivals_ns, arrivals_ew = arrivals_per_intersection[i]
                queue_NS[i] += arrivals_ns
                queue_EW[i] += arrivals_ew

                # Determine who has the green light
                total_cycle = green_NS_duration + green_EW_duration
                time_in_cycle = current_time % int(total_cycle)

                if time_in_cycle < green_NS_duration:
                    # NS is Green, EW is Red
                    queue_NS[i] = max(0, queue_NS[i] - SERVICE_RATE)
                else:
                    # EW is Green, NS is Red
                    queue_EW[i] = max(0, queue_EW[i] - SERVICE_RATE)

                current_total_queue = queue_NS[i] + queue_EW[i]
                total_wait += current_total_queue
                queue_accumulator += current_total_queue

        avg_queue = queue_accumulator / (len(traffic_stream) * NUM_INTERSECTIONS)
        objective = total_wait + CONGESTION_WEIGHT * avg_queue

        return {
            'total_wait': float(total_wait),
            'avg_queue': float(avg_queue),
            'objective': float(objective)
        }


if __name__ == "__main__":
    try:
        with open('seeds.json', 'r') as f:
            seeds = json.load(f)["seeds"]
    except FileNotFoundError as e:
        print(f"Warning: seeds.json not found. Generating new seeds. {e}")
        seeds = [random.randint(1, 10000) for _ in range(30)]

    results = []
    convergence_data = defaultdict(list)
    
    print("=" * 80)
    print("Traffic Signal Optimization using PSO with Fair Baseline Comparison")
    print("=" * 80)
    
    for i, seed_val in enumerate(seeds):
        random.seed(seed_val)
        np.random.seed(seed_val)
        
        # Generate traffic scenario once for this seed
        # Simulate 600 seconds (10 minutes)
        sim_time = 600
        traffic_stream = []
        for _ in range(sim_time):
            arrivals_snapshot = []
            for intersection_idx in range(NUM_INTERSECTIONS):
                # Slightly different patterns per intersection to mimic a small network.
                arrivals_ns = random.randint(0, 2 + (intersection_idx % 2))
                arrivals_ew = random.randint(0, 1 + (1 if intersection_idx in (1, 3) else 0))
                arrivals_snapshot.append((arrivals_ns, arrivals_ew))
            traffic_stream.append(arrivals_snapshot)
        
        print(f"\n--- RUN {i+1} (Seed {seed_val}) ---")
        
        # BASELINE: Fixed 60/60 timing on the same traffic stream
        baseline_particle = Particle(traffic_stream=traffic_stream)
        baseline_timings = np.array([60.0, 60.0] * NUM_INTERSECTIONS)
        baseline_metrics = baseline_particle.simulate(baseline_timings, traffic_stream)
        print(
            f"Baseline -> Wait: {baseline_metrics['total_wait']:.2f}, "
            f"Avg Queue: {baseline_metrics['avg_queue']:.2f}, "
            f"Objective: {baseline_metrics['objective']:.2f}"
        )
        
        # PSO OPTIMIZATION: Use same traffic stream
        swarm = Swarm(num_particles=30, traffic_stream=traffic_stream)
        swarm.optimize(iterations=50)
        
        # Store convergence history
        convergence_data[seed_val] = swarm.fitness_history

        optimized_metrics = baseline_particle.simulate(swarm.global_best, traffic_stream)
        objective_improvement = (
            ((baseline_metrics['objective'] - optimized_metrics['objective']) / baseline_metrics['objective']) * 100
            if baseline_metrics['objective'] > 0 else 0
        )

        timings_by_intersection = swarm.global_best.reshape(NUM_INTERSECTIONS, 2)
        timing_text = ", ".join(
            [f"I{i+1}(NS={t[0]:.1f}, EW={t[1]:.1f})" for i, t in enumerate(timings_by_intersection)]
        )

        print(f"Optimized Timings: {timing_text}")
        print(
            f"Optimized -> Wait: {optimized_metrics['total_wait']:.2f}, "
            f"Avg Queue: {optimized_metrics['avg_queue']:.2f}, "
            f"Objective: {optimized_metrics['objective']:.2f}"
        )
        print(f"Objective Improvement: {objective_improvement:.2f}%")
        
        # Collect results
        results.append({
            'run': i + 1,
            'seed': seed_val,
            'baseline_wait': baseline_metrics['total_wait'],
            'baseline_congestion': baseline_metrics['avg_queue'],
            'baseline_objective': baseline_metrics['objective'],
            'optimized_wait': optimized_metrics['total_wait'],
            'optimized_congestion': optimized_metrics['avg_queue'],
            'optimized_objective': optimized_metrics['objective'],
            'improvement_percent': objective_improvement,
            'timings': timing_text,
            'final_fitness': swarm.global_best_fit
        })
    
    # Save results to CSV
    print("\n" + "=" * 80)
    print("Saving results to CSV...")
    csv_filename = 'pso_results.csv'
    with open(csv_filename, 'w', newline='') as csvfile:
        fieldnames = [
            'run', 'seed',
            'baseline_wait', 'baseline_congestion', 'baseline_objective',
            'optimized_wait', 'optimized_congestion', 'optimized_objective',
            'improvement_percent', 'timings', 'final_fitness'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"Results saved to {csv_filename}")
    

    baseline_waits = [r['baseline_wait'] for r in results]
    baseline_congestions = [r['baseline_congestion'] for r in results]
    baseline_objectives = [r['baseline_objective'] for r in results]
    optimized_waits = [r['optimized_wait'] for r in results]
    optimized_congestions = [r['optimized_congestion'] for r in results]
    optimized_objectives = [r['optimized_objective'] for r in results]
    improvements = [r['improvement_percent'] for r in results]

    # Save seeds and summary statistics to JSON
    metadata_filename = 'metadata.json'
    metadata = {
        'num_runs': len(seeds),
        'summary_statistics': {
            'mean_baseline_wait': float(np.mean(baseline_waits)),
            'mean_baseline_congestion': float(np.mean(baseline_congestions)),
            'mean_baseline_objective': float(np.mean(baseline_objectives)),
            'mean_optimized_wait': float(np.mean(optimized_waits)),
            'mean_optimized_congestion': float(np.mean(optimized_congestions)),
            'mean_optimized_objective': float(np.mean(optimized_objectives)),
            'mean_improvement_percent': float(np.mean(improvements)),
            'std_improvement_percent': float(np.std(improvements)),
            'min_improvement_percent': float(np.min(improvements)),
            'max_improvement_percent': float(np.max(improvements))
        }
    }
    with open(metadata_filename, 'w') as jsonfile:
        json.dump(metadata, jsonfile, indent=2)
    
    print(f"Metadata saved to {metadata_filename}")

    # seeds_filename = 'seeds.json'
    # with open(seeds_filename, 'w') as seedfile:
    #     json.dump({'seeds': seeds}, seedfile, indent=2)
    
    # print(f"Seeds saved to {seeds_filename}")
    

    # Print summary statistics
    print("\n" + "=" * 80)
    print("Statistical Summary")
    print("=" * 80)
    
    print(f"Baseline Wait (Mean ± Std): {np.mean(baseline_waits):.2f} ± {np.std(baseline_waits):.2f}")
    print(f"Optimized Wait (Mean ± Std): {np.mean(optimized_waits):.2f} ± {np.std(optimized_waits):.2f}")
    print(f"Baseline Congestion (Mean): {np.mean(baseline_congestions):.2f}")
    print(f"Optimized Congestion (Mean): {np.mean(optimized_congestions):.2f}")
    print(f"Baseline Objective (Mean): {np.mean(baseline_objectives):.2f}")
    print(f"Optimized Objective (Mean): {np.mean(optimized_objectives):.2f}")
    print(f"Objective Improvement (Mean ± Std): {np.mean(improvements):.2f}% ± {np.std(improvements):.2f}%")
    print(f"Objective Improvement Range: [{np.min(improvements):.2f}%, {np.max(improvements):.2f}%]")
    
    # Generate Grpahs
    print("\n" + "=" * 80)
    print("Generating visualizations...")

    # Plot 1: Convergence curves (average across all runs)
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    mean_convergence = np.mean([convergence_data[s] for s in seeds], axis=0)
    ax1.plot(mean_convergence, linewidth=2, color='blue', label='PSO Convergence (Mean)')
    ax1.set_xlabel('Iteration', fontsize=12)
    ax1.set_ylabel('Best Fitness (Objective)', fontsize=12)
    ax1.set_title('PSO Convergence Over 50 Iterations (Objective, 4 Intersections)', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    fig1.tight_layout()
    fig1.savefig('convergence_curve.png', dpi=300)
    print("Saved: convergence_curve.png")
    
    # Plot 2: Baseline vs Optimized (bar chart)
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    x_pos = np.arange(len(results))
    width = 0.35
    
    ax2.bar(x_pos - width/2, baseline_objectives, width, label='Baseline Objective', alpha=0.8, color='orange')
    ax2.bar(x_pos + width/2, optimized_objectives, width, label='PSO Objective', alpha=0.8, color='green')
    
    ax2.set_xlabel('Run Number', fontsize=12)
    ax2.set_ylabel('Objective Value', fontsize=12)
    ax2.set_title('Baseline vs PSO Objective (30 Runs, 4 Intersections)', fontsize=14)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([str(r+1) for r in range(len(results))])
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    fig2.tight_layout()
    fig2.savefig('baseline_vs_optimized.png', dpi=300)
    print("Saved: baseline_vs_optimized.png")
    
    # Plot 3: Improvement percentage distribution
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    ax3.hist(improvements, bins=10, color='purple', alpha=0.7, edgecolor='black')
    ax3.axvline(np.mean(improvements), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(improvements):.2f}%')
    ax3.set_xlabel('Objective Improvement (%)', fontsize=12)
    ax3.set_ylabel('Frequency', fontsize=12)
    ax3.set_title('Distribution of Objective Improvements (30 Runs)', fontsize=14)
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    fig3.tight_layout()
    fig3.savefig('improvement_distribution.png', dpi=300)
    print("Saved: improvement_distribution.png")