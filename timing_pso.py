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
        return self.simulate(time, traffic_stream)
    
    def simulate(self, time, traffic_stream=None):
        queue_NS = [0] * NUM_INTERSECTIONS
        queue_EW = [0] * NUM_INTERSECTIONS
        total_wait = 0

        for current_time in range(len(traffic_stream)):
            arrivals_per_intersection = traffic_stream[current_time]

            for intersection in range(NUM_INTERSECTIONS):
                # Time array: [NS_green_duration, EW_green_duration]
                green_NS_duration = time[2 * intersection]
                green_EW_duration = time[2 * intersection + 1]

                if green_NS_duration <= 0 or green_EW_duration <= 0:
                    return float('inf')
            
                arrivals_ns, arrivals_ew = arrivals_per_intersection[intersection]
                queue_NS[intersection] += arrivals_ns
                queue_EW[intersection] += arrivals_ew

                # Determine who has the green light
                total_cycle = green_NS_duration + green_EW_duration
                time_in_cycle = current_time % total_cycle

                if time_in_cycle < green_NS_duration:
                    # NS is Green, EW is Red
                    # 2 cars pass per second
                    queue_NS[intersection] = max(0, queue_NS[intersection] - 2)  # Ensure queue doesn't go negative
                else:
                    # EW is Green, NS is Red
                    queue_EW[intersection] = max(0, queue_EW[intersection] - 2)  # Ensure queue doesn't go negative

                # Add all waiting cars to the penalty score
                total_wait += (queue_NS[intersection] + queue_EW[intersection])

        return total_wait


if __name__ == "__main__":
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
        for t in range(sim_time):
            b = []
            for intersection in range(NUM_INTERSECTIONS):
                # NS gets 0, 1, or 2 cars (Average: 1 car per sec)
                arrivals_ns = random.randint(0, 2)

                # EW gets 0 or 1 car (Average: 0.5 cars per sec)
                arrivals_ew = random.randint(0, 1)
                b.append((arrivals_ns, arrivals_ew))

            traffic_stream.append(b)
        
        print(f"\n--- RUN {i+1} (Seed {seed_val}) ---")
        
        # BASELINE: Fixed 60/60 timing on the same traffic stream
        baseline_particle = Particle(traffic_stream=traffic_stream)
        baseline_timings = np.array([60.0, 60.0] * NUM_INTERSECTIONS)
        baseline_wait = baseline_particle.simulate(baseline_timings, traffic_stream)
        print(f"Baseline Wait Time (60s/60s fixed): {baseline_wait:.2f} cars waiting")
        
        # PSO OPTIMIZATION: Use same traffic stream
        swarm = Swarm(num_particles=30, traffic_stream=traffic_stream)
        swarm.optimize(iterations=50)
        
        # Store convergence history
        convergence_data[seed_val] = swarm.fitness_history
        
        optimized_wait = swarm.global_best_fit
        improvement = ((baseline_wait - optimized_wait) / baseline_wait) * 100 if baseline_wait > 0 else 0
        
        print(f"Optimized Timings: NS={swarm.global_best[0]:.2f}s, EW={swarm.global_best[1]:.2f}s")
        print(f"Optimized Wait Time: {optimized_wait:.2f} cars waiting")
        print(f"Improvement: {improvement:.2f}%")
        
        # Collect results
        results.append({
            'run': i + 1,
            'seed': seed_val,
            'baseline_wait': baseline_wait,
            'optimized_wait': optimized_wait,
            'improvement_percent': improvement,
            'ns_timing': swarm.global_best[0],
            'ew_timing': swarm.global_best[1],
            'final_fitness': swarm.global_best_fit
        })
    
    # Save results to CSV
    print("\n" + "=" * 80)
    print("Saving results to CSV...")
    csv_filename = 'pso_results.csv'
    with open(csv_filename, 'w', newline='') as csvfile:
        fieldnames = ['run', 'seed', 'baseline_wait', 'optimized_wait', 'improvement_percent', 'ns_timing', 'ew_timing', 'final_fitness']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"Results saved to {csv_filename}")
    

    baseline_waits = [r['baseline_wait'] for r in results]
    optimized_waits = [r['optimized_wait'] for r in results]
    improvements = [r['improvement_percent'] for r in results]

    # Save seeds and summary statistics to JSON
    json_filename = 'pso_seeds_and_metadata.json'
    metadata = {
        'num_runs': len(seeds),
        'seeds': seeds,
        'summary_statistics': {
            'mean_baseline_wait': float(np.mean(baseline_waits)),
            'mean_optimized_wait': float(np.mean(optimized_waits)),
            'mean_improvement_percent': float(np.mean(improvements)),
            'std_improvement_percent': float(np.std(improvements)),
            'min_improvement_percent': float(np.min(improvements)),
            'max_improvement_percent': float(np.max(improvements))
        }
    }
    with open(json_filename, 'w') as jsonfile:
        json.dump(metadata, jsonfile, indent=2)
    print(f"Seeds and metadata saved to {json_filename}")
    

    # Print summary statistics
    print("\n" + "=" * 80)
    print("Statistical Summary")
    print("=" * 80)
    
    print(f"Baseline (Mean ± Std): {np.mean(baseline_waits):.2f} ± {np.std(baseline_waits):.2f} cars waiting")
    print(f"Optimized (Mean ± Std): {np.mean(optimized_waits):.2f} ± {np.std(optimized_waits):.2f} cars waiting")
    print(f"Improvement (Mean ± Std): {np.mean(improvements):.2f}% ± {np.std(improvements):.2f}%")
    print(f"Improvement Range: [{np.min(improvements):.2f}%, {np.max(improvements):.2f}%]")
    
    # Generate Grpahs
    print("\n" + "=" * 80)
    print("Generating visualizations...")
    
    # Plot 1: Convergence curves (average across all runs)
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    mean_convergence = np.mean([convergence_data[s] for s in seeds], axis=0)
    ax1.plot(mean_convergence, linewidth=2, color='blue', label='PSO Convergence (Mean)')
    ax1.set_xlabel('Iteration', fontsize=12)
    ax1.set_ylabel('Best Fitness (Total Wait Time)', fontsize=12)
    ax1.set_title('PSO Convergence Over 50 Iterations (Averaged Across 30 Runs)', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    fig1.tight_layout()
    fig1.savefig('convergence_curve.png', dpi=300)
    print("Saved: convergence_curve.png")
    
    # Plot 2: Baseline vs Optimized (bar chart)
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    x_pos = np.arange(len(results))
    width = 0.35
    
    ax2.bar(x_pos - width/2, baseline_waits, width, label='Baseline (60/60)', alpha=0.8, color='orange')
    ax2.bar(x_pos + width/2, optimized_waits, width, label='PSO Optimized', alpha=0.8, color='green')
    
    ax2.set_xlabel('Run Number', fontsize=12)
    ax2.set_ylabel('Total Wait Time (cars)', fontsize=12)
    ax2.set_title('Baseline vs PSO-Optimized Signal Timings (30 Runs)', fontsize=14)
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
    ax3.set_xlabel('Improvement (%)', fontsize=12)
    ax3.set_ylabel('Frequency', fontsize=12)
    ax3.set_title('Distribution of Improvement Percentages (30 Runs)', fontsize=14)
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    fig3.tight_layout()
    fig3.savefig('improvement_distribution.png', dpi=300)
    print("Saved: improvement_distribution.png")