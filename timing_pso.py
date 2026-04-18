import random
import numpy as np
import csv
import json

C1 = 2.0
C2 = 2.0

# Constrained Optimisation
# Constraint: 10 <= Green Signal Duration <= 120
# An individual is the time of green signal for NS and EW for each intersection.
# Constraint is handled by clipping the values to the valid range after each update.
# np.clip(self.time, a_min=10, a_max=120)

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
            
            w = max(0.4, w - 0.02)

class Particle:
    def __init__(self, traffic_stream=None):
        ns = random.uniform(10, 60)
        ew = random.uniform(10, 60)
        vns = random.uniform(-10, 10)
        vew = random.uniform(-10, 10)

        self.time = np.array([ns, ew])
        self.v = np.array([vns, vew])
        self.traffic_stream = traffic_stream

        self.fitness = self.evaluate(self.time, self.traffic_stream)
        self.pbest = self.time.copy()
        self.pbest_fit = self.fitness

    def update(self, global_best, w, traffic_stream=None):
        r1 = np.random.random(len(self.time))
        r2 = np.random.random(len(self.time))

        self.v = w * self.v + C1 * r1 * (self.pbest - self.time) + C2 * r2 * (global_best - self.time)
        self.v = np.clip(self.v, a_min=-10, a_max=10)

        self.time += self.v

        mutation_rate = 0.10  # 10% chance to mutate
        if random.random() < mutation_rate:
            # Pick one signal (NS or EW) to mutate at random
            gene_index = random.randint(0, len(self.time) - 1)
            
            # Add or subtract a random amount of time (between -15 and 15 seconds)
            mutation_shift = random.uniform(-15, 15)
            self.time[gene_index] += mutation_shift

        self.time = np.clip(self.time, a_min=10, a_max=120)

        self.fitness = self.evaluate(self.time, traffic_stream)
        if self.fitness < self.pbest_fit:
            self.pbest = self.time.copy()
            self.pbest_fit = self.fitness

    def evaluate(self, time, traffic_stream=None):
        return self.simulate(time, traffic_stream)
    
    def simulate(self, time, traffic_stream=None):
        # Time array: [NS_green_duration, EW_green_duration]
        green_NS_duration = time[0]
        green_EW_duration = time[1]

        if green_NS_duration <= 0 or green_EW_duration <= 0:
            return float('inf')

        total_cycle = green_NS_duration + green_EW_duration

        queue_NS = 0
        queue_EW = 0
        total_wait = 0

        for current_time in range(len(traffic_stream)):
            arrivals_ns, arrivals_ew = traffic_stream[current_time]
            queue_NS += arrivals_ns
            queue_EW += arrivals_ew

            # Determine who has the green light
            time_in_cycle = current_time % total_cycle

            if time_in_cycle < green_NS_duration:
                # NS is Green, EW is Red
                if queue_NS > 0:
                    queue_NS -= 2  # 2 cars pass per second
            else:
                # EW is Green, NS is Red
                if queue_EW > 0:
                    queue_EW -= 2  # 2 cars pass per second

            # Add all waiting cars to the penalty score
            total_wait += (queue_NS + queue_EW)

        return total_wait


if __name__ == "__main__":
    seeds = [random.randint(1, 10000) for _ in range(30)]
    results = []
    
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
            # NS gets 0, 1, or 2 cars (Average: 1 car per sec)
            arrivals_ns = random.randint(0, 2)

            # EW gets 0 or 1 car (Average: 0.5 cars per sec)
            arrivals_ew = random.randint(0, 1)
            traffic_stream.append((arrivals_ns, arrivals_ew))
        
        print(f"\n--- RUN {i+1} (Seed {seed_val}) ---")
        
        # BASELINE: Fixed 60/60 timing on the same traffic stream
        baseline_particle = Particle(traffic_stream=traffic_stream)
        baseline_timings = np.array([60.0, 60.0])
        baseline_wait = baseline_particle.simulate(baseline_timings, traffic_stream)
        print(f"Baseline Wait Time (60s/60s fixed): {baseline_wait:.2f} cars waiting")
        
        # PSO OPTIMIZATION: Use same traffic stream
        swarm = Swarm(num_particles=30, traffic_stream=traffic_stream)
        swarm.optimize(iterations=50)
        
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
    
    print(f"Baseline (Mean ± Std): {np.mean(baseline_waits):.2f} ± {np.std(baseline_waits):.2f}")
    print(f"Optimized (Mean ± Std): {np.mean(optimized_waits):.2f} ± {np.std(optimized_waits):.2f}")
    print(f"Improvement (Mean ± Std): {np.mean(improvements):.2f}% ± {np.std(improvements):.2f}%")
    print(f"Improvement Range: [{np.min(improvements):.2f}%, {np.max(improvements):.2f}%]")
    