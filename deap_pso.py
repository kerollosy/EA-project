import csv
import json
import random
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from deap import base, creator, tools

# Hyperparameters
W = 0.7
C1 = 2.0
C2 = 2.0

NUM_INTERSECTIONS = 1
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

toolbox = base.Toolbox()

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Particle", np.ndarray, fitness=creator.FitnessMin, speed=None, best=None, smin=-V_MAX, smax=V_MAX)

def createParticle():
    particle = creator.Particle(np.random.uniform(MIN_GREEN, MAX_GREEN, NUM_INTERSECTIONS * 2))
    particle.speed = np.random.uniform(-V_MAX, V_MAX, NUM_INTERSECTIONS * 2)
    return particle

toolbox.register("particleCreator", createParticle)
toolbox.register("populationCreator", tools.initRepeat, list, toolbox.particleCreator)

def updateParticle(particle, global_best):
    # create random factors:
    # localUpdateFactor = np.random.uniform(0, C1, particle.size)
    # globalUpdateFactor = np.random.uniform(0, C2, particle.size)
    r1 = np.random.random(particle.size)
    r2 = np.random.random(particle.size)

    # # calculate local and global speed updates:
    # localSpeedUpdate = localUpdateFactor * (particle.best - particle)
    # globalSpeedUpdate = globalUpdateFactor * (global_best - particle)

    # calculate updated speed (inertia + cognitive + social)
    particle.speed = (
        W * particle.speed
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

def evaluate(particle, traffic_stream):
    green_NS_duration = particle[0]
    green_EW_duration = particle[1]

    if green_NS_duration <= 0 or green_EW_duration <= 0:
        return (float('inf'),)

    total_cycle = green_NS_duration + green_EW_duration
    
    queue_NS = 0
    queue_EW = 0
    total_wait = 0

    for current_time in range(len(traffic_stream)):
        arrivals_ns, arrivals_ew = traffic_stream[current_time][0]
        queue_NS += arrivals_ns
        queue_EW += arrivals_ew

        # Determine who has the green light
        time_in_cycle = current_time % total_cycle

        if time_in_cycle < green_NS_duration:
            # NS is Green, EW is Red
            # 2 cars pass per second
            queue_NS = max(0, queue_NS - SERVICE_RATE)  # Ensure queue doesn't go negative
        else:
            # EW is Green, NS is Red
            queue_EW = max(0, queue_EW - SERVICE_RATE)  # Ensure queue doesn't go negative

        # Add all waiting cars to the penalty score
        total_wait += (queue_NS + queue_EW)

    return (total_wait,)

toolbox.register("evaluate", evaluate)

def generate_traffic_stream(sim_time):
    traffic_stream = []
    for _ in range(sim_time):
        arrivals_snapshot = []
        for intersection_idx in range(NUM_INTERSECTIONS):
            arrivals_ns = random.randint(0, 2 + (intersection_idx % 2))
            arrivals_ew = random.randint(0, 1 + (1 if intersection_idx in (1, 3) else 0))
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

if __name__ == "__main__":
    seeds = load_or_create_seeds('seeds.json', num_runs=30)

    # prepare the statistics object:
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min)
    stats.register("avg", np.mean)

    logbook = tools.Logbook()
    logbook.header = ["gen", "evals"] + stats.fields

    for i, seed_val in enumerate(seeds):
        best = None

        random.seed(seed_val)
        np.random.seed(seed_val)

        # call populationCreator with n=30 to create a population of 30 particles:
        population = toolbox.populationCreator(n=30)

        sim_time = 600
        traffic_stream = generate_traffic_stream(sim_time)

        print(f"\n--- RUN {i+1} (Seed {seed_val}) ---")

        for generation in range(sim_time):

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

            # update each particle's speed and position:
            for particle in population:
                toolbox.update(particle, best)

            # record the statistics for the current generation and print it:
            # logbook.record(gen=generation, evals=len(population), **stats.compile(population))
            # print(logbook.stream)
    
        # print info for best solution found:
        print("-- Best Particle = ", np.round(best, 2))
        print("-- Best Fitness  = ", best.fitness.values[0])