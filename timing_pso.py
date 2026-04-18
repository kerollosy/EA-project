import random
import numpy as np

C1 = 2.0
C2 = 2.0

# Constrained Optimisation
# Constrain: 10 <= Green Signal Duration <= 120
# An individual is the time of green signal for NS and EW for each intersection.
# Constraint is handled by clipping the values to the valid range after each update.

class Swarm:
    def __init__(self, num_particles):
        self.global_best = None
        self.global_best_fit = float('inf')
        self.particles = []

        for _ in range(num_particles):
            particle = Particle()
            self.particles.append(particle)

            if particle.fitness < self.global_best_fit:
                self.global_best_fit = particle.fitness
                self.global_best = particle.time.copy()

    def optimize(self, iterations):
        w = 0.9 
        for _ in range(iterations):
            for particle in self.particles:
                particle.update(self.global_best, w) 
                
                if particle.fitness < self.global_best_fit:
                    self.global_best_fit = particle.fitness
                    self.global_best = particle.time.copy()
            
            w = max(0.4, w - 0.02)

class Particle:
    def __init__(self):
        ns = random.uniform(10, 60)
        ew = random.uniform(10, 60)
        vns = random.uniform(-10, 10)
        vew = random.uniform(-10, 10)

        self.time = np.array([ns, ew])
        self.v = np.array([vns, vew])

        self.fitness = self.evaluate(self.time)
        self.pbest = self.time.copy()
        self.pbest_fit = self.fitness

    def update(self, global_best, w):
        r1 = np.random.random(len(self.time))
        r2 = np.random.random(len(self.time))

        self.v = w * self.v + C1 * r1 * (self.pbest - self.time) + C2 * r2 * (global_best - self.time)
        self.time += self.v
        self.time = np.clip(self.time, a_min=10, a_max=120)

        self.fitness = self.evaluate(self.time)
        if self.fitness < self.pbest_fit:
            self.pbest = self.time.copy()
            self.pbest_fit = self.fitness

    def evaluate(self, time):
        return self.simulate(time)
    
    def simulate(self, time):
        # Time array: [NS_green_duration, EW_green_duration]
        green_NS_duration = time[0]
        green_EW_duration = time[1]

        if green_NS_duration <= 0 or green_EW_duration <= 0:
            return float('inf')

        total_cycle = green_NS_duration + green_EW_duration

        queue_NS = 0
        queue_EW = 0
        total_wait = 0

        for current_time in range(100): # 100 seconds of simulation
            queue_NS += random.randint(0, 5)
            queue_EW += random.randint(0, 1)

            # Determine who has the green light
            time_in_cycle = current_time % total_cycle

            if time_in_cycle < green_NS_duration:
                # NS is Green, EW is Red
                if queue_NS > 0:
                    queue_NS -= 1
            else:
                # EW is Green, NS is Red
                if queue_EW > 0:
                    queue_EW -= 1

            # Add all waiting cars to the penalty score
            total_wait += (queue_NS + queue_EW)

        return total_wait


if __name__ == "__main__":
    seeds = [random.randint(1, 10000) for _ in range(30)] # Generate 30 seeds
    
    for i, seed_val in enumerate(seeds):
        random.seed(seed_val)
        np.random.seed(seed_val)
        
        swarm = Swarm(num_particles=30)
        swarm.optimize(iterations=50)
        print(f"Run {i+1} (Seed {seed_val}) - Global Best: {swarm.global_best}")