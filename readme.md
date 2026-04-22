# Traffic Signal Timing Optimisation using Particle Swarm Optimisation (PSO) [Preferably in combination with other EAs/SI approaches of your choice]: 
o Model a simplified urban traffic network with multiple intersections. 

o Define control variables (signal timings) and objectives (minimising wait times, congestion). 

o Use PSO to explore the space of signal timing configurations. 

o Simulate traffic flows using time-step updates and evaluate performance.

o Produce visual output demonstrating traffic improvements as signals adjust. 

o You may employ other EAs/SI approaches instead of the PSO. 

### Detailed Description:
Context and Problem Statement: Optimising traffic signals improves urban mobility by reducing delays and fuel consumption. This project leverages PSO to dynamically adjust signal timings in a simulated city environment.

### Key Terms and Concepts: 
o Traffic Signal Optimization: Adjusting the timing of traffic lights to improve flow. 

o Particle Swarm Optimisation (PSO): Uses simple “particles” (candidate solutions) that move within a search space influenced by personal and group bests.

o Objective Functions: Criteria such as average waiting time and number of stops that measure traffic performance.

### Requirements/Deliverables:
o A Python simulation that represents an intersection network along with adjustable signals. 

o An implementation of PSO to optimise the timing settings.

o Recorded results comparing baseline signal settings with optimized ones. 

o A final report with charts, simulation videos/screenshots, and a discussion of parameter effects


## Important Guidelines for ALL Projects to follow:

a) Clearly define and formalise your problem as one of the problem types we’ve studied throughout this module: Optimisation, Modelling, Simulation, Constraint Satisfaction, Free Optimisation, Constrained Optimisation, etc. 

b) If your selected idea mandates Constraint Handling, clearly implement an approach to handling constraints (i.e., Penalty Functions, Repair Functions, Restricting Search to the Feasible Region, Decoder Functions, etc.). 

c) If your selected idea mandates Coevolution, clearly implement a coevolutionary approach (cooperative or competitive). 

d) Clearly define the Components of your Evolutionary Algorithm: For instance, in the case of a genetic algorithm, clearly define the Representation (Definition of Individuals), Evaluation Function (Fitness Function), Population, Parent Selection Mechanism, Variation Operators (Mutation and Recombination), Survivor Selection Mechanism (Replacement), Initialisation, and Termination Condition(s). 

e) Select variation operators (mutation and recombination) suitable for the selected representation.
 
• If possible, use at least 2 parent selection techniques (each independently) and report the results for each. 

• If possible, use at least 2 recombination techniques (each independently) and report the results for each. 

• If possible, use at least 2 mutation techniques (each independently) and report the results for each. 

f) If possible, use at least 2 population-management-models/survivor-selection (each independently) and report the results for each. 

g) Clearly describe and implement approaches to control/tune the parameters. 

h) Describe and implement a suitable approach for preserving diversity (i.e., Fitness Sharing, Crowding, Automatic Speciation Using Mating Restrictions, Running Multiple Populations in Tandem, such as the Island Model EAs, Spatial Distribution within One Population, such as Cellular EAs, etc.). 

i) Incorporate a functional user interface demonstrating the algorithm, parameters, inputs, and results. 

j) The students may be awarded bonus marks in the following cases (only if the experiments are carried out properly, and the results/performance were measured, reported, and analysed adequately: 

o Investigating the effect of multiple (at least 2) representations (when possible). 

o Investigating the effect of multiple (at least 2) initialisation approaches (when possible). 

o Investigating the effect of over-selection for large populations (when possible). 

o An educational visual interface that illustrates (simulates) the changes in the evolutionary process and solutions when different parameters/options/approaches are selected. I.e., an interface/simulation that can be later used to teach students the effects of varying selected approaches/parameters, etc. 

o Contributions (experiments and results) with high publication potential. 

o Hybrid approaches (employing more than a single EAs/SI approach). 

o Employing SOTA novel variants of the EAs/SI approaches rather than the traditional implementations.

k) Report the results (independently) for each setting (e.g., a choice of a mutation operator, a recombination operator, a representation, a parent selection approach, a survivor selection approach, an initialisation, an approach for preserving diversity, etc.). 

• The evolution should be carried out multiple times (optimally, 30 runs per setting). 

• The list of seeds (used to initialise the random number generator before each run) should be stored & provided.
