from Genetic_Algorithm import *
from Snake_Game import *
import matplotlib.pyplot as plt
import time

# n_x -> no. of input units
# n_h -> no. of units in hidden layer 1
# n_h2 -> no. of units in hidden layer 2
# n_y -> no. of output units

# hyperparameters
# The population will have sol_per_pop chromosome where each chromosome has num_weights genes.
sol_per_pop = 50
num_generations = 100
crossover_percentage = 0.2
mutation_intensity = 0.01

datetimeCurr = str(time.strftime("%Y%m%d-%H%M%S"))
filename = "output"+datetimeCurr+".txt"
graphname = "graph"+datetimeCurr+".png"

num_parents_mating = (int)(crossover_percentage*sol_per_pop) # has to be even
num_weights = n_x*n_h + n_h*n_h2 + n_h2*n_y
max_fitness = []

# Defining the population size.
pop_size = (sol_per_pop, num_weights)
#Creating the initial population.
new_population = np.random.choice(np.arange(-1, 1, step = 0.01), size = pop_size, replace=True)

file1 = open(filename, "w")
file1.close()

for generation in range(num_generations):
    file1 = open(filename, "a+")
    file1.write("##############        GENERATION " + str(generation)+ "  ############### \n")
    file1.close()
    # print('##############        GENERATION ' + str(generation)+ '  ###############' )
    
    # Measuring the fitness of each chromosome in the population.
    fitness = cal_pop_fitness(new_population, filename)
    max_fitness.append(np.max(fitness))
    
    file1 = open(filename, "a+")
    file1.write("#######  fittest chromosome in generation " + str(generation) + " is having fitness value:  " + str(np.max(fitness)) + "\n")
    file1.close()
    # print('#######  fittest chromosome in generation ' + str(generation) + ' is having fitness value:  ', np.max(fitness))
    
    # Selecting the best parents in the population for mating.
    parents = select_mating_pool(new_population, fitness, num_parents_mating)
    # Generating next generation using crossover.
    offspring_crossover = crossover(parents, offspring_size = (pop_size[0] - parents.shape[0], num_weights))
    # Adding some variations to the offsrping using mutation.
    offspring_mutation = mutation(offspring_crossover, mutation_intensity)
    # Creating the new population based on the parents and offspring.
    new_population[0:parents.shape[0], :] = parents
    new_population[parents.shape[0]:, :] = offspring_mutation
    
gen_count = list(range(1, num_generations+1))
#Plotting Graph
plt.plot(gen_count, max_fitness )
plt.xlabel('Generation count')
plt.ylabel('Max Fitness value')
plt.title('Plot')
#plt.show()
plt.savefig(graphname)