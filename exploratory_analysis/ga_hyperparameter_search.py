##############################################################################
# Based on code from the 'Hands-On-Genetic-Algorithms-with-Python' repository
# https://github.com/PacktPublishing/Hands-On-Genetic-Algorithms-with-Python/tree/master/Chapter08
##############################################################################

# 3rd party
from deap import base
from deap import creator
from deap import tools
import numpy
import matplotlib.pyplot as plt
import seaborn as sns
from scoop import futures

# standard library
import random
import time
import multiprocessing

# internal imports
import random_forest_encapsulated
import elitism_selection


# boundaries for RandomForestClassifier
# min_samples_leaf: 0-25
# max_features: 0.0-1.0
# max_depth: 0-20
# n_estimators: 50-200
# [min_samples_leaf, max_features, max_depth, n_estimators]
BOUNDS_LOW =  [  1, 0.01,  1, 50  ]
BOUNDS_HIGH = [ 25, 1.00, 20, 200 ]

NUM_OF_PARAMS = len(BOUNDS_HIGH)

# Genetic Algorithm constants:
POPULATION_SIZE = 20
P_CROSSOVER = 0.9  # probability for crossover
P_MUTATION = 0.6   # probability for mutating an individual
MAX_GENERATIONS = 5
HALL_OF_FAME_SIZE = 5
CROWDING_FACTOR = 20.0  # crowding factor for crossover and mutation

# set the random seed:
RANDOM_SEED = 53
random.seed(RANDOM_SEED)

# create the classifier accuracy test class:
test = random_forest_encapsulated.HyperparameterTuningGenetic(RANDOM_SEED)

toolbox = base.Toolbox()

# define a single objective, maximizing fitness strategy:
creator.create("FitnessMax", base.Fitness, weights=(1.0,))

# create the Individual class based on list:
creator.create("Individual", list, fitness=creator.FitnessMax)

# define the hyperparameter attributes individually:
for i in range(NUM_OF_PARAMS):
    # "hyperparameter_0", "hyperparameter_1", ...
    toolbox.register("hyperparameter_" + str(i),
                     random.uniform,
                     BOUNDS_LOW[i],
                     BOUNDS_HIGH[i])

# create a tuple containing an attribute generator for each param searched:
hyperparameters = ()
for i in range(NUM_OF_PARAMS):
    hyperparameters = hyperparameters + \
                      (toolbox.__getattribute__("hyperparameter_" + str(i)),)

# create the individual operator to fill up an Individual instance:
toolbox.register("individualCreator",
                 tools.initCycle,
                 creator.Individual,
                 hyperparameters,
                 n=1)

# create the population operator to generate a list of individuals:
toolbox.register("populationCreator", tools.initRepeat, list, toolbox.individualCreator)

# fitness calculation
def classificationAccuracy(individual):
    return test.getAccuracy(individual),

toolbox.register("evaluate", classificationAccuracy)

# genetic operators:mutFlipBit

# genetic operators:
toolbox.register("select", tools.selTournament, tournsize=2)
toolbox.register("mate",
                 tools.cxSimulatedBinaryBounded,
                 low=BOUNDS_LOW,
                 up=BOUNDS_HIGH,
                 eta=CROWDING_FACTOR)

toolbox.register("mutate",
                 tools.mutPolynomialBounded,
                 low=BOUNDS_LOW,
                 up=BOUNDS_HIGH,
                 eta=CROWDING_FACTOR,
                 indpb=1.0 / NUM_OF_PARAMS)

# make map operations parallel
#toolbox.register('map', futures.map)
#toolbox.register('map', multiprocessing.Pool().map)

# Genetic Algorithm flow:
def main():

    # prepare the statistics object:
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("max", numpy.max)
    stats.register("avg", numpy.mean)

    # define the hall-of-fame object:
    hof = tools.HallOfFame(HALL_OF_FAME_SIZE)

    # start timer
    start_time = time.perf_counter()
    
    # create initial population (generation 0):
    population = toolbox.populationCreator(n=POPULATION_SIZE)

    # prepare the statistics object:
    # stats = tools.Statistics(lambda ind: ind.fitness.values)
    # stats.register("max", numpy.max)
    # stats.register("avg", numpy.mean)

    # # define the hall-of-fame object:
    # hof = tools.HallOfFame(HALL_OF_FAME_SIZE)

    # perform the Genetic Algorithm flow with hof feature added:
    population, logbook = elitism_selection.eaSimpleWithElitism(population,
                                                                toolbox,
                                                                cxpb=P_CROSSOVER,
                                                                mutpb=P_MUTATION,
                                                                ngen=MAX_GENERATIONS,
                                                                stats=stats,
                                                                halloffame=hof,
                                                                verbose=True)

    # get wall time measure of efficiency
    print(f'Elapsed seconds: {time.perf_counter() - start_time:.1f}')
    
    # print best solution found:
    print("- Best solution is: ")
    print("params = ", test.formatParams(hof.items[0]))
    print("Accuracy = %1.5f" % hof.items[0].fitness.values[0])

    # extract statistics:
    # maxFitnessValues, meanFitnessValues = logbook.select("max", "avg")
    

    # plot statistics:
    # sns.set_style("whitegrid")
    # plt.plot(maxFitnessValues, color='red')
    # plt.plot(meanFitnessValues, color='green')
    # plt.xlabel('Generation')
    # plt.ylabel('Max / Average Fitness')
    # plt.title('Max and Average fitness over Generations')
    # plt.show()


if __name__ == "__main__":
    main()