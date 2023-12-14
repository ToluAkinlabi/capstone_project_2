##############################################################################
# Based on code from the 'Hands-On-Genetic-Algorithms-with-Python' repository
# Author: Eyal Wirsansky, ai4java
# Link: https://github.com/PacktPublishing/Hands-On-Genetic-Algorithms-with-Python/tree/master/Chapter08
# Link: https://github.com/PacktPublishing/Hands-On-Genetic-Algorithms-with-Python/tree/master/Chapter09
##############################################################################

# 3rd party
try:
    from deap import base
    from deap import creator
    from deap import tools
except:
    print('pip install deap\n^^^ Deap is required to run the genetic algorithm search.')
    exit(1)
import numpy
import matplotlib.pyplot as plt
import seaborn as sns

# standard library
import random
import time
import multiprocessing

# internal imports
try:
    import models_encapsulated
    import elitism_selection
# handle PATH issues when calling from a different directory
except:
    import exploratory_analysis.models_encapsulated as models_encapsulated
    import exploratory_analysis.elitism_selection as elitism_selection


# boundaries for RandomForestClassifier
# min_samples_leaf: 1-25
# max_features: 0.01-1.0
# max_depth: 1-20
# n_estimators: 50-200
# [min_samples_leaf, max_features, max_depth, n_estimators]
BOUNDS_LOW_RF: list  = [ 1, 0.01,  1,  50]
BOUNDS_HIGH_RF: list = [25, 1.00, 20, 200]

# boundaries for DecisionTreeClassifier
# max_depth: 1-25
# min_samples_split: 0.01-1.0
# min_samples_leaf: 1-30
# max_features: 0.01-1.0
# min_impurity_decrease: 0.0001-0.03
# [max_depth, min_samples_split, min_samples_leaf, max_features, min_impurity_decrease]
BOUNDS_LOW_TREE: list  = [ 1, 0.01,  1, 0.01, 0.0003]
BOUNDS_HIGH_TREE: list = [25, 1.00, 30, 1.00, 0.0300]

# boundaries for all MLP parameters:
# 'hidden_layer_sizes': first three values
# 'activation': ['tanh', 'relu', 'logistic'] -> 0, 1, 2
# 'alpha': float in the range of [0.0001, 2.0]
BOUNDS_LOW_NN =  [ 3, -1, -10, 0,     0.0001]
BOUNDS_HIGH_NN = [20, 10,  10, 2.999, 2.0000]

# genetic algorithm search:
def search_parameters(MODEL_TYPE: str='random forest',  # random forest, neural networks, decision tree
                      BOUNDS_LOW: list=BOUNDS_LOW_RF, BOUNDS_HIGH: list=BOUNDS_HIGH_RF,
                      POPULATION_SIZE: int=20,
                      P_CROSSOVER: float=0.9,       # probability for crossover
                      P_MUTATION: float=0.6,        # probability for mutating an individual
                      MAX_GENERATIONS: int=5,
                      HALL_OF_FAME_SIZE: int=5,
                      CROWDING_FACTOR: int=20.0,    # crowding factor for crossover and mutation
                      RANDOM_SEED: int=42):
    assert len(BOUNDS_LOW) == len(BOUNDS_HIGH), 'Lower and upper bounds are of different length'

    # create the classifier accuracy test class:
    if MODEL_TYPE.lower() == 'random forest':
        if len(BOUNDS_LOW_RF) != len(BOUNDS_LOW):
            print('Resetting to default random forest bounds because passed bounds do not match random forest parameters')
            BOUNDS_LOW = BOUNDS_LOW_RF
            BOUNDS_HIGH = BOUNDS_HIGH_RF
        test = models_encapsulated.HyperparameterTuningRandomForest(RANDOM_SEED)
    elif MODEL_TYPE.lower() == 'neural networks':
        if len(BOUNDS_LOW_NN) != len(BOUNDS_LOW):
            print('Resetting to default MLP bounds because passed bounds do not match MLP parameters')
            BOUNDS_LOW = BOUNDS_LOW_NN
            BOUNDS_HIGH = BOUNDS_HIGH_NN
        test = models_encapsulated.HyperparameterTuningMlp(RANDOM_SEED)
    elif MODEL_TYPE.lower() == 'decision tree':
        if len(BOUNDS_LOW_TREE) != len(BOUNDS_LOW):
            print('Resetting to default decision tree bounds because passed bounds do not match decision tree parameters')
            BOUNDS_LOW = BOUNDS_LOW_TREE
            BOUNDS_HIGH = BOUNDS_HIGH_TREE
        test = models_encapsulated.HyperparameterTuningDecisionTree(RANDOM_SEED)
    else:
        raise Exception(f'Model type "{MODEL_TYPE}" is not valid for genetic algorithm search')

    # get number of parameters
    NUM_OF_PARAMS = len(BOUNDS_HIGH)
    # set the random seed:
    random.seed(RANDOM_SEED)
    
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
    print("---Best Solution---")
    print("params: ", test.formatParams(hof.items[0]))
    print("accuracy: %1.3f" % hof.items[0].fitness.values[0])

    # remove functions from creator
    del creator.FitnessMax
    del creator.Individual

if __name__ == "__main__":
    search_parameters()