import math
import random
import itertools
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
import math, sys



class City:
    def __init__(self, x, y, index):
        self.x = x
        self.y = y
        self.cityindex = index
    def distance(self, city):
        return math.hypot(self.x - city.x, self.y - city.y)
def read_cities():
    cities = []
    idx = 0
    with open(f'test.dat', 'r') as handle:
        lines = handle.readlines()
        for line in lines:
            x, y = map(float, line.split())
            cities.append(City(x, y,idx))
            idx = idx + 1
    return cities

def getfitness(route):
    sol = [cities[gi] for gi in route]
    return 1/path_cost(sol)

def path_cost(route):
    return sum([city.distance(route[index - 1]) for index, city in enumerate(route)])
class Fitness:
    def __init__(self, route):
        self.route = route
        self.distance = 0
        self.fitness = 0.0

    def path_cost(self):
        if self.distance == 0:
            self.distance = sum([city.distance(self.route[index]) for index, city in enumerate(self.route)])
        return self.distance

    def path_fitness(self):
        if self.fitness == 0:
            self.fitness = 1 / float(self.path_cost())
        return self.fitness

class GeneticDPForTSP:
    def __init__(self, iterations, population_size, cities,
                 greedy_seed=0, plot_progress=True, pvalue = 20, bvalue = 30, avalue = 0.5):

        self.plot_progress = plot_progress
        self.progress = []
        self.cities = cities
        self.iterations = iterations
        self.population_size = population_size
        self.greedy_seed = greedy_seed

        self.population = self.initial_population()
        self.distance_matrix = [[x.distance(y) for y in cities] for x in cities]
        self.ranked_population = None
        self.pvalue = pvalue
        self.bvalue = bvalue
        self.avalue = avalue

    def best_chromosome(self):
        return self.ranked_population[0][0]

    def best_distance(self):
        return 1 / self.ranked_population[0][1]

    def random_route(self):

        return random.sample(self.cities, len(self.cities))

    def initial_population(self):
        p1 = [self.random_route() for _ in range(self.population_size - self.greedy_seed)]
        greedy_population = [greedy_route(start_index % len(self.cities), self.cities)
                             for start_index in range(self.greedy_seed)]
        return [*p1, *greedy_population]

    def rank_population(self):
        fitness = [(chromosome, Fitness(chromosome).path_fitness()) for chromosome in self.population]
        self.ranked_population = sorted(fitness, key=lambda f: f[1], reverse=True)
    

    # partial order D
    def findPartialOrderSet(self,chromosome1,chromosome2):
        sigma1 = [(city.cityindex) for city in chromosome1]
        sigma2 = [(city.cityindex) for city in chromosome2]
        DValue = []
        for idx in range(0, len(sigma1)):
            for jdx in range(idx, len(sigma1)):
                ivalue = sigma1[idx]
                jvalue = sigma1[jdx]
                if(sigma2.index(ivalue) <= sigma2.index(jvalue)):
                    DValue.append((idx, jdx))
        frozen =  set(frozenset(pair)  for pair in DValue)  
        return frozen 

    # 
    def selection(self,chromosome1,chromosome2,chromosomenew):
        sig1 = path_cost(chromosome1)
        sig2 = path_cost(chromosome2)
        signew = path_cost(chromosomenew) 
        delta1 = sig1 - signew
        delta2 = sig2 - signew
        xvalue = 0
        if delta1 == delta2 and delta2 == 0:
            xvalue = 1
        else:
            xvalue = delta1/delta2
        prob = min(xvalue/self.avalue,1)
        if prob >= 0.5:
            return chromosome1,chromosomenew
        else:
            return chromosomenew,chromosome2

    #step2        
    def crossoverOneStepByDP(self,chromosome1,chromosome2):
        
        DSet = self.findPartialOrderSet(chromosome1,chromosome2)
        
        distance_matrix = self.distance_matrix
        cities_a = {(frozenset([0, idx + 1]), idx + 1): (dist, [0, idx + 1]) for idx, dist in
                    enumerate(distance_matrix[0][1:])}
        
        cities = self.cities        
        #crossover by DP
        for m in range(2, len(cities)):
            cities_b = {}
            for cities_set in [frozenset(C) | {0} for C in itertools.combinations(range(1, len(cities)), m)]:
                
                pairset  = set( frozenset(pair)  for pair in itertools.combinations(cities_set,2))
                
                if(pairset.issubset(DSet) == False): 
                    continue
                for i in cities_set - {0}:
                    cities_b[(cities_set, i)] = min([(cities_a[(cities_set - {i}, j)][0] + distance_matrix[j][i],
                                                    cities_a[(cities_set - {i}, j)][1] + [i])
                                                    for j in cities_set if j != 0 and j != i])
            cities_a = cities_b

        resultarr = [(cities_a[d][0] + distance_matrix[0][d[1]], cities_a[d][1]) for d in iter(cities_a)]
        if(len(resultarr) == 0):
            return chromosome1,chromosome2
        res = min(resultarr)
        sol = [cities[gi] for gi in res[1]]
        
        return self.selection(chromosome1,chromosome2,sol)


    def crossoverByDP(self):
        self.population = [gene[0] for gene in self.ranked_population]
        for idx in range(0,len(self.population),2):
            self.population[idx],self.population[idx+1] = self.crossoverOneStepByDP(self.population[idx],self.population[idx+1])


    def next_generation(self):
        self.rank_population()
        
        

    def run(self):
        if self.plot_progress:
            plt.ion()
        
        for ind in range(0, self.iterations):
            self.next_generation()
            self.crossoverByDP()
            self.progress.append(self.best_distance())
            if self.plot_progress and ind % 10 == 0:
                self.plot()
            elif not self.plot_progress and ind % 10 == 0:
                print(self.best_distance())

    def plot(self):
        print(self.best_distance())
        fig = plt.figure(0)
        plt.plot(self.progress, 'g')
        fig.suptitle('genetic algorithm generations')
        plt.ylabel('Distance')
        plt.xlabel('Generation')

        x_list, y_list = [], []
        for city in self.best_chromosome():
            x_list.append(city.x)
            y_list.append(city.y)
        x_list.append(self.best_chromosome()[0].x)
        y_list.append(self.best_chromosome()[0].y)
        # fig = plt.figure(1)
        # fig.clear()
        # fig.suptitle('genetic algorithm TSP')
        # plt.plot(x_list, y_list, 'ro')
        # plt.plot(x_list, y_list, 'g')

        # if self.plot_progress:
        #     plt.draw()
        #     plt.pause(0.05)
        # plt.show()


def greedy_route(start_index, cities):
    unvisited = cities[:]
    del unvisited[start_index]
    route = [cities[start_index]]
    while len(unvisited):
        index, nearest_city = min(enumerate(unvisited), key=lambda item: item[1].distance(route[-1]))
        route.append(nearest_city)
        del unvisited[index]
    return route



def solve_tsp_dynamic(cities):
    distance_matrix = [[x.distance(y) for y in cities] for x in cities]
    cities_a = {(frozenset([0, idx + 1]), idx + 1): (dist, [0, idx + 1]) for idx, dist in
                enumerate(distance_matrix[0][1:])}
    
               
    for m in range(2, len(cities)):
        cities_b = {}
        for cities_set in [frozenset(C) | {0} for C in itertools.combinations(range(1, len(cities)), m)]:
            
            for i in cities_set - {0}:
                cities_b[(cities_set, i)] = min([(cities_a[(cities_set - {i}, j)][0] + distance_matrix[j][i],
                                                  cities_a[(cities_set - {i}, j)][1] + [i])
                                                 for j in cities_set if j != 0 and j != i])
        cities_a = cities_b
        
    res = min([(cities_a[d][0] + distance_matrix[0][d[1]], cities_a[d][1]) for d in iter(cities_a)])
    return res[1]


def fnGetPartialOrder(sigma1, sigma2):
    DValue = []

    for idx in range(0, len(sigma1)):
        for jdx in range(idx, len(sigma1)):
            ivalue = sigma1[idx]
            jvalue = sigma1[jdx]
            if(sigma2.index(ivalue) <= sigma2.index(jvalue)):
                DValue.append((idx+1, jdx+1))
    frozen =  set(frozenset(pair)  for pair in DValue)           
    return frozen 

def visualize_tsp(title, cities):
    fig = plt.figure()
    fig.suptitle(title)
    x_list, y_list = [], []
    for city in cities:
        x_list.append(city.x)
        y_list.append(city.y)
    x_list.append(cities[0].x)
    y_list.append(cities[0].y)

    plt.plot(x_list, y_list, 'ro')
    plt.plot(x_list, y_list, 'g')
    plt.show(block=True)

def testD():
    
    sigma1 = [6,5,4,3,2,1]
    sigma2 = [2,3,1,5,4,6]
    
    DSet = (fnGetPartialOrder(sigma1,sigma2))
    
if __name__ == '__main__':
    
    cities = read_cities()
    genetic_algorithm = GeneticDPForTSP(cities=cities, iterations=100, population_size=20,
                                         greedy_seed=1, plot_progress=True)
    genetic_algorithm.run()
    print(genetic_algorithm.best_distance())
    