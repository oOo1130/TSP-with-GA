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
            idx = idx+1
            cities.append(City(x, y,idx))
    return cities

def getfitness(route):
    sol = [cities[gi] for gi in route]
    return 1/path_cost(sol)


class Fitness:
    def __init__(self, route):
        self.route = route
        self.distance = 0
        self.fitness = 0.0

    def path_cost(self):
        if self.distance == 0:
            self.distance = sum([city.distance(self.route[index - 1]) for index, city in enumerate(self.route)])
        return self.distance

    def path_fitness(self):
        if self.fitness == 0:
            self.fitness = 1 / float(self.path_cost())
        return self.fitness

class GeneticAlgorithm:
    def __init__(self, iterations, population_size, cities, elites_num, mutation_rate,
                 greedy_seed=0, roulette_selection=True, plot_progress=True, pvalue = 20, bvalue = 30):

        self.plot_progress = plot_progress
        self.roulette_selection = roulette_selection
        self.progress = []
        self.mutation_rate = mutation_rate
        self.cities = cities
        self.elites_num = elites_num
        self.iterations = iterations
        self.population_size = population_size
        self.greedy_seed = greedy_seed

        self.population = self.initial_population()
        self.average_path_cost = 1
        self.ranked_population = None
        

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

    def selection(self):
        selections = [self.ranked_population[i][0] for i in range(self.elites_num)]
        if self.roulette_selection:
            df = pd.DataFrame(np.array(self.ranked_population), columns=["index", "fitness"])
            self.average_path_cost = sum(1 / df.fitness) / len(df.fitness)
            df['cum_sum'] = df.fitness.cumsum()
            df['cum_perc'] = 100 * df.cum_sum / df.fitness.sum()

            for _ in range(0, self.population_size - self.elites_num):
                pick = 100 * random.random()
                for i in range(0, len(self.ranked_population)):
                    if pick <= df.iat[i, 3]:
                        selections.append(self.ranked_population[i][0])
                        break
        else:
            for _ in range(0, self.population_size - self.elites_num):
                pick = random.randint(0, self.population_size - 1)
                selections.append(self.ranked_population[pick][0])
        self.population = selections

    def produce_child(parent1, parent2):
        gene_1 = random.randint(0, len(parent1))
        gene_2 = random.randint(0, len(parent1))
        gene_1, gene_2 = min(gene_1, gene_2), max(gene_1, gene_2)
        child = [parent1[i] for i in range(gene_1, gene_2)]
        child.extend([gene for gene in parent2 if gene not in child])
        return child

    def generate_population(self):
        length = len(self.population) - self.elites_num
        children = self.population[:self.elites_num]
        for i in range(0, length):
            child = self.produce_child(self.population[i],
                    self.population[(i + random.randint(1, self.elites_num)) % length])
            children.append(child)
        return children

    def mutate(self, individual):
        for index, city in enumerate(individual):
            if random.random() < max(0, self.mutation_rate):
                sample_size = min(min(max(3, self.population_size // 5), 100), len(individual))
                random_sample = random.sample(range(len(individual)), sample_size)
                sorted_sample = sorted(random_sample,
                                       key=lambda c_i: individual[c_i].distance(individual[index - 1]))
                random_close_index = random.choice(sorted_sample[:max(sample_size // 3, 2)])
                individual[index], individual[random_close_index] = \
                    individual[random_close_index], individual[index]
        return individual




    def next_generation(self):
        self.rank_population()
        
        self.selection()
        self.population = self.generate_population()
        self.population[self.elites_num:] = [self.mutate(chromosome)
                                             for chromosome in self.population[self.elites_num:]]

    def run(self):
        if self.plot_progress:
            plt.ion()
        for ind in range(0, self.iterations):
            self.next_generation()
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
        fig = plt.figure(1)
        fig.clear()
        fig.suptitle('genetic algorithm TSP')
        plt.plot(x_list, y_list, 'ro')
        plt.plot(x_list, y_list, 'g')

        if self.plot_progress:
            plt.draw()
            plt.pause(0.05)
        plt.show()


def greedy_route(start_index, cities):
    unvisited = cities[:]
    del unvisited[start_index]
    route = [cities[start_index]]
    while len(unvisited):
        index, nearest_city = min(enumerate(unvisited), key=lambda item: item[1].distance(route[-1]))
        route.append(nearest_city)
        del unvisited[index]
    return route

def path_cost(route):
    return sum([city.distance(route[index - 1]) for index, city in enumerate(route)])

def solve_tsp_dynamic(cities):
    distance_matrix = [[x.distance(y) for y in cities] for x in cities]
    cities_a = {(frozenset([0, idx + 1]), idx + 1): (dist, [0, idx + 1]) for idx, dist in
                enumerate(distance_matrix[0][1:])}
    for m in range(2, len(cities)):
        cities_b = {}
        for cities_set in [frozenset(C) | {0} for C in itertools.combinations(range(1, len(cities)), m)]:
            for j in cities_set - {0}:
                cities_b[(cities_set, j)] = min([(cities_a[(cities_set - {j}, k)][0] + distance_matrix[k][j],
                                                  cities_a[(cities_set - {j}, k)][1] + [j])
                                                 for k in cities_set if k != 0 and k != j])
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
                DValue.append([ivalue, jvalue])
    return DValue 

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
    D = []
    print(fnGetPartialOrder(sigma1,sigma2))

if __name__ == '__main__':
    # main()
    # cities = read_cities()#size = 16
    # g = solve_tsp_dynamic(cities)
    # sol = [cities[gi] for gi in g]
    # sol_path = [cities[gi].cityindex for gi in g]
    # print(sol_path)
    # print(path_cost(sol))
    cities = read_cities()
    genetic_algorithm = GeneticAlgorithm(cities=cities, iterations=1200, population_size=100,
                                         elites_num=20, mutation_rate=0.008, greedy_seed=1,
                                         roulette_selection=True, plot_progress=True)
    genetic_algorithm.run()
    print(genetic_algorithm.best_distance())
    genetic_algorithm.plot()
    plt.show(block=True)
    visualize_tsp("",genetic_algorithm.best_chromosome())