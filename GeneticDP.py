def initializePopulation(self):
        p1 = [self.random_route() for _ in range(self.size_population - self.greedy_seed)]
        greedy_population = [greedy_route(start_index % len(self.cities), self.cities)
                             for start_index in range(self.greedy_seed)]
        return [*p1, *greedy_population]

    def rankPopulation(self):
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

    # selection
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
                                                    for j in cities_set if j != 0 and j != i and frozenset((i,j)) in DSet])
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

    def nextGeneration(self):
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