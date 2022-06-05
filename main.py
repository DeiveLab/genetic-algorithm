import random
import math
from matplotlib import pyplot as plt

class GenericAlgorithm():
    def __init__(self, pop_size, cross_prob, mut_prob, max_gen, variable_bits_size, num_of_variables, min_max_interval, function):
        self.POPULATION_SIZE = pop_size
        self.CROSSOVER_PROBABILITY = cross_prob
        self.MUTATION_PROBABILITY = mut_prob
        self.MAX_GENERATIONS = max_gen
        self.VARIABLE_BITS_SIZE = variable_bits_size
        self.NUM_OF_VARIABLES = num_of_variables
        self.FUNCTION = function
        self.PERSON_BIT_SIZE = variable_bits_size * num_of_variables
        self.MIN_MAX_INTERVAL = min_max_interval
        self.CURRENT_GENERATION = 0
        self.MEAN_FITNESS = []

    def get_real_value_from_chromosome(self, person_chromosome):
        min_value, max_value = self.MIN_MAX_INTERVAL
        return min_value + (max_value - min_value) * (self.extract_integer_value_from_chromosome(person_chromosome) / ((2 ** self.VARIABLE_BITS_SIZE) - 1))

    def extract_integer_value_from_chromosome(self, person_chromosome):
        result = 0
        for index, bit in enumerate(person_chromosome):
            result += 2**((len(person_chromosome) - 1) - index) if bit == 1 else 0
        return result

    def split_chromosome_variables(self, person_chromosome):
        return [person_chromosome[i:i + self.VARIABLE_BITS_SIZE] for i in range(0, len(person_chromosome), self.VARIABLE_BITS_SIZE)]

    def get_person_fitness(self, person_chromosome):
        return (1 / self.get_person_func_value(person_chromosome))
    
    def get_person_func_value(self, person_chromosome):
        variables = [self.get_real_value_from_chromosome(variable) for variable in self.split_chromosome_variables(person_chromosome)]
        return self.FUNCTION(*variables)

    def get_random_bit(self):
        return random.randint(0, 1)

    def generate_random_population(self):
        return [[self.get_random_bit() for _ in range(self.PERSON_BIT_SIZE)] for _ in range(self.POPULATION_SIZE)]

    def calculate_population_fitness(self, population):
        result = [self.get_person_fitness(person) for person in population]
        population_mean = sum(result)/len(result)
        self.MEAN_FITNESS.append(population_mean)
        return result

    def order_population_by_fitness(self, population):
        pop_fit = self.calculate_population_fitness(population)
        population = [person for _, person in sorted(zip(pop_fit, population))]
        population.reverse()
        pop_fit = sorted(pop_fit, reverse=True)
        return population, pop_fit

    def choose_parents_roulette(self, population, population_fitness):
        max = sum(population_fitness)
        selection_probs = [c/max for c in population_fitness]
        first_parent = random.choices(population, weights=selection_probs, k=1)[0]
        second_parent = random.choices(population, weights=selection_probs, k=1)[0]
        move_on = 0
        while(first_parent == second_parent and move_on < self.POPULATION_SIZE):
            move_on += 1
            second_parent = random.choices(population, weights=selection_probs, k=1)[0]
        if move_on == self.POPULATION_SIZE:
            return [first_parent, population[random.randint(1, self.POPULATION_SIZE - 1)]]
        return [first_parent, second_parent]
            

    def get_children_from_parents_simple_crossover(self, parents):
        result = []
        single_point = self.PERSON_BIT_SIZE//2
        for child_index in range(2):
            result.append([parents[0 + child_index][i] if i < single_point else parents[1 - child_index][i] for i in range(self.PERSON_BIT_SIZE)])    
        return result

    def get_children_from_parents_uniform_crossover(self, parents, random_mask):
        result = []
        for child_index in range(2):
            result.append([parents[0 + child_index][i] if random_mask[i] == 1 else parents[1 - child_index][i] for i in range(self.PERSON_BIT_SIZE)])    
        return result

    def generate_population_by_mutation(self, population):
        new_population = population.copy()
        for person_chromosome in new_population:
            for index, gene in enumerate(person_chromosome):
                if random.random() <= self.MUTATION_PROBABILITY:
                    person_chromosome[index] = abs(gene - 1)
        new_population.append(population[0])
        return new_population

    def generate_population_by_simple_crossover(self, population, population_fitness):
        new_population = []
        for _ in range(0, len(population), 2):
            if random.random() <= self.CROSSOVER_PROBABILITY:
                parents = self.choose_parents_roulette(population, population_fitness)
                children = self.get_children_from_parents_simple_crossover(parents)
                new_population = new_population + children
        new_population.append(population[0])
        return new_population

    def generate_population_by_uniform_crossover(self, population, population_fitness):
        new_population = []
        random_mask = [self.get_random_bit() for _ in range(self.PERSON_BIT_SIZE)]
        for _ in range(0, len(population), 2):
            if random.random() <= self.CROSSOVER_PROBABILITY:
                parents = self.choose_parents_roulette(population, population_fitness)
                children = self.get_children_from_parents_uniform_crossover(parents, random_mask)
                new_population = new_population + children
        new_population.append(population[0])
        return new_population

    def choose_fittest_people_from_populations(self, pop1, pop2, pop3, pop4):
        new_population = pop1 + pop2 + pop3 + pop4
        new_population, pop_fit = self.order_population_by_fitness(new_population)
        return new_population[:self.POPULATION_SIZE], pop_fit[:self.POPULATION_SIZE]

    def get_fittest_person(self, population):
        return self.get_real_value_from_chromosome(population[0])

    def get_best_variables(self):
        pop = self.generate_random_population()
        pop, pop_fit = self.order_population_by_fitness(pop)

        while(self.CURRENT_GENERATION < self.MAX_GENERATIONS):
            print('Geração ' + str(self.CURRENT_GENERATION) + ' de ' + str(self.MAX_GENERATIONS))
            pop1 = self.generate_population_by_simple_crossover(pop, pop_fit)
            pop2 = self.generate_population_by_uniform_crossover(pop, pop_fit)
            pop3 = self.generate_population_by_mutation(pop)
            pop4 = self.generate_random_population()

            pop, pop_fit = self.choose_fittest_people_from_populations(pop1, pop2, pop3, pop4)
            variables = [self.get_real_value_from_chromosome(variable) for variable in self.split_chromosome_variables(pop[0])]
            message = 'Fittest variable(s): ' + str(variables)
            print(message)

            self.CURRENT_GENERATION = self.CURRENT_GENERATION + 1

        plt.plot(range(self.MAX_GENERATIONS + 1), self.MEAN_FITNESS)
        plt.grid(True, zorder=0)
        plt.title("Media do fitness das populacoes")
        plt.xlabel("Geracao")
        plt.ylabel("Media da populacao")
        plt.show()
        return self.get_fittest_person(pop), pop

    def teste(self):
        while(self.CURRENT_GENERATION < self.MAX_GENERATIONS):
            print(self.CURRENT_GENERATION)
            self.CURRENT_GENERATION = self.CURRENT_GENERATION + 1

def gold(x, y):
    a = 1 + (x+y+1)**2*(19-14*x+3*x**2-14*y+6*x*y+3*y**2)
    b = 30 + (2*x-3*y)**2*(18-32*x+12*x**2+48*y-36*x*y+27*y**2)
    return a * b

def Ackley(x, y):
    variables = [x,y]
    n = 2
    a = 20
    b = 0.2
    c = 2 * math.pi
    s1 = 0
    s2 = 0
    for variable in variables:
        s1 = s1 + variable**2
        s2 = s2 + math.cos(c * variable)
    return -a * math.e**(-b * math.sqrt((1/n)*s1)) - math.e**((1/n)*s2) + a + math.e
    
def sumS(x, y):
    variables = [x, y]
    n=2
    s=0
    for j in range(1, n+1):
        s=s+j*(variables[j-1]**2)
    return s

def deJong(x, y):
    z = 100*(((x**2) - (y**2))**2) + ((1-(x**2))**2)
    return z

def rastrigin(x, y):
    return (x**2)+(y**2)-(10*math.cos(2*math.pi*x))-(10*math.cos(2*math.pi*y))+10

def bump(x, y):
    if (x*y) < 0.75:
        z = None
    elif (x+y > 7.5*2):
        z = None
    else:
        temp0 = math.pow(math.cos(x), 4) + math.pow(math.cos(y), 4)
        temp1 = 2 * math.pow(math.cos(x), 2) * math.pow(math.cos(y), 2)
        temp2 = math.sqrt(math.pow(x, 2) + 2 * math.pow(y, 2))
        z = -math.abs((temp0 - temp1) / temp2)
    return z

a = GenericAlgorithm(500, 0.8, 0.5, 10000, 32, 2, [-5, 5], rastrigin)

ue, pop = a.get_best_variables()
variables = [a.get_real_value_from_chromosome(variable) for variable in a.split_chromosome_variables(pop[0])]
print(variables)
# print(pop)

# a.teste()
