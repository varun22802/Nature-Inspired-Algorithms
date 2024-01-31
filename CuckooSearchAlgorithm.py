import numpy as np
import math

def objective_function(x, y):
    return x**2 + y**2 + np.cos(y)

def generate_initial_population(n):
    return np.random.rand(n, 2) * 2 - 1  # Random values between -1 and 1 for x and y

def levy_flight(scale, size):
    beta = 1.5
    sigma = (math.gamma(1 + beta) * np.sin(np.pi * beta / 2) / (math.gamma((1 + beta) / 2) * beta * 2**((beta - 1) / 2)))**(1 / beta)
    u = np.random.normal(0, sigma, size)
    v = np.random.normal(0, 1, size)
    step = u / abs(v)**(1 / beta)
    return scale * step

def evaluate_fitness(population):
    return np.array([objective_function(x, y) for x, y in population])

def cuckoo_search(max_generation, n, pa):
    population = generate_initial_population(n)
    fitness = evaluate_fitness(population)
    best_solution_index = np.argmin(fitness)
    best_solution = population[best_solution_index]

    for t in range(1, max_generation + 1):
        # Use Levy flight for generating new solutions
        levy_steps = levy_flight(0.01, (n, 2))
        new_solutions = best_solution + levy_steps

        new_fitness = evaluate_fitness(new_solutions)

        for i in range(n):
            j = np.random.randint(n)
            if new_fitness[i] < fitness[j]:
                population[j] = new_solutions[i]
                fitness[j] = new_fitness[i]

        # Abandon a fraction (Pa) of worse nests
        num_abandoned = int(pa * n)
        abandoned_indices = np.argsort(fitness)[:num_abandoned]
        population[abandoned_indices] = generate_initial_population(num_abandoned)

        # Keep the better solutions
        all_solutions = np.vstack([population, best_solution.reshape(1, 2)])
        all_fitness = evaluate_fitness(all_solutions)
        sorted_indices = np.argsort(all_fitness)
        population = all_solutions[sorted_indices[:n]]

        # Find the current best solution
        best_solution = population[0]

    return best_solution

if __name__ == "__main__":
    max_generation = 100
    n = 20
    pa = 0.2

    best_solution = cuckoo_search(max_generation, n, pa)

    print("Best solution:", best_solution)
    print("Minimum value of f(x, y):", objective_function(*best_solution))



