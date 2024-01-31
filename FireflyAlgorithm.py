import numpy as np

def objective_function(x, y):
    return x**2 + y**2 + np.cos(y)

def is_within_bounds(x, y):
    return x**2 + y**2 <= 1

def initialize_fireflies(num_fireflies, bounds):
    return np.random.uniform(bounds[0], bounds[1], (num_fireflies, 2))

def attractiveness(distance):
    beta = 1  
    return np.exp(-beta * distance*distance)

def move_firefly(current, other, alpha,generation):
    r = np.linalg.norm(other - current)
    beta_i = attractiveness(r)
    return current + beta_i * (other - current) + alpha * (np.random.rand(2) - 0.5)**generation

def firefly_algorithm(num_fireflies, max_generations, bounds):
    alpha = 0.2   
    

    fireflies = initialize_fireflies(num_fireflies, bounds)

    for generation in range(max_generations):
        for i in range(num_fireflies):
            for j in range(num_fireflies):
                if i != j and is_within_bounds(fireflies[i, 0], fireflies[i, 1]):
                    if objective_function(fireflies[j, 0], fireflies[j, 1]) < objective_function(fireflies[i, 0], fireflies[i, 1]):
                        fireflies[i] = move_firefly(fireflies[i], fireflies[j], alpha,generation)

    best_firefly = fireflies[np.argmin([objective_function(x, y) for x, y in fireflies])]

    return best_firefly

# Example usage
num_fireflies = 20
max_generations = 40
bounds = [-1, 1]

best_solution = firefly_algorithm(num_fireflies, max_generations, bounds)
print("Best solution:", best_solution)
print("Minimum value:", objective_function(*best_solution))
