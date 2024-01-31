import numpy as np

# Objective function
def objective_function(x, y):
    return x**2 + y**2 + np.cos(y)

# Bat algorithm
def bat_algorithm(num_bats, num_iterations, Loudness, Pulse_rate, alpha, gamma, f_min, f_max):
    bats = np.random.rand(num_bats, 2)  
    velocities = np.zeros((num_bats, 2))
    loudness = np.random.uniform(0, Loudness, (num_bats, 1))
    pulserate = np.random.uniform(0, Pulse_rate, (num_bats, 1))
    fitness = np.array([objective_function(x, y) for x, y in bats])

    # Find the index of the bat with the best fitness
    best_bat_index = np.argmin(fitness)
    best_bat = bats[best_bat_index]
    f_best = fitness[best_bat_index]

    for t in range(num_iterations):
        # Update bat positions and velocities
        frequencies = f_min + (f_max - f_min) * np.random.rand(num_bats)
        velocities += (bats - best_bat) * frequencies[:, np.newaxis]
        new_bats = bats + velocities

        for i in range(num_bats):
            if np.random.rand() > pulserate[i]:
                new_bats[i] = best_bat + alpha * np.random.normal(0, 1, 2)

            new_bats[i] = np.clip(new_bats[i], -1, 1)

        # Evaluate new solutions
        new_fitness = np.array([objective_function(x, y) for x, y in new_bats])

        # Update the best solution
        if np.min(new_fitness) < f_best:
            best_bat_index = np.argmin(new_fitness)
            best_bat = new_bats[best_bat_index]
            f_best = new_fitness[best_bat_index]

        # Update bats based on the fitness values
        if np.random.rand() < loudness[i] and objective_function(new_bats[i,0],new_bats[i,1])<objective_function(best_bat[0],best_bat[1]):
            # Update Loudness and Pulserates
            Loudness=Loudness*alpha
            Pulse_rate=Pulse_rate*(1-np.exp(-gamma*t))
            bats = new_bats
            fitness = new_fitness


        
        
    return best_bat, f_best

# Main function
if __name__ == "__main__":
    num_bats = 40
    num_iterations = 100

    Loudness = 1 # Loudness
    Pulse_rate = 1 # Pulse rate
    alpha = 0.9  # Alpha parameter
    gamma = 0.9  # Gamma parameter
    f_min, f_max = 0, 1  # Frequency range

    best_solution, min_value = bat_algorithm(num_bats, num_iterations, Loudness, Pulse_rate, alpha, gamma, f_min, f_max)

    print(f"Best Solution: {best_solution}")
    print(f"Minimum Value: {min_value}")
