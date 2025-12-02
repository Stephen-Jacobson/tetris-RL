import tensorflow as tf
import random
import numpy as np
from Tetris import TetrisEnv
import heapq



def create_model(input_size, output_size, max_layers = 3):
    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.Input(input_size))

    num_layers = max_layers

    neurons = [64, 32, 32]
    activation = ["relu", "sigmoid", "tanh"]
    for i in range(num_layers):
        model.add(tf.keras.layers.Dense(neurons[i], activation = random.choice(activation)))

    model.add(tf.keras.layers.Dense(output_size, activation = "linear"))

    return model

def play_game(model, render = False):
    env = TetrisEnv()
    clears = 0
    total_reward = 0
    not_done = True
    steps = 0
    while not_done:
        reward = 0
        state = []
        next_pieces = []
        for i in range(len(env.next_pieces)):
            next_pieces.append(((env.next_pieces[i] / 7) * 2) - 1)
            state.append(next_pieces[i])
        if env.held_piece == None:
            held_piece = -1
        else:
            held_piece = ((env.held_piece.type / 7) * 2) - 1
        cur_piece = ((env.cur_piece.type / 7) * 2) - 1
        bumpiness = ((env.get_bumpiness() / 216) * 2) - 1
        holiness = ((env.get_holes() / 230) * 2) - 1
        agg_height = ((env.get_aggregate_height() / 240) * 2) - 1
        t_spin = env.tslot_exists()
        if t_spin:
            t_spin = 1
        else: 
            t_spin = 0
        t_spin = ((t_spin / 1) * 2) - 1
        clears = ((clears / 4) * 2) - 1

        state.append(held_piece)
        state.append(cur_piece)
        state.append(bumpiness)
        state.append(holiness)
        state.append(agg_height)
        state.append(t_spin)
        state.append(clears)

        output = model(tf.convert_to_tensor([state], dtype=tf.float32))

        action = tf.argmax(output[0]).numpy()

        step = env.step(action)
        steps += 1

        if render:
            env.render()

        reward = step[1]

        reward = reward - 0.2*bumpiness - 0.3*holiness - 0.2*agg_height

        total_reward += reward

        clears = env.clears - clears
        not_done = not(step[2])
        if steps >= 100:
            not_done = False

    return total_reward

def mutate_model(model, mutation_rate = 0.1, mutation_strength = 0.2):
    for i in model.layers:
        weights = i.get_weights()
        if len(weights) > 0:
            new_weights = []
            for weight_matrix in weights: 
                mutation_mask = np.random.random(weight_matrix.shape) < mutation_rate
                
                mutations = np.random.normal(0, mutation_strength, weight_matrix.shape)
                
                mutated = weight_matrix + (mutations * mutation_mask)
                new_weights.append(mutated)
            
            i.set_weights(new_weights)
    return model



def evolve_population(input_size, output_size, pop_size, generations, best_fit):
    population = []
    for i in range(pop_size):
        population.append(create_model(input_size, output_size))

    for t in range(generations):
        fitness_scores = []
        best_scores = []
        print(f"Generation: {t}")
        for i in range(pop_size):
            print(i)
            fitness_scores.append(play_game(population[i], ((t + 1) // 10) == 0 and t != 0 and i == 0))

        best_scores = heapq.nlargest(best_fit, enumerate(fitness_scores), key=lambda x: x[1])

        best_models = []
        for i in range(len(best_scores)):
            best_models.append(population[best_scores[i][0]])
        new_population = []

        for k in range(pop_size // (best_fit // 2)):
            random.shuffle(best_models)
            for i in range(0, best_fit, 2):
                child = tf.keras.models.Sequential()
                child.add(tf.keras.layers.Input(input_size))
                model_a = best_models[i]
                model_b = best_models[i + 1]
                for j in range(3):
                    rand_num = random.randint(0, 1)
                    if rand_num == 0:
                        source = model_a.layers[j]
                    else:
                        source = model_b.layers[j]

                    child.add(tf.keras.layers.Dense(source.units, source.activation))
                    print(len(child.layers))
                    print(j)
                    child.layers[j].set_weights(source.get_weights())
                child.add(tf.keras.layers.Dense(output_size, activation = "linear"))

                mutate_model(child)

                new_population.append(child)
        population = new_population
    
if __name__ == "__main__":
    evolve_population(10, 8, 100, 100, 10)






    





