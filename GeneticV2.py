import tensorflow as tf
import random
import numpy as np
from Tetris import TetrisEnv
import heapq
import multiprocessing as mp
import os
import pygame
import time


def create_model(input_size, output_size, max_layers = 3):
    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.Input(shape=(input_size,)))

    num_layers = max_layers

    neurons = [32, 32, 32]
    activation = ["tanh"]
    for i in range(num_layers):
        model.add(tf.keras.layers.Dense(neurons[i], activation = random.choice(activation)))

    model.add(tf.keras.layers.Dense(output_size, activation = "linear"))

    return model

def play_game(model, render=False):
    env = TetrisEnv()
    env.genetic = True
    if render:
        env.render()
    done = False
    final_points = 0
    while not(done):
        # if env.level > 20:
        #     env.step(7)
        every_move = env.get_every_move()
        all_moves = every_move[0]
        best_move = []
        best_score = None
        list_pos = 0
        for i in range(len(all_moves)):
            t0 = time.perf_counter()

            state = env.simulate_conditions(all_moves[i])
            t1 = time.perf_counter()

            state = np.array(state, dtype=np.float32).reshape(1, -1)
            t2 = time.perf_counter()

            output = model(state)[0][0]
            output = float(output.numpy())
            t3 = time.perf_counter()

            # if i % 50 == 0:   # don’t spam every iteration
                # print(
                #     f"i={i} | "
                #     f"simulate={t1-t0:.5f}s | "
                #     f"numpy={t2-t1:.5f}s | "
                #     f"model={t3-t2:.5f}s"
                # )

            if best_score is None or output >= best_score:
                best_score = output
                best_move = all_moves[i]
                list_pos = i

        env.tslots = every_move[1][list_pos]

        if best_move != []:
            for i in range(len(best_move[0])):
                if env.placed:
                    env.placed = False
                    step = env.step(7)
                    break
                else:
                    step = env.step(best_move[0][i])
                if step[2] != ():
                    if step[2] != best_move[3]:
                        print("Final pos does NOT MATCH")
                        print(f"Actual pos: {step[2]}")
                        print(f"Expected pos: {best_move[3]}")
                        print(best_move[0])
                        time.sleep(5)
                if render:
                    env.render()
                done = step[0]
                if done:
                    final_points = step[1]
        else:
            done = True

    return final_points, best_score, best_move

def mutate_model(layer, mutation_rate = 0.5, mutation_strength = 0.001):
    weights = layer.get_weights()
    if len(weights) > 0:
        new_weights = []
        for weight_matrix in weights: 
            mutation_mask = np.random.random(weight_matrix.shape) < mutation_rate
            
            mutations = np.random.normal(0, mutation_strength, weight_matrix.shape)
            
            mutated = weight_matrix + (mutations * mutation_mask)
            new_weights.append(mutated)
        
        layer.set_weights(new_weights)
def crossover_weights(weights_a, weights_b):
    """Uniform crossover on weight arrays. Assumes weights_a and weights_b have same shapes."""
    child_w = []
    for wa, wb in zip(weights_a, weights_b):
        mask = np.random.rand(*wa.shape) < 0.5
        child = np.where(mask, wa, wb)
        # a small chance to average instead of pick
        if np.random.rand() < 0.05:
            child = 0.5 * (wa + wb)
        child_w.append(child.astype(np.float32))
    return child_w

def mutate_weights(weights, mutation_rate=0.05, mutation_strength=0.1):
    """Apply gaussian mutation to arrays in weights list."""
    new_weights = []
    for w in weights:
        if np.random.rand() < mutation_rate:
            noise = np.random.normal(0, mutation_strength, w.shape)
            new_weights.append((w + noise).astype(np.float32))
        else:
            new_weights.append(w.copy().astype(np.float32))
    return new_weights

# def evolve_model(input_size, output_size, pop_size, generations, best_fit):
#     population = []

#     for i in range(pop_size):
#         population.append(create_model(input_size, output_size))
#         print(population[i].layers[0].get_weights())
    
#     for i in range(generations):
#         print(f"Generation: {i}")
#         new_population = []
#         scores = []
#         for j in range(pop_size):
#             print(f"Pop: {j}")
#             scores.append([j, population[j], play_game(population[j], True)[0]])
#             print(scores[j][2])
        
#         best_scores = sorted(scores, key=lambda x: x[2], reverse=True)[:best_fit]
#         play_game(best_scores[0][1], True)
#         if i == generations - 1:
#             return best_scores[0][1]
        
#         for k in range(pop_size // (best_fit // 2)):
#             random.shuffle(best_scores)
#             for i in range(0, best_fit, 2):
#                 child = create_model(input_size, output_size)
#                 model_a = best_scores[i][1]
#                 model_b = best_scores[i + 1][1]
#                 print(model_b)
#                 print("gay")
#                 # rand_num2 = random.random()
#                 # for j in range(4):
#                 #     rand_num1 = random.random()
                    
#                 #     if rand_num1 < 0.3:
#                 #         temp = model_a.layers[j].get_weights().copy()
#                 #         for z in range(len(model_a.layers[j].get_weights())):
#                 #             temp[z] = (temp[z] + model_b.layers[j].get_weights()[z]) / 2
#                 #         child.add(tf.keras.layers.Dense(model_a.layers[j].units, model_a.layers[j].activation))
#                 #         child.layers[j].set_weights(temp)
#                 #     else:
#                 #         if rand_num2 < 0.5:
#                 #             source = model_a.layers[j]
#                 #         else:
#                 #             source = model_b.layers[j]
#                 #         child.add(tf.keras.layers.Dense(source.units, source.activation))
#                 #         child.layers[j].set_weights(source.get_weights())
#                     # weights = child.layers[j].get_weights()
#                     # std = np.std(np.concatenate([w.flatten() for w in weights]))
#                     # mutate_model(child.layers[j], 0.3, std)
#                 gone_to = [[-1, -1, -1]]
#                 for i in range(input_size):
#                     prev = []
#                     for j in range(4):
#                         start = [-1, -1, -1]
#                         if j == 0:
#                             while start in gone_to:
#                                 start = [j, i, random.randint(1, len(model_a.layers[j].get_weights()[0][0])) - 1]
#                             gone_to.append(start)
                            
#                             weights = model_a.layers[j].get_weights()
#                             b = model_b.layers[j].get_weights()
#                             weights[0][i][start[2]] = (weights[0][i][start[2]] + b[0][i][start[2]]) / 2
#                         else:
#                             while start in gone_to:
#                                 start = [j, prev[1], random.randint(1, len(model_a.layers[j].get_weights()[0][0])) - 1]
#                             gone_to.append(start)

#                             weights = model_a.layers[j].get_weights()
#                             b = model_b.layers[j].get_weights()
#                             # print(prev[2])
#                             # print(start[2])
#                             # print("poes")
#                             # print(len(weights[0]))
#                             # print((weights[0]))
#                             # print(len(weights[0][0]))
#                             # print(weights[0][prev[2]][start[2]])
#                             weights[0][prev[2]][start[2]] = (weights[0][prev[2]][start[2]] + b[0][prev[2]][start[2]]) / 2
#                         model_a.layers[j].set_weights(weights)
#                         child.layers[j].set_weights(weights)
#                         prev = start
#                 for i in range(4):
#                     weights = child.layers[i].get_weights()[0]
#                     std = np.std(np.concatenate([w.flatten() for w in weights]))
#                     mutate_model(child.layers[i], 0.3, std)
                    
#                 new_population.append(child)
#         population = new_population

def evolve_model(input_size, output_size, pop_size, generations, best_fit, eval_runs=1):
    # 1) init population
    population = [create_model(input_size, output_size) for _ in range(pop_size)]

    for gen in range(generations):
        print(f"Generation {gen}")
        # 2) evaluate each model multiple times -> average fitness
        scores = []
        for idx, model in enumerate(population):
            total = 0
            for _ in range(eval_runs):
                total += play_game(model, True)[0]  # render False for speed
            avg = total / eval_runs
            scores.append((avg, idx, model))
            print(f"  pop {idx}: avg score {avg}")

        # 3) sort and keep elites
        scores.sort(reverse=True, key=lambda x: x[0])
        elites = [s[2] for s in scores[:best_fit]]
        print(f"  best score this gen: {scores[0][0]}")

        # optionally save best model periodically
        if gen == generations - 1:
            return elites[0]

        # 4) create new population
        new_pop = []
        # keep elites
        for e in elites:
            # deep copy weights to keep those models unchanged
            m = create_model(input_size, output_size)
            m.set_weights([w.copy() for w in e.get_weights()])
            new_pop.append(m)

        # fill the rest by crossover+mutation
        while len(new_pop) < pop_size:
            a, b = random.sample(elites, 2)
            wa = a.get_weights()
            wb = b.get_weights()
            child_w = crossover_weights(wa, wb)
            # mutation_strength relative to std of all weights
            # compute std to scale mutation_strength
            flat = np.concatenate([w.flatten() for w in child_w if w.size > 0])
            std = np.std(flat) if flat.size > 0 else 0.1
            child_w = mutate_weights(child_w, mutation_rate=0.2, mutation_strength=std*0.1)
            child = create_model(input_size, output_size)
            child.set_weights(child_w)
            new_pop.append(child)

        population = new_pop





if __name__ == "__main__":
    best_model = evolve_model(7, 1, 20, 40, 4)

    best_model.save("best_tetris_model.keras")  

    play_game(best_model, True)

