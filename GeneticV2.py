import tensorflow as tf
import random
import numpy as np
from Tetris import TetrisEnv
import heapq
import multiprocessing as mp
import os
import pygame

def create_model(input_size, output_size, max_layers = 3):
    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.Input(shape=(input_size,)))

    num_layers = max_layers

    neurons = [8, 4, 4]
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
        all_moves = env.get_every_move()
        best_move = []
        best_score = None
        for i in range(len(all_moves)):
            print(i)
            state = env.simulate_conditions(all_moves[i])
            state = np.array(state, dtype=np.float32).reshape(1, -1)
            output = model(state)[0][0]
            output = float(output.numpy())
            if best_score == None:
                best_score = output
                best_move = all_moves[i]
            elif output >= best_score:
                best_score = output
                best_move = all_moves[i]
        if best_move != []:
            for i in range(len(best_move[0])):
                step = env.step(best_move[0][i])
                if render:
                    env.render()
                done = step[0]
                if done:
                    final_points = step[1]
        else:
            done = True

    return final_points, best_score, best_move

def mutate_model(model, mutation_rate = 0.5, mutation_strength = 0.001):
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

def evolve_model(input_size, output_size, pop_size, generations, best_fit):
    population = []

    for i in range(pop_size):
        population.append(create_model(input_size, output_size))
    
    for i in range(generations):
        print(f"Generation: {i}")
        new_population = []
        scores = []
        for j in range(pop_size):
            print(f"Pop: {j}")
            scores.append([j, population[j], play_game(population[j], True)[0]])
            print(scores[j][2])
        
        best_scores = sorted(scores, key=lambda x: x[2], reverse=True)[:best_fit]
        play_game(best_scores[0][1], True)
        if i == generations - 1:
            return best_scores[0][1]
        
        for k in range(pop_size // (best_fit // 2)):
            random.shuffle(best_scores)
            for i in range(0, best_fit, 2):
                child = tf.keras.models.Sequential()
                child.add(tf.keras.layers.Input(input_size))
                model_a = best_scores[i][1]
                model_b = best_scores[i + 1][1]
                print(model_b)
                print("gay")
                for j in range(4):
                    rand_num = random.random()
                    if rand_num > 0.1:
                        source = model_a.layers[j]
                    else:
                        source = model_b.layers[j]

                    child.add(tf.keras.layers.Dense(source.units, source.activation))
                    child.layers[j].set_weights(source.get_weights())
                child.add(tf.keras.layers.Dense(output_size, activation = "linear"))

                mutate_model(child)

                new_population.append(child)
        population = new_population




if __name__ == "__main__":
    best_model = evolve_model(9, 1, 20, 50, 4)

    best_model.save("best_tetris_model.keras")  

    play_game(best_model, True)

