import tensorflow as tf
import random
import numpy as np
from Tetris import TetrisEnv
import heapq
import multiprocessing as mp
import os

# Set environment variable to suppress pygame welcome message
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
# Optional: quiet TF logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def _worker_init():
    # configure tensorflow threading & force CPU-only inside worker processes
    import os, tensorflow as tf
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'   # force CPU-only in workers
    # limit TF threading per worker
    try:
        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.config.threading.set_inter_op_parallelism_threads(1)
    except Exception:
        # older TF versions may not have these APIs available at that point
        pass

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
    cur_piece = -1
    next_pieces = [-1, -1, -1]
    held_piece = -1
    bumpiness = -1
    holiness = -1
    agg_height = -1
    t_spin = False
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
        
        if t_spin:
            t_spin = 1
        else: 
            t_spin = 0
        t_spin = ((t_spin / 1) * 2) - 1

        clears = env.clears - clears
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

        reward = step[0]
        if step[2]:
            cur_piece = ((env.cur_piece.type / 7) * 2) - 1
            bumpiness = ((env.get_bumpiness() / 216) * 2) - 1
            holiness = ((env.get_holes() / 230) * 2) - 1
            agg_height = ((env.get_aggregate_height() / 240) * 2) - 1
            t_spin = env.tslot_exists()
            
            reward = reward - 0.2*bumpiness - 0.3*holiness - 0.2*agg_height

        total_reward += reward

        
        not_done = not(step[1])
        if steps >= 50:
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

def evaluate_model_wrapper(args):
    """Wrapper to unpack arguments for multiprocessing"""
    idx, weights_list, render = args
    
    # Recreate model from weights
    model = create_model(10, 8)
    for i, layer_weights in enumerate(weights_list):
        if len(layer_weights) > 0:
            model.layers[i].set_weights(layer_weights)
    
    fitness = play_game(model, render)
    return idx, fitness

def evaluate_population_parallel(population, num_processes=None, should_render=False):
    # don't render inside child processes (render only in main process)
    if num_processes is None:
        num_processes = min(mp.cpu_count(), 4)  # conservative default

    # prepare args: pass weights (numpy arrays) not models
    eval_args = []
    for idx, model in enumerate(population):
        weights_list = [layer.get_weights() for layer in model.layers]
        # always False for render in workers
        eval_args.append((idx, weights_list, False))

    # Use spawn context and initializer to configure workers once
    ctx = mp.get_context('spawn')
    with ctx.Pool(processes=num_processes, initializer=_worker_init) as pool:
        results = pool.map(evaluate_model_wrapper, eval_args)

    # sort results by index to keep order
    results.sort(key=lambda x: x[0])
    fitness_scores = [fitness for _, fitness in results]

    # if user requested rendering, do it in main process for the best model
    if should_render:
        best_idx = int(np.argmax(fitness_scores))
        print("Rendering best model in main process...")
        play_game(population[best_idx], render=True)

    return fitness_scores

def evolve_population(input_size, output_size, pop_size, generations, best_fit, num_envs=None):
    population = []

    if num_envs is None:
        num_envs = 20

    for i in range(pop_size):
        population.append(create_model(input_size, output_size))

    for t in range(generations):
        print(f"\nGeneration: {t}")
        
        # Evaluate population in parallel
        fitness_scores = evaluate_population_parallel(population, num_envs)
        
        print(f"  Max fitness: {max(fitness_scores):.2f}")
        print(f"  Avg fitness: {sum(fitness_scores)/len(fitness_scores):.2f}")
        print(f"  Min fitness: {min(fitness_scores):.2f}")
        
        # Render best model every 10 generations
        should_render = ((t + 1) % 10) == 0 and t != 0
        if should_render:
            best_idx = fitness_scores.index(max(fitness_scores))
            print(f"  Rendering best model...")
            play_game(population[best_idx], render=True)
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
                    child.layers[j].set_weights(source.get_weights())
                child.add(tf.keras.layers.Dense(output_size, activation = "linear"))

                mutate_model(child)

                new_population.append(child)
        population = new_population
    final_fitness = evaluate_population_parallel(population, num_envs)
    best_idx = final_fitness.index(max(final_fitness))
    return population[best_idx]
    
if __name__ == "__main__":
    # Use all available CPUs
    best_model = evolve_population(10, 8, 100, 100, 10, 20)
    
    # Save the best model
    best_model.save('best_tetris_model.keras')
    print("Training complete! Best model saved.")






    





