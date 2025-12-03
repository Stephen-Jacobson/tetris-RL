import tensorflow as tf
import random
import numpy as np
from Tetris import TetrisEnv
import heapq
import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import pickle

# Set environment variable to suppress pygame welcome message
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=all, 1=info, 2=warning, 3=error

def create_model(input_size, output_size, max_layers=3):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Input(input_size))
    
    num_layers = max_layers
    neurons = [64, 32, 32]
    activation = ["relu", "sigmoid", "tanh"]
    
    for i in range(num_layers):
        model.add(tf.keras.layers.Dense(neurons[i], activation=random.choice(activation)))
    
    model.add(tf.keras.layers.Dense(output_size, activation="linear"))
    return model

def get_model_weights(model):
    """Extract weights from model for serialization"""
    weights = []
    config = []
    for layer in model.layers:
        if hasattr(layer, 'units'):  # Dense layer
            config.append({
                'units': layer.units,
                'activation': layer.activation.__name__ if hasattr(layer.activation, '__name__') else str(layer.activation)
            })
            weights.append(layer.get_weights())
    return weights, config

def build_model_from_weights(input_size, output_size, weights, config):
    """Rebuild model from weights and config"""
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Input(input_size))
    
    for i, layer_config in enumerate(config[:-1]):  # Exclude output layer
        model.add(tf.keras.layers.Dense(layer_config['units'], activation=layer_config['activation']))
        if i < len(weights):
            model.layers[i].set_weights(weights[i])
    
    # Output layer
    model.add(tf.keras.layers.Dense(output_size, activation="linear"))
    if len(config) <= len(weights):
        model.layers[-1].set_weights(weights[-1])
    
    return model

def play_game_worker(args):
    """Worker function that can be pickled - receives weights instead of model"""
    input_size, output_size, weights, config, render, max_steps = args
    
    # CRITICAL: Force CPU only in worker processes to avoid GPU memory issues
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    
    # Rebuild model in this process
    model = build_model_from_weights(input_size, output_size, weights, config)
    
    env = TetrisEnv()
    clears = 0
    total_reward = 0
    not_done = True
    steps = 0
    
    try:
        while not_done and steps < max_steps:
            state = []
            
            # Next pieces
            for i in range(len(env.next_pieces)):
                next_pieces_norm = ((env.next_pieces[i] / 7) * 2) - 1
                state.append(next_pieces_norm)
            
            # Held piece
            if env.held_piece == None:
                held_piece = -1
            else:
                held_piece = ((env.held_piece.type / 7) * 2) - 1
            
            # Current piece
            cur_piece = ((env.cur_piece.type / 7) * 2) - 1
            
            # Board features
            bumpiness = ((env.get_bumpiness() / 216) * 2) - 1
            holiness = ((env.get_holes() / 230) * 2) - 1
            agg_height = ((env.get_aggregate_height() / 240) * 2) - 1
            
            # T-spin detection
            t_spin = 1 if env.tslot_exists() else -1
            
            # Clears normalized
            clears_norm = ((clears / 4) * 2) - 1
            
            state.extend([held_piece, cur_piece, bumpiness, holiness, agg_height, t_spin, clears_norm])
            
            # Get action from model
            output = model(tf.convert_to_tensor([state], dtype=tf.float32), training=False)
            action = tf.argmax(output[0]).numpy()
            
            # Take step
            step_result = env.step(action)
            steps += 1
            
            if render:
                env.render()
            
            reward = step_result[1]
            reward = reward - 0.2*bumpiness - 0.3*holiness - 0.2*agg_height
            total_reward += reward
            
            clears = env.clears
            not_done = not step_result[2]
        
        return total_reward
    
    except Exception as e:
        print(f"Error in worker: {e}")
        return -1000

def mutate_model(model, mutation_rate=0.1, mutation_strength=0.2):
    """Mutate model weights in place"""
    for layer in model.layers:
        weights = layer.get_weights()
        if len(weights) > 0:
            new_weights = []
            for weight_matrix in weights:
                mutation_mask = np.random.random(weight_matrix.shape) < mutation_rate
                mutations = np.random.normal(0, mutation_strength, weight_matrix.shape)
                mutated = weight_matrix + (mutations * mutation_mask)
                new_weights.append(mutated)
            layer.set_weights(new_weights)
    return model

def crossover_models(model_a, model_b, input_size, output_size):
    """Create a child model by crossing over two parent models"""
    child = tf.keras.models.Sequential()
    child.add(tf.keras.layers.Input(input_size))
    
    # Crossover hidden layers
    for j in range(3):
        source = model_a.layers[j] if random.randint(0, 1) == 0 else model_b.layers[j]
        child.add(tf.keras.layers.Dense(source.units, activation=source.activation))
        child.layers[j].set_weights(source.get_weights())
    
    # Output layer
    child.add(tf.keras.layers.Dense(output_size, activation="linear"))
    
    return child

def evolve_population(input_size, output_size, pop_size, generations, best_fit, num_workers=None):
    """Evolve population using safe multiprocessing with weight serialization"""
    
    if num_workers is None:
        num_workers = max(1, mp.cpu_count() - 1)  # Leave one core free
    
    print(f"Using {num_workers} worker processes")
    print(f"Initializing population of {pop_size} models...")
    population = [create_model(input_size, output_size) for _ in range(pop_size)]
    
    for generation in range(generations):
        print(f"\n{'='*50}")
        print(f"Generation {generation + 1}/{generations}")
        print(f"{'='*50}")
        
        # Prepare tasks for parallel evaluation
        should_render = ((generation + 1) % 10) == 0 and generation != 0
        tasks = []
        
        for i, model in enumerate(population):
            render_this = should_render and i == 0
            weights, config = get_model_weights(model)
            tasks.append((input_size, output_size, weights, config, render_this, 2000))
        
        # Evaluate in parallel
        fitness_scores = [0] * pop_size
        
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Submit all tasks
            future_to_idx = {executor.submit(play_game_worker, task): i 
                           for i, task in enumerate(tasks)}
            
            # Collect results as they complete
            completed = 0
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    fitness = future.result(timeout=60)  # 60 second timeout per game
                    fitness_scores[idx] = fitness
                except Exception as e:
                    print(f"  Error evaluating model {idx}: {e}")
                    fitness_scores[idx] = -1000
                
                completed += 1
                if completed % 10 == 0:
                    print(f"  Evaluated {completed}/{pop_size} models...")
        
        # Statistics
        max_fitness = max(fitness_scores)
        avg_fitness = sum(fitness_scores) / len(fitness_scores)
        min_fitness = min(fitness_scores)
        
        print(f"\nGeneration {generation + 1} Results:")
        print(f"  Max fitness: {max_fitness:.2f}")
        print(f"  Avg fitness: {avg_fitness:.2f}")
        print(f"  Min fitness: {min_fitness:.2f}")
        
        # Select best models
        best_indices = heapq.nlargest(best_fit, range(len(fitness_scores)), 
                                      key=lambda x: fitness_scores[x])
        best_models = [population[i] for i in best_indices]
        
        # Create new population through crossover and mutation
        new_population = []
        
        num_children_per_iteration = best_fit // 2
        iterations_needed = pop_size // num_children_per_iteration
        
        for k in range(iterations_needed):
            random.shuffle(best_models)
            for i in range(0, best_fit - 1, 2):
                child = crossover_models(best_models[i], best_models[i + 1], 
                                        input_size, output_size)
                mutate_model(child)
                new_population.append(child)
                
                if len(new_population) >= pop_size:
                    break
            if len(new_population) >= pop_size:
                break
        
        population = new_population[:pop_size]
        
        # Save best model periodically
        if (generation + 1) % 10 == 0:
            best_idx = best_indices[0]
            population[best_idx].save(f'checkpoint_gen_{generation + 1}.keras')
            print(f"  Checkpoint saved: checkpoint_gen_{generation + 1}.keras")
    
    # Final evaluation
    print("\nFinal evaluation...")
    final_fitness = []
    
    tasks = []
    for model in population:
        weights, config = get_model_weights(model)
        tasks.append((input_size, output_size, weights, config, False, 2000))
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(play_game_worker, task) for task in tasks]
        for future in as_completed(futures):
            try:
                fitness = future.result(timeout=60)
                final_fitness.append(fitness)
            except Exception as e:
                print(f"Error in final evaluation: {e}")
                final_fitness.append(-1000)
    
    best_idx = final_fitness.index(max(final_fitness))
    print(f"\nBest model fitness: {final_fitness[best_idx]:.2f}")
    
    return population[best_idx]

if __name__ == "__main__":
    # CRITICAL: Set start method for Windows
    mp.set_start_method('spawn', force=True)
    
    print("Starting genetic algorithm training with multiprocessing...")
    
    # Determine number of workers
    num_cores = mp.cpu_count()
    # Use fewer workers to be safe - each process needs memory
    num_workers = max(8, max(1, num_cores // 2))  # Cap at 8 workers, use half cores
    print(f"Detected {num_cores} CPU cores, using {num_workers} workers")
    print("Workers will use CPU only (GPU disabled in worker processes)")
    
    best_model = evolve_population(
        input_size=10,
        output_size=8,
        pop_size=30,       # Reduced population for stability
        generations=100,
        best_fit=10,
        num_workers=num_workers
    )
    
    # Save the best model
    best_model.save('best_tetris_model.keras')
    print("\nTraining complete! Best model saved to 'best_tetris_model.keras'")