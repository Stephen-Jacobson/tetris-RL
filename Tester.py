import tensorflow as tf
import random
import numpy as np
from Tetris import TetrisEnv
import heapq
import multiprocessing as mp
import os

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def _worker_init():
    import os, tensorflow as tf
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    try:
        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.config.threading.set_inter_op_parallelism_threads(1)
    except Exception:
        pass

def create_model(input_size, output_size, max_layers=3):
    """FIXED: Use consistent activations instead of random"""
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Input(input_size))
    
    neurons = [64, 32, 32]
    for i in range(max_layers):
        model.add(tf.keras.layers.Dense(neurons[i], activation="relu"))
    
    model.add(tf.keras.layers.Dense(output_size, activation="linear"))
    return model

def play_game(model, render=False, max_pieces=100):
    """
    COMPLETELY REWRITTEN - Tracks pieces placed, not steps
    Rewards only given when pieces are placed - encourages efficiency!
    """
    env = TetrisEnv()
    env.reset()
    
    total_reward = 0
    pieces_placed = 0
    actions_taken = 0
    
    # Normalization constants
    MAX_BUMP = 216.0
    MAX_HOLES = 230.0
    MAX_AGG = 240.0
    
    if render:
        env.render()
    
    # Run until max pieces placed (not max actions!)
    while pieces_placed < max_pieces:
        # Build state vector (10 features)
        state = []
        
        # Next 3 pieces (normalized -1 to 1)
        for i in range(len(env.next_pieces)):
            state.append(((env.next_pieces[i] / 7.0) * 2.0) - 1.0)
        
        # Held piece (0 if None, else normalized)
        if env.held_piece is None:
            state.append(0.0)
        else:
            state.append(((env.held_piece.type / 7.0) * 2.0) - 1.0)
        
        # Current piece type
        if env.cur_piece is not None:
            state.append(((env.cur_piece.type / 7.0) * 2.0) - 1.0)
        else:
            state.append(0.0)
        
        # Board metrics (normalized -1 to 1)
        bumpiness = (env.get_bumpiness() / MAX_BUMP) * 2.0 - 1.0
        holes = (env.get_holes() / MAX_HOLES) * 2.0 - 1.0
        agg_height = (env.get_aggregate_height() / MAX_AGG) * 2.0 - 1.0
        
        state.append(bumpiness)
        state.append(holes)
        state.append(agg_height)
        
        # T-slot exists (binary: -1 or 1)
        state.append(1.0 if env.tslot_exists() else -1.0)
        
        # Lines cleared (normalized)
        state.append(((env.clears / 10.0) * 2.0) - 1.0)
        
        # Ensure exactly 10 features
        state = state[:10]
        
        # Get action from model
        logits = model(tf.convert_to_tensor([state], dtype=tf.float32))
        probs = tf.nn.softmax(logits[0]).numpy()
        # numerical safety: if tiny negative numeric fixes, clip
        probs = probs.clip(0, 1)
        probs = probs / probs.sum()
        action = int(np.random.choice(len(probs), p=probs))

        # Step environment
        reward, done, placed = env.step(action)
        
        actions_taken += 1
        
        if placed:
            pieces_placed += 1
            if render:
                print(f"Piece #{pieces_placed} | Actions: {env.cur_piece.steps:2d} | Reward: {reward:7.1f} | Total: {total_reward:7.1f} | Lines: {env.clears}")
        
        total_reward += reward
        
        if render:
            env.render()
        
        if done:
            break
        
        # Safety: prevent infinite loops (max 50 actions per piece average)
        if actions_taken > max_pieces * 50:
            break
    
    # Cleanup
    if env.screen is not None:
        import pygame
        if render:
            pygame.time.delay(500)
        pygame.quit()
    
    # Fitness based on reward and pieces placed
    # More pieces = better survival
    fitness = total_reward + (pieces_placed * 5.0)
    
    if render:
        print(f"\n{'='*60}")
        print(f"GAME SUMMARY")
        print(f"{'='*60}")
        print(f"Total Reward:      {total_reward:8.1f}")
        print(f"Pieces Placed:     {pieces_placed:8d}")
        print(f"Lines Cleared:     {env.clears:8d}")
        print(f"Actions Taken:     {actions_taken:8d}")
        print(f"Avg Actions/Piece: {actions_taken/max(1,pieces_placed):8.1f}")
        print(f"Final Fitness:     {fitness:8.1f}")
        print(f"{'='*60}\n")
    
    return fitness

def crossover(parent_a, parent_b, input_size, output_size):
    """Per-weight crossover between two parents"""
    child = tf.keras.models.clone_model(parent_a)
    child.build((None, input_size))
    
    wa = parent_a.get_weights()
    wb = parent_b.get_weights()
    new_weights = []
    
    for a, b in zip(wa, wb):
        mask = np.random.rand(*a.shape) < 0.5
        new_weights.append(np.where(mask, a, b))
    
    child.set_weights(new_weights)
    return child

def mutate_model(model, mutation_rate=0.15, mutation_strength=0.3):
    """IMPROVED: More conservative mutation"""
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

def evaluate_model_wrapper(args):
    """Wrapper for multiprocessing - NEVER render in workers"""
    idx, weights_list = args
    
    # Recreate model from weights
    model = create_model(10, 8)
    for i, layer_weights in enumerate(weights_list):
        if len(layer_weights) > 0:
            model.layers[i].set_weights(layer_weights)
    
    # Always render=False in workers
    # max_pieces=100 means game ends after 100 pieces placed
    fitness = play_game(model, render=False, max_pieces=100)
    return idx, fitness

def evaluate_population_parallel(population, num_processes=None):
    """Evaluate population in parallel (no rendering)"""
    if num_processes is None:
        num_processes = min(mp.cpu_count(), 8)
    
    # Prepare arguments
    eval_args = []
    for idx, model in enumerate(population):
        weights_list = [layer.get_weights() for layer in model.layers]
        eval_args.append((idx, weights_list))
    
    # Parallel evaluation
    ctx = mp.get_context('spawn')
    with ctx.Pool(processes=num_processes, initializer=_worker_init) as pool:
        results = pool.map(evaluate_model_wrapper, eval_args)
    
    results.sort(key=lambda x: x[0])
    fitness_scores = [fitness for _, fitness in results]
    
    return fitness_scores

def render_best_model(model, max_pieces=200):
    """Render best model in MAIN PROCESS ONLY"""
    print("\n" + "="*50)
    print("RENDERING BEST MODEL")
    print("="*50)
    fitness = play_game(model, render=True, max_pieces=max_pieces)
    print(f"Final Fitness: {fitness:.2f}")
    print("="*50 + "\n")
    return fitness

def evolve_population(input_size, output_size, pop_size, generations, best_fit, num_envs=None):
    """
    IMPROVED: Better selection, elitism, and rendering
    """
    population = []
    
    if num_envs is None:
        num_envs = min(mp.cpu_count(), 8)
    
    # Initialize population
    print("Initializing population...")
    for i in range(pop_size):
        population.append(create_model(input_size, output_size))
    
    best_fitness_ever = float('-inf')
    best_model_ever = None
    
    for gen in range(generations):
        print(f"\n{'='*60}")
        print(f"GENERATION {gen + 1}/{generations}")
        print(f"{'='*60}")
        
        # Evaluate population in parallel
        fitness_scores = evaluate_population_parallel(population, num_envs)
        
        # Stats
        max_fitness = max(fitness_scores)
        avg_fitness = sum(fitness_scores) / len(fitness_scores)
        min_fitness = min(fitness_scores)
        
        print(f"Max Fitness:  {max_fitness:8.2f}")
        print(f"Avg Fitness:  {avg_fitness:8.2f}")
        print(f"Min Fitness:  {min_fitness:8.2f}")
        
        # Track best ever
        if max_fitness > best_fitness_ever:
            best_fitness_ever = max_fitness
            best_idx = fitness_scores.index(max_fitness)
            best_model_ever = tf.keras.models.clone_model(population[best_idx])
            best_model_ever.set_weights(population[best_idx].get_weights())
            print(f"*** NEW BEST FITNESS: {best_fitness_ever:.2f} ***")
        
        # Render every 5 generations (in main process)
        if (gen + 1) % 5 == 0:
            best_idx = fitness_scores.index(max_fitness)
            render_best_model(population[best_idx], max_pieces=200)
        
        # Selection - keep top performers
        best_indices = heapq.nlargest(best_fit, enumerate(fitness_scores), key=lambda x: x[1])
        best_models = [population[idx] for idx, _ in best_indices]
        
        # Create new population with elitism
        elite_count = max(2, best_fit // 4)  # Keep top 25% of best
        new_population = []
        
        # Add elites unchanged
        for i in range(elite_count):
            elite = tf.keras.models.clone_model(best_models[i])
            elite.set_weights(best_models[i].get_weights())
            new_population.append(elite)
        
        # Fill rest with crossover + mutation
        while len(new_population) < pop_size:
            parent_a, parent_b = random.sample(best_models, 2)
            child = crossover(parent_a, parent_b, input_size, output_size)
            
            # Adaptive mutation - stronger early, weaker later
            mutation_strength = 0.4 * (1.0 - gen / generations) + 0.1
            mutate_model(child, mutation_rate=0.15, mutation_strength=mutation_strength)
            
            new_population.append(child)
        
        population = new_population[:pop_size]
    
    # Final evaluation
    print("\n" + "="*60)
    print("FINAL EVALUATION")
    print("="*60)
    final_fitness = evaluate_population_parallel(population, num_envs)
    best_idx = final_fitness.index(max(final_fitness))
    
    # Compare with best ever
    if max(final_fitness) > best_fitness_ever:
        best_model_final = population[best_idx]
    else:
        best_model_final = best_model_ever
        print(f"Using best model from generation (fitness: {best_fitness_ever:.2f})")
    
    # Render final best
    print("\nRendering final best model...")
    render_best_model(best_model_final, max_pieces=300)
    
    return best_model_final

if __name__ == "__main__":
    print("="*60)
    print("TETRIS GENETIC ALGORITHM TRAINING")
    print("="*60)
    print(f"Population Size: 100")
    print(f"Generations: 100")
    print(f"Best Fit: 20")
    print(f"Workers: 16")
    print("="*60 + "\n")
    
    best_model = evolve_population(
        input_size=10,
        output_size=8,
        pop_size=100,
        generations=100,
        best_fit=20,
        num_envs=16
    )
    
    # Save the best model
    best_model.save('best_tetris_model.keras')
    print("\n✓ Training complete! Best model saved to 'best_tetris_model.keras'")