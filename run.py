import tensorflow as tf
from GeneticV2 import play_game

# Load the trained model
model = tf.keras.models.load_model("best_tetris_model.keras")

# Run and render the game using the loaded model
final_points, best_score, best_move = play_game(model, render=True)

print("Final points:", final_points)
print("Best score:", best_score)
print("Best move:", best_move)
