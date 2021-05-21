from Snake_Game import *
from Feed_Forward_Neural_Network import *

def run_game_with_ML(display, clock, weights):
    max_score = 0
    avg_score = 0
    test_games = 1
    score1 = 0
    steps_per_game = 2500
    score2 = 0

    for _ in range(test_games):
        # Initialising the game by setting snake position,apple position etc.
        snake_start, snake_position, apple_position, score = starting_positions()

        count_same_direction = 0
        prev_direction = 0

        for _ in range(steps_per_game): #running game for 2500 steps
            # Get current snake direction and blocked directions for snake.
            current_direction_vector, is_front_blocked, is_left_blocked, is_right_blocked = blocked_directions(snake_position)
            angle, snake_direction_vector, apple_direction_vector_normalized, snake_direction_vector_normalized = angle_with_apple(
                snake_position, apple_position)
            predictions = []
            # Predict direction(Left,right,forward) based on output from neural network.
            predicted_direction = np.argmax(np.array(forward_propagation(np.array(
                [is_left_blocked, is_front_blocked, is_right_blocked, apple_direction_vector_normalized[0],
                 snake_direction_vector_normalized[0], apple_direction_vector_normalized[1],
                 snake_direction_vector_normalized[1]]).reshape(-1, 7), weights))) - 1
            # Increment counter if predicted direction is same as past direction.
            if predicted_direction == prev_direction:
                count_same_direction += 1
            else:
                count_same_direction = 0
                prev_direction = predicted_direction
            # Based on predicted direction, calculate snake direction.
            new_direction = np.array(snake_position[0]) - np.array(snake_position[1])
            if predicted_direction == -1:
                new_direction = np.array([new_direction[1], -new_direction[0]])
            if predicted_direction == 1:
                new_direction = np.array([-new_direction[1], new_direction[0]])

            button_direction = generate_button_direction(new_direction)
            # Evaluate the next step of snake.
            next_step = snake_position[0] + current_direction_vector
            # Check if snake collides with a boundary or with itself.
            if collision_with_boundaries(snake_position[0]) == 1 or collision_with_self(next_step.tolist(),
                                                                                        snake_position) == 1:
                score1 -= 150 # Give a negative score to mention that its a wrong move.
                break

            else:
                score1 += 0
            # Play game with current parameters
            snake_position, apple_position, score = play_game(snake_start, snake_position, apple_position,
                                                              button_direction, score, display, clock)

            if score > max_score:
                max_score = score
            # Checking condition for snake movement in loop.
            if count_same_direction > 8 and predicted_direction != 0:
                score2 -= 1 # Give a negative score to mention that its a wrong move. 
            else:
                score2 += 2 # Else give a positive score

    # High weightage is given for maximum score of snake in its 2500 steps.
    # return fitness value
    return score1 + score2 + max_score * 5000