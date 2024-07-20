import pygame
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import sys
import os
import time
#add ending screen
#add bg music
#add score board

# Initialize Pygame
pygame.init()

# Screen dimensions
WIDTH, HEIGHT = 400, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('Flappy Bird')

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# Clock for controlling the frame rate
clock = pygame.time.Clock()

# Bird settings
#pygame.Rect(left, top, width, height)
bird = pygame.Rect(50, 300, 30, 30)
bird_velocity = 0
gravity = 0.5
import tensorflow as tf
print(tf.__path__)


# Pipe settings
pipe_width = 50
pipe_gap = 150
pipe_velocity = -3

# Initialize Pygame
pygame.init()

# Define the path to the assets directory
assets_dir = os.path.join(os.path.dirname(__file__), 'assets')
model_dir = os.path.join(os.path.dirname(__file__), 'model')

# Define the path to the assets directory
assets_dir = os.path.join(os.path.dirname(__file__), 'assets')
model_dir = os.path.join(os.path.dirname(__file__), 'model')


# Load assets using relative paths
bird_image = pygame.image.load(os.path.join(assets_dir, 'bird.png'))
bird_image = pygame.transform.scale(bird_image, (30, 30))  # Resize bird to 30x30 pixels
pipe_image = pygame.image.load(os.path.join(assets_dir, 'pipe.png')).convert_alpha()
t1op_pipe_image = pygame.image.load(os.path.join(assets_dir, 'toppipe.png')).convert_alpha() 
jump_sound = pygame.mixer.Sound(os.path.join(assets_dir, 'jump.wav'))
button_sound = pygame.mixer.Sound(os.path.join(assets_dir, 'but_mv.mp3'))
collision_sound = pygame.mixer.Sound(os.path.join(assets_dir, 'collision.wav'))
train_button=pygame.image.load(os.path.join(assets_dir, 'b_3.png'))
font_path=os.path.join(assets_dir, 'Evil Empire.otf')
bg_img=pygame.image.load(os.path.join(assets_dir,'bg.png'))
# Your game loop and other code here
from tensorflow.keras.layers import Dense, Dropout
import pickle

import pickle

# def save_training_data(training_data, file_path):
#     with open(file_path, 'wb') as f:
#         pickle.dump(training_data, f)
def save_training_data(training_data, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(training_data, f)

# def load_training_data(file_path):
#     with open(file_path, 'rb') as f:
#         return pickle.load(f)
# training_data_file = os.path.join(model_dir, 'training_data.pkl')
# from tensorflow.keras.optimizers import Adam

def load_training_data(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)
training_data_file = os.path.join(model_dir, 'training_data.pkl')
from tensorflow.keras.optimizers import Adam



if(os.path.exists(os.path.join(model_dir,'model.keras'))):
    model=tf.keras.models.load_model(os.path.join(model_dir,'model.keras'))
    print("MODEL LOADED")
else:
    print("Creating New Model")
    model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(5,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(2, activation='softmax')  # Use softmax for classification
])
    
    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy')
model.summary()
if(os.path.exists(os.path.join(model_dir,'model.keras'))):
    model=tf.keras.models.load_model(os.path.join(model_dir,'model.keras'))
    print("MODEL LOADED")
else:
    print("Creating New Model")
    model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(6,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')  # Use softmax for classification
])
    
    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
model.summary()

#bird_y: Vertical position of the bird (normalized).
#bird_velocity: Vertical velocity of the bird (normalized).
#pipe_x: Horizontal position of the nearest pipe (normalized).
#pipe_gap_y: Vertical position of the gap between the nearest pipes (normalized).

COLLISION_PENALTY = -1
STAY_ALIVE_REWARD = 0.01
SAFE_REWARD = 0.5
SUCCESSFUL_PASS_REWARD = 1

def get_state(bird, pipes, bird_velocity):
    # Normalize the bird's y position and velocity
    bird_y = bird.y / HEIGHT  # Normalize bird's y position
    bird_velocity = bird_velocity / 10  # Normalize bird's velocity
    
    # Normalize the position and size of the pipes
    pipe = pipes[0] if pipes else (pygame.Rect(WIDTH, 0, pipe_width, HEIGHT), pygame.Rect(width, HEIGHT, pipe_width, HEIGHT))
    pipe_x = pipe[0].x / WIDTH  # Normalize pipe's x position
    top_pipe_bottom = pipe[0].bottom
    bottom_pipe_top = pipe[1].top
    pipe_gap = (top_pipe_bottom - bottom_pipe_top) / HEIGHT  # Normalize pipe gap size
    
    # Calculate the distance between the bird and the closest pipe
    distance_to_pipe = (pipe[0].x + pipe_width - bird.x) / WIDTH  # Normalize distance to the pipe
    
    # Determine if the bird is in a safe region
    bird_bottom = bird.y + bird.height
    in_safe_region = int(bird_bottom < top_pipe_bottom and bird.y > bottom_pipe_top)
    
    # Normalize the safe region indicator (0 or 1)
    safe_region_indicator = in_safe_region

    return np.array([bird_y, bird_velocity, pipe_x, pipe_gap, distance_to_pipe, safe_region_indicator])



def get_reward(bird, pipes):
    # Define reward and penalty values
    COLLISION_PENALTY = -10
    SUCCESSFUL_PASS_REWARD = 10

    # Extract pipe information
    top_pipe = pipes[0][0]
    bottom_pipe = pipes[0][1]

    # Check for collision with any part of the pipes or out of bounds
    if bird.colliderect(top_pipe) or bird.colliderect(bottom_pipe) or bird.y > HEIGHT or bird.y < 0:
        return COLLISION_PENALTY  # Collision penalty

    # Check if the bird has passed through the pipes
    pipe_x = top_pipe.x
    if bird.x > pipe_x + pipe_width:
        return SUCCESSFUL_PASS_REWARD  # Reward for passing through a pipe

    # Check if the bird is within the safe vertical distance from the pipe gap
    if bird.bottom < top_pipe.bottom - 10 or bird.top > bottom_pipe.top + 10:
        return -1  # Penalty for staying outside the safe vertical distance
    else:
        return 0  # Neutral reward for staying within the safe region

# Ensure to replace `height`, `pipe_width`, `COLLISION_PENALTY`, and `SUCCESSFUL_PASS_REWARD` with actual integer values as needed.

def create_pipe():
    height = random.randint(100, 400)
    top_pipe = pygame.Rect(WIDTH, 0, pipe_width, height)
    bottom_pipe = pygame.Rect(WIDTH, height + pipe_gap, pipe_width, HEIGHT - height - pipe_gap)
    top_pipe_bottom = top_pipe.bottom
    bottom_pipe_top = bottom_pipe.top
    return top_pipe, bottom_pipe, top_pipe_bottom, bottom_pipe_top

def draw_bird_and_pipes(bird, pipes):
    global background_x
    background_x -= 1  # Adjust speed as needed
    if background_x <= -background_width:
        background_x = 0

    # Clear the screen
    screen.fill((0, 0, 0))

    # Draw the background
    screen.blit(background_image, (background_x, 0))
    screen.blit(background_image, (background_x + background_width, 0))
    screen.blit(bird_image, (bird.x, bird.y))
    for top_pipe, bottom_pipe, top_pipe_bottom, bottom_pipe_top in pipes:
        top_pipe_image = pygame.transform.scale(t1op_pipe_image, (pipe_width, top_pipe.height))
        bottom_pipe_image = pygame.transform.scale(pipe_image, (pipe_width, HEIGHT - top_pipe.height - pipe_gap))
        screen.blit(top_pipe_image, (top_pipe.x, top_pipe.y))
        screen.blit(bottom_pipe_image, (bottom_pipe.x, bottom_pipe.y))
    pygame.display.flip()

def reset_game():
    global bird, bird_velocity, pipes
    bird = pygame.Rect(50, 300, 30, 30)
    bird_velocity = 0
    pipes = [create_pipe()]

# def collect_training_data():
#     global bird_velocity
#     training_data = []

#     reset_game()
#     running = True
#     while running:
#         for event in pygame.event.get():
#             if event.type == pygame.QUIT:
#                 running = False
#                 pygame.quit()
#                 sys.exit()
#             if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
#                 bird_velocity = -8
#                 if hasattr(pygame, 'mixer'):
#                     jump_sound.play()

#         # Bird movement
#         bird_velocity += gravity
#         bird.y += bird_velocity

#         # Pipe movement
#         for pipe in pipes:
#             pipe[0].x += pipe_velocity
#             pipe[1].x += pipe_velocity

#         # Remove pipes off the screen
#         if pipes[0][0].x < -pipe_width:
#             pipes.pop(0)
#             pipes.append(create_pipe())

#         # Get state, action, and reward
#         state = get_state(bird, pipes, bird_velocity)
#         action = int(bird_velocity < 0)  # Assuming action is based on velocity; modify if necessary
#         reward = get_reward(bird, pipes)
#         next_state = get_state(bird, pipes, bird_velocity)

#         training_data.append((state, action, reward, next_state))

#         if reward == COLLISION_PENALTY:
#             if hasattr(pygame, 'mixer'):
#                 collision_sound.play()
#             running = False

#         # Draw the bird and pipes
#         draw_bird_and_pipes(bird, pipes)
#         clock.tick(30)
#         print(f"{is_in_safe_region(bird,pipes,pipe_gap)}")


#     # Print some training data for debugging

#     return training_data

def collect_training_data():
    global bird_velocity
    training_data = []

    reset_game()
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                bird_velocity = -8
                if hasattr(pygame, 'mixer'):
                    jump_sound.play()

        # Bird movement
        bird_velocity += gravity
        bird.y += bird_velocity

        # Pipe movement
        for pipe in pipes:
            pipe[0].x += pipe_velocity
            pipe[1].x += pipe_velocity

        # Remove pipes off the screen
        if pipes[0][0].x < -pipe_width:
            pipes.pop(0)
            pipes.append(create_pipe())

        # Get state, action, and reward
        state = get_state(bird, pipes, bird_velocity)
        action = int(bird_velocity < 0)  # Assuming action is based on velocity
        reward = get_reward(bird, pipes)
        next_state = get_state(bird, pipes, bird_velocity)

        training_data.append((state, action, reward, next_state))

        if reward == COLLISION_PENALTY:
            if hasattr(pygame, 'mixer'):
                collision_sound.play()
            running = False

        # Draw the bird and pipes
        draw_bird_and_pipes(bird, pipes)
        clock.tick(30)

    return training_data
# def train_model(model, training_data, epochs=10):
    # # Unpack training data
    # states, actions, rewards, next_states = zip(*training_data)
    # states = np.array(states)
    # actions = np.array(actions)
    # rewards = np.array(rewards)
    # next_states = np.array(next_states)
    
    # # Prepare target values (Q-values)
    # next_q_values = model.predict(next_states)
    # targets = rewards + 0.99 * np.max(next_q_values, axis=1)
    
    # # Convert actions to one-hot encoding if necessary
    # actions_one_hot = tf.keras.utils.to_categorical(actions, num_classes=model.output_shape[1])
    
    # # Train the model
    # model.fit(states, targets, epochs=epochs, verbose=1)
    
    # # Save the model
    # model_path = os.path.join(model_dir, 'model.keras')
    # model.save(model_path)


# def train_model(model, training_data, epochs=10):
#     states, rewards = zip(*training_data)
#     states = np.array(states)
#     rewards = np.array(rewards)
#     model.fit(states, rewards, epochs=epochs)
    
#     model_path = os.path.join(model_dir, 'model.keras')
#     model.save(model_path)

def train_model(model, training_data, epochs=10):
    # Extract states, actions, rewards, and next_states from training data
    states, actions, rewards, next_states = zip(*training_data)

    # Debugging: Print shapes to check consistency
    print(f"Number of samples: {len(states)}")
    print(f"State example shape: {np.shape(states[0])}")
    print(f"Next state example shape: {np.shape(next_states[0])}")
    
    # Convert to numpy arrays, ensuring all states are of the same shape
    states = np.array(states)
    actions = np.array(actions)
    rewards = np.array(rewards)
    next_states = np.array(next_states)
    
    # Check shapes after conversion
    print(f"States shape: {states.shape}")
    print(f"Actions shape: {actions.shape}")
    print(f"Rewards shape: {rewards.shape}")
    print(f"Next states shape: {next_states.shape}")

    # Assuming a Q-learning setup where you need to train on Q-values
    # Calculate target Q-values (if applicable)
    targets = rewards  # Simplified for illustration; adjust as needed

    # Compile the model
    model.compile(optimizer=Adam(), loss='mean_squared_error')  # Adjust loss function as needed

    # Train the model
    model.fit(states, targets, epochs=epochs, verbose=1)

    # Save the model
    model_path = os.path.join(model_dir, 'model.keras')
    model.save(model_path)
def is_in_safe_region(bird, pipes, pipe_gap):
    """
    Checks if the bird is in a safe region, where a safe region is defined as 
    being vertically away from the pipes by a safe margin.
    
    Parameters:
    - bird (pygame.Rect): The bird's current position and dimensions.
    - pipes (list): A list containing the pipe pair, where each pipe is a tuple of pygame.Rect.
    - pipe_gap (float): The vertical gap between the top and bottom pipes.

    Returns:
    - str: "safe" if the bird is in a safe region, otherwise "not safe".
    """
    top_pipe, bottom_pipe, top_pipe_bottom, bottom_pipe_top = pipes[0]

    # Calculate the safe vertical margin
    safe_margin = 10  # Define a safe margin as needed

    # Check if the bird is within the safe margin from the top pipe's bottom and bottom pipe's top
    if (bird.bottom < top_pipe.bottom - safe_margin or
        bird.top > bottom_pipe.top + safe_margin):
        return "!safe"
    else:
        return "safe"

def automatic_play():
    print("Starting automatic play")
    global bird_velocity
    reset_game()
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                pygame.quit()
                sys.exit()

        # Get state and predict action
        state = get_state(bird, pipes, bird_velocity)
        state = np.expand_dims(state, axis=0)  # Add batch dimension
        action_probs = model.predict(state)[0]
        action = np.argmax(action_probs)   # Get the probabilities for the actions
        print(is_in_safe_region(bird,pipes,pipe_gap))
        print(f"State: {state}, Action probabilities: {action_probs}")

        # Interpret action probabilities
        action = 1 if action_probs[1] < -0.5 else 0  # Adjust threshold as needed
        print(f"Selected action: {action}")
        action_probs = model.predict(state)[0]
        print(f"Action probabilities: {action_probs}")
        print(f"Raw model output: {model.predict(state)}")

        # Bird movement based on action
        if action == 1:  # Assuming action 1 corresponds to "jump"
            bird_velocity = -8
            if hasattr(pygame, 'mixer'):
                jump_sound.play()

        # Update bird's position
        bird_velocity += gravity
        bird.y += bird_velocity

        # Pipe movement
        for pipe in pipes:
            pipe[0].x += pipe_velocity
            pipe[1].x += pipe_velocity

        # Remove pipes off the screen
        if pipes[0][0].x < -pipe_width:
            pipes.pop(0)
            pipes.append(create_pipe())

        # Collision detection
        if (bird.colliderect(pipes[0][0]) or bird.colliderect(pipes[0][1])
                or bird.y > HEIGHT or bird.y < 0):
            if hasattr(pygame, 'mixer'):
                collision_sound.play()
            running = False

        # Draw the bird and pipes
        draw_bird_and_pipes(bird, pipes)

if __name__ == "__main__":
    game_font=pygame.font.Font(font_path,32)
    game_font_s=pygame.font.Font(font_path,16)
    Train_text=game_font.render('Train',True,(255,255,255))
    Test_text=game_font.render('Test',True,(255,255,255))
    Train_text_s=game_font.render('Train',True,(255,255,105))
    Test_text_s=game_font.render('Test',True,(255,255,105))
    Quit_text_s=game_font_s.render('Esc to quit',True,((255, 255, 0)))
    Arrow_text=game_font_s.render('Use Arrow Keys',True,((255, 255, 0)))
    background_x=0
    background_image_i = bg_img.convert()
    background_image=pygame.transform.scale(background_image_i,(background_image_i.get_width()*3,background_image_i.get_height()*3))
    background_width, background_height = background_image.get_size()
    play_select=True
    reset_game()
    menu_running = True
    while menu_running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN and (event.key == pygame.K_DOWN or event.key == pygame.K_UP):
                play_select=not play_select
                bird_velocity = -8
                if hasattr(pygame, 'mixer'):
                    button_sound.play()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN:
                if(play_select):
                    main()
                else:
                    automatic_play()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                pygame.quit()
        # if(play_select):
        #     #show button train higlighted
        # else:
        #     #show model run highlighted
        screen.fill(BLACK)
        screen.blit(pygame.transform.smoothscale(train_button,(train_button.get_width()/4,train_button.get_height()/4)), (100, 200))
        screen.blit(Quit_text_s,(10,10))
        screen.blit(Arrow_text,(screen.get_width()-120,10))
        if(play_select):
            screen.blit(Train_text,(165,220))
        else:
            screen.blit(Train_text_s,(165,220))
        screen.blit(pygame.transform.smoothscale(train_button,(train_button.get_width()/4,train_button.get_height()/4)), (100, 300))
        if(not play_select):
            screen.blit(Test_text,(172,320))
        else:
            screen.blit(Test_text_s,(172,320))
        pygame.display.flip()
        clock.tick(30)    


    





#Pygame Coordinate System
#Origin (0, 0): The top-left corner of the screen.
#X-axis: Increases to the right.
#Y-axis: Increases downward.
#Specifics
#Top-left corner: (0, 0)
#Bottom-left corner: (0, HEIGHT)
#Top-right corner: (WIDTH, 0)
#Bottom-right corner: (WIDTH, HEIGHT)
#Bird's Movement
#Vertical Position (y-coordinate):

#Moving Up: Decrease in the y-coordinate.
#Moving Down: Increase in the y-coordinate.
#Horizontal Position (x-coordinate):

#Typically constant for the bird in Flappy Bird. The bird's horizontal position is usually fixed, and the pipes move horizontally.


# =====================================