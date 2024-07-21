import pygame
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import sys
import os
import time
# Initialize Pygame
pygame.init()
# Define the path to the assets directory
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

# Pipe settings
pipe_width = 50
pipe_gap = 150
pipe_velocity = -3

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
bg_music = pygame.mixer.Sound(os.path.join(assets_dir, 'bg_music.wav'))


# Create a neural network model
model = Sequential([
    Dense(24, activation='relu', input_shape=(8,)),
    Dense(24, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='mse')

def get_state(bird, pipes, bird_velocity):
    # Normalize bird's vertical position
    bird_y = bird.y / HEIGHT
    
    # Normalize bird's vertical velocity
    bird_velocity /= 10
    
    # Normalize pipe's horizontal position
    pipe_x = pipes[0][0].x / WIDTH
    
    # Calculate the top and bottom of the gap between pipes
    top_pipe_bottom = pipes[0][1].top
    bottom_pipe_top = pipes[0][0].bottom
    
    # Normalize the top and bottom of the gap
    pipe_gap_top = top_pipe_bottom / HEIGHT
    pipe_gap_bottom = bottom_pipe_top / HEIGHT
    pipe_gap_size = (top_pipe_bottom - bottom_pipe_top) / HEIGHT
    
    # Normalize bird's dimensions
    bird_height = bird.height / HEIGHT
    bird_width = bird.width / WIDTH

    if bird.y > bottom_pipe_top:
        distance_from_gap = bird.y - bottom_pipe_top
    else:
        distance_from_gap = top_pipe_bottom - bird.y

    # Normalize the distance from the gap
    gap_height = top_pipe_bottom - bottom_pipe_top
    normalized_distance = distance_from_gap / gap_height
    
    # Return the state as a NumPy array
    return np.array([
        bird_y,                  # Bird's vertical position
        bird_velocity,           # Bird's vertical velocity
        pipe_x,                  # Distance to the nearest pipe
        pipe_gap_top,            # Top of the pipe gap
        pipe_gap_bottom,         # Bottom of the pipe gap
        pipe_gap_size,           # Size of the pipe gap
        bird_height,             # Height of the bird
        normalized_distance               # Width of the bird
    ])


def get_reward(bird, pipes):
    # Extract pipe information
    top_pipe = pipes[0][0]
    bottom_pipe = pipes[0][1]

    # Check for collision with any part of the pipes or ground
    if bird.colliderect(top_pipe) or bird.colliderect(bottom_pipe) or bird.y > HEIGHT or bird.y < 0:
        return -1  # Collision penalty

    # Calculate the gap between the top and bottom pipes
    gap_top = top_pipe.bottom
    gap_bottom = bottom_pipe.top

    # Calculate the center of the gap
    gap_center = (gap_top + gap_bottom) / 2

    # Calculate the distance from the bird's center to the center of the gap
    bird_center = bird.y + bird.height / 2
    distance_from_center = abs(bird_center - gap_center)

    # Normalize the distance based on the size of the gap
    gap_size = gap_bottom - gap_top
    normalized_distance = distance_from_center / gap_size

    # Calculate the reward
    if bird.y > gap_top and (bird.y + bird.height) < gap_bottom:
        # Reward for being within the gap
        reward = 1 - normalized_distance  # Closer to 0 means better (center of the gap)
    else:
        # Penalty for not being within the gap
        reward = -1 - normalized_distance  # More negative as the bird is farther from the center

    return reward

def create_pipe():
    height = random.randint(100, 400)
    top_pipe = pygame.Rect(WIDTH, 0, pipe_width, height)
    bottom_pipe = pygame.Rect(WIDTH, height + pipe_gap, pipe_width, HEIGHT - height - pipe_gap)
    return top_pipe, bottom_pipe

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
    for top_pipe, bottom_pipe in pipes:
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

def collect_training_data():
    global bird_velocity
    training_data = []
    reset_game()
    running = True
    frame_count = 0  # Frame counter for debugging

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
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

        # Collision detection and reward calculation
        reward = get_reward(bird, pipes)
        state = get_state(bird, pipes, bird_velocity)
        training_data.append((state, reward))

        # Print state and reward for debugging
        print(f"Frame {frame_count} - State: {state}, Reward: {reward}")

        if reward == -1:
            if hasattr(pygame, 'mixer'):
                collision_sound.play()
            running = False

        # Draw the bird and pipes
        draw_bird_and_pipes(bird, pipes)
        clock.tick(30)

        frame_count += 1  # Increment frame counter

    return training_data

def train_model(model, training_data, epochs=10):
    states, rewards = zip(*training_data)
    states = np.array(states)
    rewards = np.array(rewards)
    model.fit(states, rewards, epochs=epochs)
    model.save(os.path.join(model_dir,'model.keras'))

def automatic_play():
    global bird_velocity
    COOLDOWN_PERIOD = 0# Cooldown period in seconds
    last_jump_time = 0
    if(os.path.join(model_dir,'model.keras')):
        model=tf.keras.models.load_model(os.path.join(model_dir,'model.keras'))
        print("model loaded")
    reset_game()
    t_act=np.array([[.0],[.0]])
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                pygame.quit()
                sys.exit()

        # Get state and predict action
        state = get_state(bird, pipes, bird_velocity)
        action = model.predict(state.reshape(1, 8), verbose=0)[0][0]
        print(f"S:{state} A:{action}")
        if(t_act[0]>10):
            # print(t_act[1]/10)
            t_act[0]=0
            t_act[1]=0
        t_act[0]+=1
        # print(t_act)
        t_act[1]+=action
        # Bird movement based on action
        current_time = time.time()
        if action >0.7  and (current_time - last_jump_time) > COOLDOWN_PERIOD:
            bird_velocity = -8
            last_jump_time = current_time  

            if hasattr(pygame, 'mixer'):
                jump_sound.play()

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
        if bird.colliderect(pipes[0][0]) or bird.colliderect(pipes[0][1]) or bird.y > HEIGHT or bird.y<0:
            if hasattr(pygame, 'mixer'):
                collision_sound.play()
            running = False

        # Draw the bird and pipes
        draw_bird_and_pipes(bird, pipes)
        clock.tick(30)

def main():
    # Collect training data from multiple sessions
    all_training_data = []
    for i in range(5):  # Collect data from fewer runs for simplicity
        print(f"Collecting data run {i+1}")
        all_training_data.extend(collect_training_data())

    # Train the model with collected data
    print("Training the model")
    train_model(model, all_training_data, epochs=30)  # Reduce epochs for initial testing
def interpolate_color(color1, color2, t):
    """Interpolate between two colors."""
    r1, g1, b1 = color1
    r2, g2, b2 = color2
    r = int(r1 + (r2 - r1) * t)
    g = int(g1 + (g2 - g1) * t)
    b = int(b1 + (b2 - b1) * t)
    return (r, g, b)

if __name__ == "__main__":
    pygame.init()

    # Colors
    WHITE = (255, 255, 255)
    YELLOW = (255, 255, 0)
    GREEN = (0, 255, 0)
    BLACK = (0, 0, 0)

    game_font = pygame.font.Font(font_path, 32)
    game_font_s = pygame.font.Font(font_path, 16)
    
    Train_text = game_font.render('Train', True, WHITE)
    Test_text = game_font.render('Test', True, WHITE)
    Train_text_s = game_font.render('Train', True, (255, 255, 105))
    Test_text_s = game_font.render('Test', True, (255, 255, 105))
    Quit_text_s = game_font_s.render('Esc to quit', True, YELLOW)
    Arrow_text = game_font_s.render('Use Arrow Keys', True, YELLOW)
    
    # Load images

    
    background_x = 0
    background_image_i = bg_img.convert()
    background_image = pygame.transform.scale(background_image_i, (background_image_i.get_width() * 3, background_image_i.get_height() * 3))
    background_width, background_height = background_image.get_size()
    
    # Initialize Pygame
    pygame.display.set_caption("Flappy Bird Menu")
    clock = pygame.time.Clock()

    play_select = True
    reset_game()
    menu_running = True
    
    # Transition settings
    transition_time = 1.0  # Duration of the color transition in seconds
    start_time = pygame.time.get_ticks()
    
    while menu_running:
        if(not pygame.mixer.get_busy()):
            bg_music.set_volume(0.3)
            bg_music.play()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN and (event.key == pygame.K_DOWN or event.key == pygame.K_UP):
                play_select = not play_select
                if hasattr(pygame, 'mixer'):
                    button_sound.play()  # Replace with your sound file
            if event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN:
                if play_select:
                    main()
                else:
                    automatic_play()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                pygame.quit()
                sys.exit()
        
        # Background and color transition
        screen.fill(BLACK)
        elapsed_time = (pygame.time.get_ticks() - start_time) / 1000.0
        t = (elapsed_time % transition_time) / transition_time
        
        # Interpolate colors for dynamic text
        color_train = interpolate_color(GREEN, WHITE, t)
        color_test = interpolate_color(YELLOW, WHITE, t)
        
        Train_text_s = game_font.render('Train', True, color_train)
        Test_text_s = game_font.render('Test', True, color_test)
        
        # Draw menu
        screen.blit(pygame.transform.smoothscale(train_button, (train_button.get_width() // 4, train_button.get_height() // 4)), (100, 200))
        screen.blit(pygame.transform.smoothscale(train_button, (train_button.get_width() // 4, train_button.get_height() // 4)), (100, 300))
        screen.blit(Quit_text_s, (10, 10))
        screen.blit(Arrow_text, (screen.get_width() - 120, 10))
        screen.blit(Train_text_s, (165, 220)) if play_select else screen.blit(Train_text, (165, 220))
        screen.blit(Test_text_s, (172, 320)) if not play_select else screen.blit(Test_text, (172, 320))
        
        pygame.display.flip()
        clock.tick(30)
