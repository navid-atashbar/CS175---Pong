
import ale_py
import gymnasium as gym
import pygame
import random
gym.register_envs(ale_py)

env = gym.make("ALE/Pong-v5", render_mode="human")
obs, info = env.reset()

clock = pygame.time.Clock()
done = False
score = 0

print("Use UP and DOWN arrow keys to play!")
print("Close the window to quit.")

while not done:
    action = 0  # default: stay still
    keys = random.choice([0,2,3])
    
    if keys == 2:
        action = 2  # move up
    elif keys ==3:
        action = 3  # move down

    # handle quit
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True

    obs, reward, terminated, truncated, info = env.step(action)

    if reward == 1:
        print("You scored!")
    elif reward == -1:
        print("Opponent scored!")

    done = terminated or truncated
    clock.tick(30)  # 30 FPS

print("Game over!")
env.close()
