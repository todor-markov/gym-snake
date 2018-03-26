import gym
import gym_snake
import time

env = gym.make('Snake-v0')
for i_episode in range(1):
    observation = env.reset()
    env.render()
    for t in range(500):
        env.render()
        time.sleep(0.1)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break

env.close()
