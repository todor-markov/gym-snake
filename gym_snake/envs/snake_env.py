import numpy as np
import gym
from gym import spaces
from gym.utils import seeding
from gym.envs.classic_control import rendering
from collections import deque

GRID_W = 40
GRID_H = 40
GRID_CELL_H = 5
GRID_CELL_W = 5

SNAKE_GEN_W_RANGE = (2, GRID_W - 2)
SNAKE_GEN_H_RANGE = (2, GRID_H - 2)

START_NUM_FRUIT = 20
FRUIT_SPAWN_PROB = 0.1

DYING_REWARD = -100
FRUIT_REWARD = 1
NORM_REWARD = 0

SNAKE_HEAD_COLOR = [255, 96, 0]
SNAKE_BODY_COLOR = [255, 0, 0]
FRUIT_COLOR = [0, 255, 0]


class SnakeEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.seed()
        self.snake = None
        self.snake_direction = None
        self.fruit = None
        self.viewer = None

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(GRID_H * GRID_CELL_H, GRID_W * GRID_CELL_W, 3),
            dtype=np.uint8)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _get_adjacent_cell_coordinates(self, cell, direction):
        if direction == 0:
            adjacent_cell = (cell[0] - 1, cell[1])
        elif direction == 1:
            adjacent_cell = (cell[0], cell[1] + 1)
        elif direction == 2:
            adjacent_cell = (cell[0] + 1, cell[1])
        elif direction == 3:
            adjacent_cell = (cell[0], cell[1] - 1)

        if ((adjacent_cell[0] < 0 or
             adjacent_cell[1] < 0 or
             adjacent_cell[0] >= GRID_H or
             adjacent_cell[1] >= GRID_W)):
            return None
        else:
            return adjacent_cell

    def _generate_snake(self):
        initial_cell = (
            np.random.randint(SNAKE_GEN_W_RANGE[0], SNAKE_GEN_W_RANGE[1]),
            np.random.randint(SNAKE_GEN_H_RANGE[0], SNAKE_GEN_H_RANGE[1]))

        snake_direction = np.random.randint(4)
        snake_body = [
            self._get_adjacent_cell_coordinates(initial_cell, snake_direction),
            initial_cell]

        self.snake = deque(snake_body)
        self.snake_direction = snake_direction

        for snake_cell in self.snake:
            if snake_cell in self.fruit:
                self.fruit.remove(snake_cell)

    def _get_state(self):
        state = np.zeros(
            (GRID_H * GRID_CELL_H, GRID_W * GRID_CELL_W, 3),
            dtype=np.uint8)
        for fruit_cell in self.fruit:
            fruit_h = fruit_cell[0] * GRID_CELL_H
            fruit_w = fruit_cell[1] * GRID_CELL_W
            (state[
                fruit_h + 1: fruit_h + GRID_CELL_H,
                fruit_w + 1: fruit_w + GRID_CELL_W,
                :]) = FRUIT_COLOR
        for snake_cell in self.snake:
            color = (
                SNAKE_HEAD_COLOR if snake_cell == self.snake[0]
                else SNAKE_BODY_COLOR)
            snake_cell_h = snake_cell[0] * GRID_CELL_H
            snake_cell_w = snake_cell[1] * GRID_CELL_W
            (state[
                snake_cell_h + 1: snake_cell_h + GRID_CELL_H,
                snake_cell_w + 1: snake_cell_w + GRID_CELL_W,
                :]) = color

        return state

    def _spawn_fruit(self):
        if np.random.uniform() > FRUIT_SPAWN_PROB:
            return

        new_fruit = (
            np.random.randint(GRID_H),
            np.random.randint(GRID_W))

        if new_fruit in self.snake:
            return

        self.fruit.add(new_fruit)

    def step(self, action):
        assert self.action_space.contains(action), (
            "%r (%s) invalid" % (action, type(action)))

        new_snake_direction = (self.snake_direction + action - 1) % 4
        new_snake_head = self._get_adjacent_cell_coordinates(
            self.snake[0], new_snake_direction)

        self.snake_direction = new_snake_direction

        snake_tail = self.snake.pop()
        if new_snake_head is None or new_snake_head in self.snake:
            done = True
            reward = DYING_REWARD
        elif new_snake_head in self.fruit:
            self.fruit.remove(new_snake_head)
            self.snake.appendleft(new_snake_head)
            self.snake.append(snake_tail)
            self._spawn_fruit()
            done = False
            reward = FRUIT_REWARD
        else:
            self.snake.appendleft(new_snake_head)
            self._spawn_fruit()
            done = False
            reward = NORM_REWARD

        self.state = self._get_state()
        return self.state, reward, done, {}

    def reset(self):
        self.fruit = set((
            np.random.randint(GRID_H),
            np.random.randint(GRID_W)) for _ in range(START_NUM_FRUIT))

        self._generate_snake()
        self.state = self._get_state()
        return self.state

    def render(self, mode='human', close=False):
        if self.viewer is None:
            self.viewer = rendering.SimpleImageViewer()

        self.viewer.imshow(self.state)
        return self.viewer.isopen

    def close(self):
        if self.viewer:
            self.viewer.close()
