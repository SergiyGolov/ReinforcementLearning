#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from common import *
from ple.games.flappybird import *
from ple import PLE
import numpy as np
import argparse
from time import sleep

class FlappyBirdGameInterface(PLE):
    def __init__(self, step_sleep=0, stack_size=4, render=True):
        self.step_sleep = step_sleep
        self.game = FlappyBird()
        super(FlappyBirdGameInterface, self).__init__(self.game)
        self.display_screen = render
        self.state_dim = len(self.game.getGameState())

        self.state_shape = (stack_size, self.state_dim)

        self.actions = self.getActionSet()
        if self.actions[1] == None:
            self.actions[1] = 0

        self.nb_actions = len(self.actions)

    def reset(self):
        self.reset_game()
        return self.game.getGameState()

    def step(self, action):
        reward = self.act(action)
        observation = self.game.getGameState()

        sleep(self.step_sleep/1000)
        if self.game_over():
            playerY = observation['player_y']
            nextPipeTopY = observation['next_pipe_top_y']
            nextPipeBotY = observation['next_pipe_bottom_y']
            nextPipeDist = observation['next_pipe_dist_to_player']

            if playerY > 0 and nextPipeDist < 62:
                gapMiddleY = (nextPipeBotY-nextPipeTopY)/2+nextPipeTopY
                if playerY < gapMiddleY:
                    newReward = playerY/gapMiddleY
                else:
                    trueHeight = self.game.height*0.79
                    newReward = 1-(playerY-gapMiddleY)/(trueHeight-gapMiddleY)

                reward += newReward

        return observation, reward, self.game_over(), dict()

    def render(self, mode):
        pass


class FlappyBirdProcessor(Processor):
    """This class is used to process the data between keras-rl   and the flappybird environment (observation from env to keras-rl   and action from keras-rl to env)"""

    def __init__(self, env):
        super(FlappyBirdProcessor, self).__init__()
        self.env = env
        self.velSet = set()

    def process_observation(self, state):
        """
        This preprocesses our state from PLE. We rescale the values to be between
        0,1 and -1,1.
        """
        # taken by inspection of source code. Better way is on its way!
        max_values = np.array([self.env.game.height, 20.0, self.env.game.width, self.env.game.height,
                               self.env.game.height, self.env.game.width, self.env.game.height, self.env.game.height])

        state = np.array(list(state.values())) / max_values

        np.clip(state, -1.0, 1.0, out=state)

        return state.flatten()

    def process_action(self, action):
        return self.env.actions[action]


def flappybird_experiment():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--mode', choices=['train', 'test', 'watch'], default='train')
    parser.add_argument('--weights', type=str, default=None)
    parser.add_argument('--test-episodes', type=int,
                        default=100)
    parser.add_argument('--train-episodes', type=int,
                        default=16700)  # 15000 + 1700, 1700 is the number of episodes that takes the warmup (thanks to keras-rl implementation, the episodes are counted while warmup)
    parser.add_argument('--warmup-steps', type=int,
                        default=100000)
    parser.add_argument('--render', type=int,
                        default=1)
    parser.add_argument('--step-sleep', type=int, default=0)
    args = parser.parse_args()

    if args.render == 0:
        render = False
    else:
        render = True

    stack_size = 4 # sorry for putting this here

    isavedFoldersCount = 0

    for _, dirs, files in os.walk("flappybird_saves"):
        for dir in dirs:
            isavedFoldersCount += 1
        break

    if isavedFoldersCount > 0:
        save_folder = f"flappybird_saves/experiment-{str(isavedFoldersCount)}"
    else:
        save_folder = f"flappybird_saves/experiment"


    env = FlappyBirdGameInterface(
        step_sleep=args.step_sleep, stack_size=stack_size, render=render)

    processor = FlappyBirdProcessor(env)

    flappybird_hyperparameters = {
        'warmup_steps': args.warmup_steps,
        'memory_size': 1000000,
        # if it's set to true/false, it changed the neural network layout
        'dueling_network': True,
        'train_episodes': args.train_episodes,
        'test_episodes': args.test_episodes,
        'batch_size': 32,
        'stack_size': stack_size,
        'max_tau': 10000,
        'alpha': 0.01,  # =learning rate
        'gamma': 0.95,  # =discount rate
        'epsilon_start': 0.15,
        'epsilon_stop': 0.01,
        'epsilon_test': 0.001,
        'epsilon_decrease_steps': 150000,  # 150000 was found empirically
        'save_folder': save_folder,
        'mode': args.mode,
        'weights': args.weights,
        'savedFoldersCount': isavedFoldersCount,
        'nb_actions': env.nb_actions,
    }

    model = Sequential()

    model.add(Dense(
        128, input_shape=env.state_shape, activation="relu", kernel_initializer="he_uniform"
    ))

    model.add(Dense(
        256, activation="relu", kernel_initializer="he_uniform"
    ))

    model.add(Dense(
        512, activation="relu", kernel_initializer="he_uniform"
    ))

    model.add(Dense(
        256, activation="relu", kernel_initializer="he_uniform"
    ))

    model.add(Dense(
        128, activation="relu", kernel_initializer="he_uniform"
    ))

    model.add(Flatten())

    model.add(Dense(
        flappybird_hyperparameters['nb_actions'], activation="linear", kernel_initializer="he_uniform"
    ))

    experiment(env, processor, model, flappybird_hyperparameters)


if __name__ == "__main__":
    flappybird_experiment()
