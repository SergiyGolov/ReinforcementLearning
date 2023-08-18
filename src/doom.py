#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# inspiration for first version: https://github.com/keras-rl/keras-rl/blob/master/examples/dqn_atari.py

from common import *
from skimage import transform
import argparse
from time import sleep
from vizdoom import *
import numpy as np
import warnings  # This ignore all the warning messages that are normally printed during the training because of skiimage
warnings.filterwarnings('ignore')


class DoomGameInterface(DoomGame):
    """
    This class interfaces the VizDoom environment with keras-rl
    scenarioCfg: path to doom scenario .cfg file
    scenarioWad: path to doom scenario .wad file
    step_sleep: time in ms to sleep between each step (useful for test mode, otherwise the interval between the steps is too small and a human eye can't see much)

    The methods defined in this class are those used by keras-rl to interact with the environment (more or less like an openAI gym env)
    """

    def __init__(self, scenarioCfg, scenarioWad, step_sleep=0, initial_ammo=100, initial_health=100):
        DoomGame.__init__(self)
        self.initial_ammo = initial_ammo
        self.initial_health = initial_health
        self.current_ammo = initial_ammo
        self.current_health = initial_health
        self.current_hitcount = 0
        self.step_sleep = step_sleep
        self.load_config(scenarioCfg)
        self.set_doom_scenario_path(scenarioWad)

        self.init()
        self.nb_actions = self.get_available_buttons_size()

    def reset(self):
        self.new_episode()
        return self.get_state().screen_buffer

    def step(self, action):
        sleep(self.step_sleep/1000)
        observation = self.get_state().screen_buffer
        reward = self.make_action(action)

        if self.is_episode_finished():
            self.current_ammo = self.initial_ammo
            self.current_health = self.initial_health
            self.current_hitcount = 0
        else:
            observation = self.get_state().screen_buffer
            newAmmo = self.get_game_variable(GameVariable.SELECTED_WEAPON_AMMO)
            newHealth = self.get_game_variable(GameVariable.HEALTH)
            newHitCount = self.get_game_variable(GameVariable.HITCOUNT)

            # Idea comes from https://github.com/flyyufelix/VizDoom-Keras-RL
            if newAmmo < self.current_ammo:
                reward -= 0.1
                self.current_ammo = newAmmo

            if newHealth < self.current_health:
                reward -= 0.1
                self.current_health = newHealth

            if newHitCount > self.current_hitcount:
                reward += 1
                self.current_hitcount = newHitCount

        # the dict is required to interface with keras-rl but is useless in our case
        return observation, reward, self.is_episode_finished(), dict()

    def render(self, mode):
        pass


class DoomProcessor(Processor):
    """This class is used to process the data between keras-rl and the doom environment (observation from env to keras-rl and action from keras-rl to env)"""

    """
    game: any class that inherits from DoomGame (DoomGameInterface in our case)
    input_shape: size in hxw format of the image coming from doom
    image_slice: how to slice the image coming from doom (how much "useless" pixels to remove on the sides): must be a tuple of "slice" objects (first one for height, second for width)
    resized_shape: size to resize the image to feed the network
    """

    def __init__(self, game, input_shape, resized_shape, image_slice):
        super(DoomProcessor, self).__init__()
        self.game = game
        self.input_shape = input_shape
        self.resized_shape = resized_shape
        self.image_slice = image_slice
        self.possible_actions = np.identity(
            game.get_available_buttons_size(), dtype=int).tolist()  # one-hot encoding: [[1,0,0],[0,1,0],[0,0,1]]

    """Observation frames preprocessing: cropping then scaling, the pixel value normalization is been taken care of in the first lambda layer in keras model"""

    def process_observation(self, observation):
        # inspired by https://gist.github.com/simoninithomas/7611db5d8a6f3edde269e18b97fa4d0c
        assert observation.ndim == 2  # (height, width)
        assert observation.shape == self.input_shape

        # Crop the screen (remove part that contains no information)
        # [Up: Down, Left: right]
        cropped_frame = observation[self.image_slice]

        # Resize
        processed_observation = transform.resize(
            cropped_frame, self.resized_shape)

        assert processed_observation.shape == self.resized_shape

        # normalized_observation=processed_observation/255.0 #now it is done in the neural network

        return processed_observation

    def process_action(self, action):
        return self.possible_actions[action]


def doom_experiment():

    doomScenarioName = "defend_the_line"

    scenarioCfg = "doom_scenarios/"+doomScenarioName+".cfg"
    scenarioWad = "doom_scenarios/"+doomScenarioName+".wad"

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--mode', choices=['train', 'test', 'watch'],
        default="train")
    parser.add_argument('--weights', type=str, default=None)
    parser.add_argument('--step-sleep', type=int, default=0)
    parser.add_argument('--test-episodes', type=int,
                        default=100)
    parser.add_argument('--train-episodes', type=int,
                        default=780)  # 750 + 30, 30 is the number of episodes that takes the warmup (thanks to keras-rl implementation, the episodes are counted while warmup)
    parser.add_argument('--warmup-steps', type=int,
                        default=50000)
    parser.add_argument('--wad', type=str, default=scenarioWad)
    parser.add_argument('--image-shape', choices=["78x51", "160x84"], type=str, default="78x51")
    args = parser.parse_args()

    # this method was useful to chain multiple trains and put each model and metrics in a separate folder
    isavedFoldersCount = 0

    for _, dirs, files in os.walk("doom_saves"):
        for dir in dirs:
            if args.image_shape in str(dir):
                isavedFoldersCount += 1

    if isavedFoldersCount > 0:
        save_folder = f"doom_saves/{args.image_shape}-{str(isavedFoldersCount)}"
    else:
        save_folder = f"doom_saves/{args.image_shape}"

    env = DoomGameInterface(scenarioCfg, args.wad, step_sleep=args.step_sleep)

    # (y,x)
    input_shape = (240, 320)

    if args.image_shape == "78x51":
        resized_shape = (51, 78)
        # used in the 78x51 version
        image_slice = (slice(40, -30), slice(30, -30))
    elif args.image_shape == "160x84":
        resized_shape = (84, 160)
        image_slice = (slice(40, -32), slice(None))  # used in 160x84 version

    # resized_shape = (54, 99) #could be used in a next version

    # image_slice = (slice(40,-32), slice(6,-6)) # could be used in a next version (99x54)

    processor = DoomProcessor(env, input_shape, resized_shape, image_slice)

    doom_hyperparameters = {
        'warmup_steps': args.warmup_steps,
        'memory_size': 250000,
        'dueling_network': False,  # if it's set to true/false, it changed the neural network layout # I forgot tu put this boolean to true since the end of november... but the results were satisfying even with this boolean set to false
        'train_episodes': args.train_episodes,
        'test_episodes': args.test_episodes,
        'batch_size': 64,
        'stack_size': 4,
        'max_tau': 10000,
        'alpha': 0.00025,  # =learning rate
        'gamma': 0.95,  # =discount rate
        'epsilon_start': 1.0,
        'epsilon_stop': 0.01,
        'epsilon_test': 0.001,
        'epsilon_decrease_steps': 80000,  # 80'000 was found empircally
        'save_folder': save_folder,
        'mode': args.mode,
        'weights': args.weights,
        'savedFoldersCount': isavedFoldersCount,
        'nb_actions': env.nb_actions,
        'scenario': doomScenarioName,
        'image_shape': args.image_shape
    }

    network_input_shape = (
        doom_hyperparameters['stack_size'],) + resized_shape

    model = Sequential()

    model.add(Lambda(lambda x: x/255.0, input_shape=network_input_shape,
                     output_shape=network_input_shape))

    model.add(Permute((2, 3, 1), input_shape=network_input_shape))

    # first convent
    model.add(Conv2D(32, (8, 8), strides=(
        4, 4), kernel_initializer=keras.initializers.glorot_uniform(), padding="valid"))
    model.add(Activation('elu'))

    # second convent
    model.add(Conv2D(64, (4, 4), strides=(
        2, 2), kernel_initializer=keras.initializers.glorot_uniform(), padding="valid"))
    model.add(Activation('elu'))

    # third convent
    model.add(Conv2D(128, (4, 4), strides=(
        2, 2), kernel_initializer=keras.initializers.glorot_uniform(), padding="valid"))
    model.add(Activation('elu'))

    model.add(Flatten())
    # We don't make the dueling network ourself because it is made by the dqn agent class (it copies the layers before the last one and divides them in different flows)

    model.add(Dense(512, kernel_initializer=keras.initializers.glorot_uniform()))
    model.add(Activation('elu'))

    model.add(Dense(doom_hyperparameters['nb_actions']))
    # "None activation in tf corresponds to linear activation"
    model.add(Activation('linear'))

    experiment(env, processor, model, doom_hyperparameters)


if __name__ == "__main__":
    doom_experiment()
