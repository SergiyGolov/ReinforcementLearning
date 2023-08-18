#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# inspiration for first version: https://github.com/keras-rl/keras-rl/blob/master/examples/dqn_atari.py

from keras.optimizers import RMSprop
import keras.backend as K
import keras
from keras.callbacks import TensorBoard
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from rl_per.agents.dqn import DQNAgent
from rl_per.memory import PrioritizedMemory
from rl.core import Processor
from rl.callbacks import FileLogger, ModelIntervalCheckpoint, Callback
import tensorflow as tf  # for custom tensorboard
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Conv2D, Permute, Lambda
import os


class TestTensorBoard(TensorBoard):
    """This class logs the test results (mean episode reward and mean episode step count) in tensorboard"""

    def __init__(self, log_dir='./logs',
                 histogram_freq=0,
                 batch_size=32,
                 write_graph=True,
                 write_grads=False,
                 write_images=False,
                 embeddings_freq=0,
                 embeddings_layer_names=None,
                 embeddings_metadata=None,
                 embeddings_data=None,
                 update_freq='epoch'):

        super(TestTensorBoard, self).__init__(log_dir,
                                              histogram_freq,
                                              batch_size,
                                              write_graph,
                                              write_grads,
                                              write_images,
                                              embeddings_freq,
                                              embeddings_layer_names,
                                              embeddings_metadata,
                                              embeddings_data,
                                              update_freq)

    def on_train_begin(self, logs):
        """ Print logs at beginning of test"""
        self.rewards = []
        self.episodeSteps = []
        super(TestTensorBoard, self).on_train_begin(logs)

    def on_episode_end(self, episode, logs):
        """ Print logs at end of each episode """
        self.rewards.append(logs['episode_reward'])
        self.episodeSteps.append(logs['nb_steps'])
        super(TestTensorBoard, self).on_epoch_end(episode, logs)

    def on_train_end(self, logs):
        meanReward = sum(self.rewards)/len(self.rewards)
        meanStepsPerEpisode = sum(self.episodeSteps)/len(self.episodeSteps)
        summaryReward = tf.Summary(
            value=[tf.Summary.Value(tag='meanReward', simple_value=meanReward)])
        self.writer.add_summary(summaryReward, 1)
        summaryStepCount = tf.Summary(value=[tf.Summary.Value(
            tag='meanStepsPerEpisode', simple_value=meanStepsPerEpisode)])
        self.writer.add_summary(summaryStepCount, 1)
        self.writer.flush()
        super(TestTensorBoard, self).on_train_end(logs)


class MyLinearAnnealedPolicy(LinearAnnealedPolicy):
    """same as LinearAnnealedPolicy but doesn't reduce epsilon before 'warmup steps' steps"""

    def __init__(self, inner_policy, attr, value_max, value_min, value_test, nb_steps, warmup_steps):
        if not hasattr(inner_policy, attr):
            raise ValueError(
                'Policy does not have attribute "{}".'.format(attr))

        super(LinearAnnealedPolicy, self).__init__()

        self.inner_policy = inner_policy
        self.attr = attr
        self.value_max = value_max
        self.value_min = value_min
        self.value_test = value_test
        self.nb_steps = nb_steps
        self.warmup_steps = warmup_steps

    def get_current_value(self):
        """Return current annealing value

        # Returns
            Value to use in annealing
        """
        if self.agent.training:
            # Linear annealed: f(x) = ax + b.
            a = -float(self.value_max - self.value_min) / float(self.nb_steps)
            b = float(self.value_max)
            value = max(self.value_min, a *
                        float(self.agent.step-self.warmup_steps) + b)
        else:
            value = self.value_test
        return value


"""
envInterface: either instance of class "DoomGameInterface" or "FlappybirdGameInterface" or any other class that interfaces keras-rl with an environment
envProcessor: either instance of class "DoomProcessor" or "FlappybirdProcessor" or any other class that helps keras-rl to process the environement from env to keras-rl or action from keras-rl to env
hyperparameters: dict containing hyperparameters (see doom_experiment for definition)
"""


def experiment(envInterface, envProcessor, model, hyperparameters):

    memory = PrioritizedMemory(
        limit=hyperparameters['memory_size'], window_length=hyperparameters['stack_size'])

    policy = MyLinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps',
                                    value_max=hyperparameters['epsilon_start'],
                                    value_min=hyperparameters['epsilon_stop'],
                                    value_test=hyperparameters['epsilon_test'],
                                    nb_steps=hyperparameters['epsilon_decrease_steps'],
                                    warmup_steps=hyperparameters['warmup_steps'])

    dqn = DQNAgent(model=model, nb_actions=hyperparameters['nb_actions'], memory=memory,
                   processor=envProcessor, nb_steps_warmup=hyperparameters['warmup_steps'],
                   gamma=hyperparameters['gamma'],
                   target_model_update=hyperparameters['max_tau'],
                   enable_double_dqn=True, enable_dueling_network=hyperparameters['dueling_network'],
                   policy=policy, batch_size=hyperparameters['batch_size'])

    dqn.compile(RMSprop(lr=hyperparameters['alpha']), metrics=['mse', 'mae'])

    if hyperparameters['mode'] == 'train':
        weights_filename = f"{hyperparameters['save_folder']}/weights.h5f"
        checkpoint_weights_filename = f"{hyperparameters['save_folder']}/weights.h5f"
        log_filename = f"{hyperparameters['save_folder']}/log.json"
        callbacks = [ModelIntervalCheckpoint(
            checkpoint_weights_filename, interval=2500)]  # 2500 steps ~=5 episodes
        callbacks += [FileLogger(log_filename, interval=2500)]
        callbacks += [TensorBoard(
            log_dir=f"{hyperparameters['save_folder']}/logs/train", write_graph=False)]
        dqn.fit(envInterface, verbose=2, nb_episodes=hyperparameters['train_episodes'],
                callbacks=callbacks)

        # After training is done, we save the final weights one more time.
        dqn.save_weights(weights_filename, overwrite=True)
        callbacks = [TestTensorBoard(
            log_dir=f"{hyperparameters['save_folder']}/logs/test", write_graph=False)]
        # Finally, evaluate our algorithm for test_episodes episodes.
        dqn.test(
            envInterface, nb_episodes=hyperparameters['test_episodes'], callbacks=callbacks)
    elif hyperparameters['mode'] == 'test' or hyperparameters['mode'] == 'watch':
        if hyperparameters['weights']:
            weights_filename = hyperparameters['weights']

            saveFolder = "/".join(weights_filename.split(
                "/")[:-1])

        else:
            # useful if we use the test mode directly afterwards with the same saved model

            isavedFoldersCount = hyperparameters['savedFoldersCount']-1
            if isavedFoldersCount > 0:
                saveFolder = "".join(hyperparameters['save_folder'].split(
                    "-")[:-1])+"-"+str(isavedFoldersCount)  # if folder is in format xxx-4, the folder used would be xxx-3
            else:
                saveFolder = hyperparameters['save_folder']

            weights_filename = f"{saveFolder}/weights.h5f"

        isavedTestFoldersCount = 0

        for _, dirs, files in os.walk(f"{saveFolder}/logs"):
            for dir in dirs:
                isavedTestFoldersCount += 1

        dqn.load_weights(weights_filename)
        callbacks = [TestTensorBoard(
            log_dir=f"{saveFolder}/logs/test{isavedTestFoldersCount}", write_graph=False)]
        callbacks += [TensorBoard(
            log_dir=f"{saveFolder}/logs/test{isavedTestFoldersCount}", write_graph=False)]

        if hyperparameters['mode'] == 'watch':
            callbacks = None

        # notice that thanks to keras-rl implementation, in test mode "nb_episode_steps" becomes "nb_steps" in tensorboard
        dqn.test(
            envInterface, nb_episodes=hyperparameters['test_episodes'], callbacks=callbacks)
