# Installation

- navigate to `src` folder
- `python3 -m venv venv_rl`
- `source venv_rl/bin/activate`
- `pip install -r requirements.txt`
- All the commands in this user guide must be executed in the $src$ folder (where `doom.py` and `flappybird.py` are located)

# Tensorboard logs location

- they are located in xxx_saves/yyy/logs/zzz
    - xxx is either "doom" or "flappybird"
    - yyy is the experiment name
    - zzz is either "train" or "testN" with N empty or a number

# Model weights location

- they are located in xxx_saves/yyy/weights.h5f
    - xxx is either "doom" or "flappybird"
    - yyy is the experiment name

# Doom (doom.py)

## Command line arguments

All those arguments have default values except "weights"

- mode: "test" "train" or "watch", if set to "watch" or "test" and no weights argument specified, it will use the last trained model (default: train)
- weights: path to trained model weights file (.h5f format) (only test and watch mode, default: none)
- step-sleep: number of ms to sleep between each step (useful to observe for a human the agent playing, default: 0)
- test-episodes: number of episodes to test (only test and watch mode, default: 100)
- train-episodes: number of episodes to train (only train mode, default: 780)
- warmup-steps: number of steps for warmup (only train mode, default: 50000)
- wad: path to wad file containing doom scenario (default: doom_scenarios/defend_the_line.wad)
- image-shape: shape of the image to be put in the neural network, 78x51 or 160x84 (default: 78x51)

## How to watch interesting trained models at ~60 fps without tensorboard logging by loading weight file

- one shot ennemy kill with 160x84 image shape:
    - `./doom.py --mode=watch --weights=doom_saves/160x84/weights.h5f --step-sleep=15 --wad=doom_scenarios/defend_the_line-oneShotKill.wad --image-shape=160x84`
- one shot ennemy kill with 76x51 image shape (sometimes the agent last longer than in the 160x84 version and sometimes his behaviour is messy):
    - `./doom.py --mode=watch --weights=doom_saves/oneShotKill/weights.h5f --step-sleep=15 --wad=doom_scenarios/defend_the_line-oneShotKill.wad --image-shape=78x51`
- normal scenario with 160x84 image shape:
    - `./doom.py --mode=watch --weights=doom_saves/160x84/weights.h5f --step-sleep=15 --image-shape=160x84`
- normal scenario with 76x51 image shape:
    - `./doom.py --mode=watch --weights=doom_saves/78x51/weights.h5f --step-sleep=15 --image-shape=78x51`

## How to train + test a new model

- creates a new folder named doom_saves/{saveFolder}-{x} where x is a number and saveFolder "78x51" for example
- example: 78x51 image size, normal scenario, 100 train episodes, 20000 warmup steps, 20 test episodes
    - `./doom.py --mode=train --train-episodes=100 --warmup-steps=20000 --test-episodes=20 --image-shape=78x51`

## Re test previously trained model (from previous example)

- creates a new folder named doom_saves/{saveFolder}/logs/test{x} folder where x is a number and saveFolder "78x51-1" for example
- example: re test previous trained model:
    - `./doom.py --mode=test --test-episodes=20 --image-shape=78x51`

## Test a model by loading weight file and log test

- creates a new folder named doom_saves/{saveFolder}/logs/test{x} folder where x is a number and saveFolder "78x51-1" for example, the saveFolder is parsed from the weight file path (if the weight file is located at foo/toto/tutu/weights.h5f, saveFolder = foo/toto/tutu)
- example: load and test previously trained model `doom_saves/78x51-1/weights.h5f`
    - `./doom.py --mode=test --test-episodes=2 --image-shape=78x51 --weights=doom_saves/78x51-1/weights.h5f`

# Flappy bird (flappybird.py)

## Command line arguments

All those arguments have default values except "weights"

- mode: "test" "train" or "watch", if set to "watch" or "test" and no weights argument specified, it will use the last trained model (default: train)
- weights: path to trained model weights file (.h5f format) (only test and watch mode, default: none)
- step-sleep: number of ms to sleep between each step (useful to observe for a human the agent playing, default: 0)
- test-episodes: number of episodes to test (only test and watch mode, default: 100)
- train-episodes: number of episodes to train (only train mode, default: 16700)
- warmup-steps: number of steps for warmup (only train mode, default: 100000)
- render: render the graphics of the game or not, if set to "0" do not render, if set to anything else, render

## How to watch interesting trained models at ~60 fps without tensorboard logging by loading weight file

- `./flappybird.py --mode=watch --step-sleep=15 --weights=flappybird_saves/best/weights.h5f`

## How to train + test a new model

- creates a new folder named flappybird_saves/experiment-{x} where x is a number, "flappybird_saves/experiment-1" for example
    - `./flappybird.py --mode=train --test-episodes=50 --train-episodes=250 --warmup-steps=1000`

## Re test previously trained model (from previous example)

- creates a new folder named flappybord_saves/experiment-{y}/logs/test{x} folder where x and y are numbers, "flappybird_saves/experiment-1/logs/test2" for example
    - `./flappybird.py --mode=test --test-episodes=50`

## Test a model by loading weight file and log test

- creates a new folder named flappybird_saves/{saveFolder}/logs/test{x} folder where x is a number and saveFolder "best" for example, the saveFolder is parsed from the weight file path (if the weight file is located at foo/toto/tutu/weights.h5f, saveFolder = foo/toto/tutu)
    - `./flappybird.py --mode=test --weights=flappybird_saves/experiment-1/weights.h5f`
