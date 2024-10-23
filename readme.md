# Flappy Bird AI using Python and NEAT
This project is a replica of the famous Flappy Bird game developed using Python and Pygame. The twist is that you can choole to let an AI, trained using the NEAT (NeuroEvolution of Augmenting Topologies) algorithm, play the game instead of a human player.

## Tech
This project was built using:
- Python 3.10+
- Pygame 2.6.1+
- NEAT-Python

## How to Run
Make sure you are in the root directory of the project.
Run the following command to start the game: `python flappy_bird_game.py`

## Gameplay Controls
**A key**: Start the AI training from generation 1.
**Mouse left click** or **Space Bar**: Jump into manual gameplay for some Flappy Bird fun.
**D Key**: Enter Debug mode
**R Key**: Restart game upon losing

## AI Overview
The AI is trained using the NEAT algorithm, which evolves neural networks to play the game. The algorithm mutates and adjusts the networks to improve fitness, which in this case is based on how far the bird travels and how long it survives.

### Configuration
The NEAT configuration is handled in the config-neat.cfg file. If you want to modify the settings for the NEAT algorithm, such as mutation rates, population size, or hidden layer settings, you can edit the config-neat.cfg file.
