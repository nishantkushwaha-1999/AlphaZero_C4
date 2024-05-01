# AlphaZero Implementation on Connect 4

This repository contains an implementation of AlphaZero, a reinforcement learning algorithm that combines Monte Carlo Tree Search (MCTS) with deep neural networks, for playing games. The implementation is designed to be flexible and can be applied to various board games.

## Prerequisites

- Python 3.x
- PyTorch
- tqdm

## Installation

1. Clone this repository to your local machine:

```bash
git clone https://github.com/yourusername/alphazero.git
```

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Running Self-Play

To generate training data through self-play, you can use the `selfPlay()` method:

```python
from AlphaZero import AlphaZero

# Initialize AlphaZero
az = AlphaZero()

# Start self-play
gameHist = az.selfPlay()
```

### Training the Model

To train the model using the generated training data, you can use the `train()` method:

```python
# Assuming you have collected training data in 'gameHist'
az.train(gameHist, epoch)
```

### Learning

To initiate the learning process, which includes self-play and training:

```python
az.learn(model, optimizer, game_env, args, save_path=None, n_parallel=False)
```

### Loading and Resuming Training

You can load a saved model and resume training using the `load_and_resume()` method:

```python
az.load_and_resume(game_env, path, train=True)
```

## Structure

- **AlphaZero.py**: Contains the implementation of the AlphaZero algorithm.
- **model.py**: Defines the neural network model used by AlphaZero.
- **monte_carlo_tree_search.py**: Implements the Monte Carlo Tree Search algorithm.
- **saved_models**: Directory to save trained models.

## Acknowledgments

This implementation is based on the AlphaZero algorithm proposed by DeepMind in their paper [Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm](https://arxiv.org/abs/1712.01815).