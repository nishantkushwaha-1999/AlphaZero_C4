# AlphaZero Implementation on Connect 4

This repository contains an implementation of AlphaZero, a reinforcement learning algorithm that combines Monte Carlo Tree Search (MCTS) with deep neural networks, for playing games. The implementation is designed to be flexible and can be applied to various board games. However, this repo focuses on implemented the algorithm to play a modified version of connect4 with popout. Popout allows a player to remove a coin from the bottom of the board based on the players turn.

## Prerequisites

- Python 3.x
- PyTorch
- tqdm
- torchinfo
- pygame

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

The game can be played using the `play.ipynb` file. The game can be played with 8 hsrdness levels - starting from 0 and final level being level 7. Please see below for a couple of points to play the game.

```python
# Specify the MCTS search parameters.
bot_params = {
    "C": 3,
    "num_searches": 1000, # number of searches to be performed within the MCTS

    # Paramters to add Dirichlet noise in the search tree
    "dirichlet_epsilon": 0.0,
    "dirichlet_alpha": 0.71
}

game_c4 = Game(level = 7) # Specify the bot level
```

- "num_searches": Specifies the number of searches to be performed while building the monte carlo tree search
- "dirichlet_epsilon" and "dirichlet_alpha": These parameters adds dirichlet noise while performing the tree search (Exploration vs Explitation)
- `level`: Argument for the setting the bot difficulty level

## Training the Model

The model can be trained again for the game connect4. To train the model you can use the `train.ipynb` method. Refer below to set the training parameters.

```python
game_c4 = Game_C4()

# Training Parameters
args = {
    'board_dim': (game_c4.rows, game_c4.columns),
    'n_actions': game_c4.action_size,
    'n_res_blocks': 10, # Number of Resnet blocks in the model
    'n_hidden': 64, # number of filters in the convolution layers
    
    # MCTS Search paramaters (Explained above)
    'C': 3,
    'num_searches': 1000,
    'dirichlet_epsilon': 0.25,
    'dirichlet_alpha': 0.65,

    # Model Training Parameters
    'lr': 0.01,
    'weight_decay': 0.001,
    'n_iters': 100,
    'epochs': 2,
    'batch_size': 64,
    'temperature': 1.1,

    # Self Play configs
    'n_selfPlay': 150,
    'n_parallel': True,
    'n_parallel_games': 75
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNet_C4(
    board_dim=args['board_dim'],
    n_actions=args['n_actions'],
    n_res=args['n_res_blocks'],
    n_hidden=args['n_hidden'],
    device=device
)
optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])
architecture = summary(model, input_size=(args['batch_size'], 3, game_c4.rows, game_c4.columns), verbose=0)
print(architecture)
```

### Learning

To initiate the learning process, which includes self-play and training:

```python
alphazero = AlphaZero()
alphazero.learn(model, optimizer, game_c4, args,
                save_path="/content/AlphaZero_C4/drive/MyDrive/AlphaZero_C4/saved_models", 
                n_parallel=args['n_parallel'])
```

### Loading and Resuming Training

Being a student myself, I realize the problems related to limited compute resources. That is why I have designed a load and resume functionality in the training process. The training can be resumes by specifying the save path of the files.

```python
game_c4 = Game_C4()

alphazero = AlphaZero()
alphazero.load_and_resume(game_c4, "/Volumes/Storage/Git Repos/AlphaZero_C4/saved_models/", train=True)
```

## Structure

- **AlphaZero.py**: Contains the implementation of the AlphaZero algorithm.
- **model.py**: Defines the neural network model used by AlphaZero.
- **monte_carlo_tree_search.py**: Implements the Monte Carlo Tree Search algorithm.
- **GameEnvironemts**: Directory that contains the game environment.

## Training on Other Games

The algorithm can be applied to other games/environments as well. The directory `GameEnvironments` contains the game logic for connect4 with popout. A different game environment can be codded containing the same methods/functions and the program should automatically train on the new game environement.

## Acknowledgments

This implementation is based on the AlphaZero algorithm proposed by DeepMind in their paper [Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm](https://arxiv.org/abs/1712.01815).