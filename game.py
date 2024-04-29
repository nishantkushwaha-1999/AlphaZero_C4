import torch
import json
import glob
import numpy as np
from typing import Union
from AlphaZero.monte_carlo_tree_search import MCTS
from GameEnvironments.C4 import Game_C4
from AlphaZero.model import ResNet_C4
from utils.setup import check_base_models


class Game(Game_C4):
    def __init__(self, level: Union[int, None]=None):
        super().__init__()
        if level != None:
            self.engine = self.load_engine(level)

    def load_engine(self, level: int):
        check_base_models("Game Engines")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        arg_files = glob.glob(f"Game Engines/*.json")
        with open(arg_files[0], 'r') as fp:
            args = json.loads(fp.read())
        
        model = ResNet_C4(
            board_dim=(self.rows, self.columns),
            n_actions=self.action_size,
            n_res=args['n_res_blocks'],
            n_hidden=args['n_hidden'],
            device=device
        )

        checkpoint = torch.load(f"Game Engines/Level{level}.pt")
        model.load_state_dict(checkpoint['model_state_dict'])
        return model.eval()
    
    def play(self):
        player = 1
        action = None
        game_c4 = Game_C4()
        
        mcts_params = {
                            "C": 3,
                            "num_searches": 600,
                            "dirichlet_epsilon": 0.0,
                            "dirichlet_alpha": 0.71
                        }
        mcts = MCTS(game_c4, mcts_params, self.engine)
        state = game_c4.initialize()

        while True:
            if player == 1:
                valid_moves = game_c4.get_valid_moves(state, player)
                print("valid_moves", [i for i in range(game_c4.action_size) if valid_moves[i] == 1])
                action = int(input(f"{player}:"))
                player_name = "player"
                    
            else:
                flipped_state = game_c4.change_perspective(state, player)
                mcts_probs = mcts.search(flipped_state, player)
                action = np.argmax(mcts_probs)
                player_name = "bot"
                
            state = game_c4.get_next_state(state, action, player)
            
            value, is_terminal = game_c4.get_value_and_terminated(state, action, player)
            
            if is_terminal:
                if value == 1:
                    print(player, "won")
                else:
                    print("draw")
                break
                
            player = game_c4.get_opponent(player)
            print("Last Play:")
            print("Player:", player_name, "Action:", action)
    

if __name__=="__main__":
    gm = Game(level=1)