import os
import torch
import random
import numpy as np
from tqdm import tqdm
from AlphaZero.monte_carlo_tree_search import MCTS

class AlphaZero:
    def __init__(self, model, optimizer, game_env, args):
        self.model = model
        self.optimizer = optimizer
        self.game_env = game_env
        self.args = args
        self.mcts = MCTS(game=game_env, args=args, model=model)
    
    def selfPlay(self):
        memory = []
        player = 1
        state = self.game_env.initialize()

        while True:
            flip_state = self.game_env.change_perspective(state, player)
            action_probs = self.mcts.search(flip_state)

            memory.append((flip_state, action_probs, player))
            
            action_probs_mod = action_probs ** (1 / self.args['temperature'])
            action_probs_mod /= np.sum(action_probs_mod)
            action = np.random.choice(self.game_env.action_size, p=action_probs_mod)

            state = self.game_env.get_next_state(state, action, player)

            value, terminated = self.game_env.get_value_and_terminated(state, action)

            if terminated:
                gameHist = []
                for flipped_state, action_prob, h_player in memory:
                    move_val = value if h_player==player else self.game_env.get_opponent_value(value)
                    gameHist.append((self.game_env.get_encoded_state(flipped_state), move_val, action_prob))
                return gameHist
        
            player = self.game_env.get_opponent(player)
    
    def train(self, gamebuffer):
        random.shuffle(gamebuffer)
        for batch_Idx in range(0 ,len(gamebuffer), self.args['batch_size']):
            batch = gamebuffer[batch_Idx:min((len(gamebuffer) - 1), (batch_Idx + self.args['batch_size']))]
            states, values, action_probs = zip(*batch)

            states, values, action_probs = np.array(states), np.array(values).reshape(-1, 1), np.array(action_probs)

            states = torch.tensor(states, dtype=torch.float32, device=self.model.device)
            values = torch.tensor(values, dtype=torch.float32, device=self.model.device)
            action_probs = torch.tensor(action_probs, dtype=torch.float32, device=self.model.device)

            pred_val, pred_action_prob = self.model(states)

            loss_val = torch.nn.functional.mse_loss(pred_val, values)
            loss_policy = torch.nn.functional.cross_entropy(pred_action_prob, action_probs)
            net_loss = loss_val + loss_policy

            self.optimizer.zero_grad()
            net_loss.backward()
            self.optimizer.step()
    
    def learn(self, save_path=None):
        for iter in range(self.args['n_iters']):
            print(f"Iter {iter+1} of {self.args['n_iters']}")
            hist_play = []

            print("Self Play:")
            self.model.eval()
            for _ in tqdm(range(self.args['n_selfPlay'])):
                hist_play += self.selfPlay()
            
            print("Model Train:")
            self.model.train()
            for epoch in tqdm(range(self.args['epochs'])):
                self.train(hist_play)
            
            if save_path==None:
                path = os.path.abspath(os.path.join(os.getcwd())) + '/saved_models'
            else:
                path=save_path
            
            os.makedirs(path, exist_ok=True)
            
            torch.save(self.model.state_dict(), f"{path}/model_{iter}_{self.game_env}.pt")
            torch.save(self.optimizer.state_dict(), f"{path}/optimizer_{iter}_{self.game_env}.pt")
            print(f"Model saved at: {path}/model_{iter}_{self.game_env}.pt")