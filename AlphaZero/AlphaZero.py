import os
import glob
import torch
import json
import copy
import random
import numpy as np
from tqdm import tqdm
from AlphaZero.model import ResNet_C4
from AlphaZero.monte_carlo_tree_search import MCTS

class SPG:
    def __init__(self, game):
        self.state = game.initialize()
        self.memory = []
        self.root = None
        self.node = None

class AlphaZero:
    def __init__(self):
        self.reload = False
    
    def __initialize(self, model, optimizer, game_env, args, n_parallel=False):
        self.args = args
        self.model = model
        self.optimizer = optimizer
        self.game_env = game_env
        self.n_parallel = n_parallel

        if not self.reload:
            self.args['latest_model_path'] = ''
            self.args['latest_args_path'] = ''
            self.args['iters_completed'] = 0
        
        self.mcts = MCTS(game=game_env, args=args, model=model)
        self.net_model_loss = np.inf

    def selfPlay(self, pbar):
        memory = []
        player = 1
        state = self.game_env.initialize()

        while True:
            flip_state = self.game_env.change_perspective(state, player)
            # action_probs = self.mcts.search(flip_state)
            action_probs = self.mcts.search(flip_state, player)

            memory.append((flip_state, action_probs, player))
            
            action_probs_mod = action_probs ** (1 / self.args['temperature'])
            action_probs_mod /= np.sum(action_probs_mod)
            action = np.random.choice(self.game_env.action_size, p=action_probs_mod)

            state = self.game_env.get_next_state(state, action, player)
            value, terminated = self.game_env.get_value_and_terminated(state, action, player)
            pbar.set_description(f"{np.sum(state==0), terminated} empty spaces remaining")

            if terminated:
                gameHist = []
                for flipped_state, action_prob, h_player in memory:
                    move_val = value if h_player==player else self.game_env.get_opponent_value(value)
                    gameHist.append((self.game_env.get_encoded_state(flipped_state), move_val, action_prob))
                return gameHist
        
            player = self.game_env.get_opponent(player)
    
    def selfParallelPlay(self, pbar):
        gameHist = []
        player = 1
        selfPlayGames = [SPG(self.game_env) for _ in range(self.args['n_parallel_games'])]
        
        while len(selfPlayGames) > 0:
            s = 0
            stack = []
            for selfPlayGame in selfPlayGames:
                stack.append(selfPlayGame.state)
                s += np.sum(selfPlayGame.state == 0)
            
            pbar.set_description(f"{s} fills remaining: ")
            
            states = np.stack(stack)
            flipped_states = self.game_env.change_perspective(states, player)
            self.mcts.search(flipped_states, player, n_prallel=True, selfPlayGames=selfPlayGames)

            for i in range(len(selfPlayGames))[::-1]:
                selfPlayGame = selfPlayGames[i]

                action_probs = np.zeros(self.game_env.action_size)
                for child in selfPlayGame.root.children:
                    action_probs[child.action_taken] = child.visit_count
                action_probs /= np.sum(action_probs)

                selfPlayGame.memory.append((selfPlayGame.root.state, action_probs, player))
                
                action_probs_mod = action_probs ** (1 / self.args['temperature'])
                action_probs_mod /= np.sum(action_probs_mod)
                action = np.random.choice(self.game_env.action_size, p=action_probs_mod)

                selfPlayGame.state = self.game_env.get_next_state(selfPlayGame.state, action, player)
                value, terminated = self.game_env.get_value_and_terminated(selfPlayGame.state, action, player)
                if terminated:
                    for flipped_states_h, action_probs_h, player_h in selfPlayGame.memory:
                        move_vals = value if player_h == player else self.game_env.get_opponent_value(value)
                        gameHist.append((self.game_env.get_encoded_state(flipped_states_h), move_vals, action_probs_h))
                    del selfPlayGames[i]
            
            player = self.game_env.get_opponent(player)
        return gameHist
    
    def train(self, gamebuffer, epoch):
        random.shuffle(gamebuffer)
        net_loss = 0
        for batch_Idx in (pbar:= tqdm(range(0, len(gamebuffer), self.args['batch_size']))):
            pbar.set_description(f"Epoch {epoch}/{self.args['epochs']}")
            batch = gamebuffer[batch_Idx:min((len(gamebuffer) - 1), (batch_Idx + self.args['batch_size']))]
            try:
                states, values, action_probs = zip(*batch)

                states, values, action_probs = np.array(states), np.array(values).reshape(-1, 1), np.array(action_probs)

                states = torch.tensor(states, dtype=torch.float32, device=self.model.device)
                values = torch.tensor(values, dtype=torch.float32, device=self.model.device)
                action_probs = torch.tensor(action_probs, dtype=torch.float32, device=self.model.device)

                pred_val, pred_action_prob = self.model(states)

                loss_val = torch.nn.functional.mse_loss(pred_val, values)
                loss_policy = torch.nn.functional.cross_entropy(pred_action_prob, action_probs)
                self.net_model_loss = loss_val + loss_policy

                self.optimizer.zero_grad()
                self.net_model_loss.backward()
                self.optimizer.step()

                net_loss += self.net_model_loss.item() * self.args['batch_size']
                pbar.set_description(f"Epoch {epoch}/{self.args['epochs']} - batch_loss: {self.net_model_loss.item()}")
            except ValueError:
                pass
        
        pbar.set_description(f"Epoch {epoch}/{self.args['epochs']} - loss: {net_loss / len(gamebuffer)}")
    
    def learn(self, model, optimizer, game_env, args, save_path=None, n_parallel=False):
        if not self.reload:
            print("Initializing...")
            self.__initialize(model, optimizer, game_env, args, n_parallel=n_parallel)

        for iter in range(self.args['iters_completed'] + 1, self.args['n_iters'] + 1):
            print(f"Iter {iter} of {self.args['n_iters']}")
            hist_play = []

            self.model.eval()
            if self.n_parallel:
                if (self.args['n_selfPlay'] % self.args['n_parallel_games']) != 0:
                    raise ValueError(f"'n_selfPlay' should be a multple of 'n_parallel_games'")
                
                print(f"Self Play: Playing {self.args['n_parallel_games']} games parallely")
                for _ in (pbar:= tqdm(range(self.args['n_selfPlay'] // self.args['n_parallel_games']))):
                    hist_play += self.selfParallelPlay(pbar)
            else:
                print(f"Self Play: Playing one game at a time")
                for _ in (pbar:= tqdm(range(self.args['n_selfPlay']))):
                    hist_play += self.selfPlay(pbar)
            
            print("Model Train:")
            self.model.train()
            for epoch in range(self.args['epochs']):
                self.train(hist_play, epoch)
            
            if save_path==None:
                path = os.path.abspath(os.path.join(os.getcwd())) + '/saved_models'
            else:
                path=save_path
            
            self.args['save_path'] = save_path
            os.makedirs(path, exist_ok=True)
            
            torch.save(self.model.state_dict(), f"{path}/model_{iter}_{self.game_env}.pt")
            torch.save(self.optimizer.state_dict(), f"{path}/optimizer_{iter}_{self.game_env}.pt")
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss': self.net_model_loss
            }, f"{path}/fullModel_{iter}_{self.game_env}.pt")
            
            self.args['latest_model_path'] = f"{path}/fullModel_{iter}_{self.game_env}.pt"
            self.args['latest_args_path'] = f'{path}/model_args_{self.game_env}.json'
            self.args['iters_completed'] = iter
            self.args['save_path'] = path
            
            with open(f'{path}/model_args_{self.game_env}.json', 'w') as fp:
                json.dump(self.args, fp)
            
            print(f"Model saved at: {path}/****_{iter}_{self.game_env}.pt")
    
    def load_and_resume(self, game_env, path, train=False):
        self.reload = True
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        arg_files = glob.glob(f"{path}/*.json")
        with open(arg_files[0], 'r') as fp:
            args = json.loads(fp.read())
        
        model = ResNet_C4(
            board_dim=args['board_dim'],
            n_actions=args['n_actions'],
            n_res=args['n_res_blocks'],
            n_hidden=args['n_hidden'],
            device=device
        )
        
        optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])
        
        checkpoint = torch.load(args['latest_model_path'])
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        self.net_model_loss = checkpoint['loss']

        if train:
            self.__initialize(model, optimizer, game_env, args, n_parallel=args['n_parallel'])
            print("Resuming Training:")
            self.learn(
                model = self.model,
                optimizer = self.optimizer,
                game_env = self.game_env,
                args = self.args,
                save_path=args['save_path'],
                n_parallel=args['n_parallel']
            )
        else:
            return model