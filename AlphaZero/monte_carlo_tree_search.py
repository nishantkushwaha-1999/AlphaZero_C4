import math
import torch
import numpy as np

class Node:
    def __init__(self, game, args, state, player, parent=None, action_taken=None, prior=0, visit_count=0):
        self.game = game
        self.args = args
        self.state = state
        self.parent = parent
        self.action_taken = action_taken
        self.prior = prior
        self.player = player
        
        self.children = []
        
        self.visit_count = visit_count
        self.value_sum = 0
        
    def is_fully_expanded(self):
        return len(self.children) > 0
    
    def select(self):
        best_child = None
        best_ucb = -np.inf
        
        for child in self.children:
            ucb = self.get_ucb(child)
            if ucb > best_ucb:
                best_child = child
                best_ucb = ucb
                
        return best_child
    
    def get_ucb(self, child):
        if child.visit_count == 0:
            q_value = 0
        else:
            q_value = 1 - ((child.value_sum / child.visit_count) + 1) / 2
        return q_value + self.args['C'] * (math.sqrt(self.visit_count) / (child.visit_count + 1)) * child.prior
    
    def expand(self, policy):
        for action, prob in enumerate(policy):
            if prob > 0:
                child_state = self.state.copy()
                child_state = self.game.get_next_state(child_state, action, 1)
                child_state = self.game.change_perspective(child_state, player=-1)

                child = Node(self.game, self.args, child_state, -1, self, action, prob)
                self.children.append(child)
                
        return child
            
    def backpropagate(self, value):
        self.value_sum += value
        self.visit_count += 1
        
        value = self.game.get_opponent_value(value)
        if self.parent is not None:
            self.parent.backpropagate(value)  


class MCTS:
    def __init__(self, game, args, model):
        self.game = game
        self.args = args
        self.model = model
        
    @torch.no_grad()
    def search(self, state, player, n_prallel=False, selfPlayGames=None):
        if n_prallel == False:
            root = Node(self.game, self.args, state, player, visit_count=1)
            
            _, policy = self.model(
                torch.tensor(self.game.get_encoded_state(state), device=self.model.device).unsqueeze(0)
            )
            policy = torch.softmax(policy, axis=1).squeeze(0).cpu().numpy()
            policy = (1 - self.args['dirichlet_epsilon']) * policy + self.args['dirichlet_epsilon'] \
                * np.random.dirichlet([self.args['dirichlet_alpha']] * self.game.action_size)
        
            valid_moves = self.game.get_valid_moves(state, player)
            policy *= valid_moves
            policy /= np.sum(policy)
            root.expand(policy)
            
            for search in range(self.args['num_searches']):
                node = root
                
                while node.is_fully_expanded():
                    node = node.select()
                    
                value, is_terminal = self.game.get_value_and_terminated(node.state, node.action_taken, node.player)
                value = self.game.get_opponent_value(value)
                
                if not is_terminal:
                    value, policy = self.model(
                        torch.tensor(self.game.get_encoded_state(node.state), device=self.model.device).unsqueeze(0)
                    )
                    policy = torch.softmax(policy, axis=1).squeeze(0).cpu().numpy()
                    valid_moves = self.game.get_valid_moves(node.state, node.player)
                    policy *= valid_moves
                    policy /= np.sum(policy)
                    
                    value = value.item()
                    
                    node.expand(policy)
                    
                node.backpropagate(value)    
                
                
            action_probs = np.zeros(self.game.action_size)
            for child in root.children:
                action_probs[child.action_taken] = child.visit_count
            action_probs /= np.sum(action_probs)
            return action_probs
        
        elif n_prallel == True:
            if type(selfPlayGames) != list:
                raise ValueError(f"selfPlayGames need to be an array when n_parallel is active. Got {type(selfPlayGames)}")
            
            _, policy = self.model(
                torch.tensor(self.game.get_encoded_state(state), device=self.model.device)
            )
            policy = torch.softmax(policy, axis=1).cpu().numpy()
            policy = (1 - self.args['dirichlet_epsilon']) * policy + self.args['dirichlet_epsilon'] \
                * np.random.dirichlet([self.args['dirichlet_alpha']] * self.game.action_size, size=policy.shape[0])
            
            for i, selfPlayGame in enumerate(selfPlayGames):
                valid_moves = self.game.get_valid_moves(state[i], player)
                gamepolicy = policy[i]
                gamepolicy *= valid_moves
                gamepolicy /= np.sum(gamepolicy)

                selfPlayGame.root = Node(self.game, self.args, state[i], player=player, visit_count=1)
                selfPlayGame.root.expand(gamepolicy)

            for search in range(self.args['num_searches']):
                for selfPlayGame in selfPlayGames:
                    selfPlayGame.node = None
                    node = selfPlayGame.root

                    while node.is_fully_expanded():
                        node = node.select()
                    
                    value, is_terminal = self.game.get_value_and_terminated(node.state, node.action_taken, node.player)
                    value = self.game.get_opponent_value(value)

                    if is_terminal:
                        node.backpropagate(value)
                    else:
                        selfPlayGame.node = node
            
                exp_selfPlayGames = [mapIdx for mapIdx in range(len(selfPlayGames)) if selfPlayGames[mapIdx].node != None]

                if len(exp_selfPlayGames) > 0:
                    state = np.stack([selfPlayGames[mapIdx].node.state for mapIdx in exp_selfPlayGames])
                    value, policy = self.model(
                        torch.tensor(self.game.get_encoded_state(state), device=self.model.device)
                    )
                    policy = torch.softmax(policy, axis=1).cpu().numpy()
                    value = value.cpu().numpy()
                
                for i, mapIdx in enumerate(exp_selfPlayGames):
                    node = selfPlayGames[mapIdx].node
                    game_value, gamepolicy = value[i], policy[i]
                    
                    valid_moves = self.game.get_valid_moves(node.state, node.player)
                    gamepolicy *= valid_moves
                    gamepolicy /= np.sum(gamepolicy)

                    node.expand(gamepolicy)
                    node.backpropagate(game_value)