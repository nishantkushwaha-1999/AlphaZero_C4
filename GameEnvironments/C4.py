import sys
import pygame
import numpy as np
import copy
from PIL import Image, ImageDraw

class Game_C4:
    def __init__(self):
        self.columns = 7
        self.rows = 6
        self.action_size = self.columns * 2 # change action dim to double this
        self.n_similar = 4
        self.popout = np.zeros(self.columns)
    
    def __repr__(self):
        return "Connect_4"
    
    def initialize(self):
        return np.zeros((self.rows, self.columns))

    def check_params(self, state: np.array=None, player=None, action=None):
        if player != None:
            if player not in [1, -1]:
                raise ValueError(f"player: {player} not found. Player must be either -1 or 1")
        
        if str(type(state))=="<class 'numpy.ndarray'>":
            if (state.shape[-1] == self.columns) and (state.shape[-2] == self.rows):
                pass
            else:
                raise ValueError(f"Invalid board passed of shape {state.shape}")
        elif state==None:
            pass
        else:
            raise TypeError(f"Ftate must be a {np.array}. state is a {type(state)}")
        
        if action != None:
            if action not in [i for i in range(self.action_size)]:
                raise ValueError(f"Action can only be from {[i for i in range(self.columns)]}")

    def get_next_state(self, state, action, player):
        self.check_params(state=state, player=player, action=action)
        
        if self.get_valid_moves(state, player)[action] == 1:
            if action < self.columns:
                row_idx = sum(state[:, action] == 0) - 1
                state[row_idx, action] = player
                return state
            else:
                if state[-1][action - self.columns] == player:
                    self.popout = copy.deepcopy(state[-1])
                    state[:, action - self.columns] = np.append(np.array(0), state[:, action - self.columns][:-1])
                    return state
                else:
                    return state
        else:
            return state

    
    def get_valid_moves(self, state, player):
        val_moves = np.append((state[0] == 0), (state[-1] == player)).astype(np.int8)
        return val_moves
    
    def check_win(self, state, action, player):
        self.check_params(state=state, player=player, action=action)
        # player = self.get_last_player(state, action)

        for column in range(self.columns):
            for row in range(self.rows - 3):
                if state[row,column] == state[row+1,column] == state[row+2,column] == state[row+3,column] == player:
                    return True

        # horizontal win
        for row in range(self.rows):
            for column in range(self.columns - 3):
                if state[row,column] == state[row,column+1] == state[row,column+2] == state[row,column+3] == player:
                    return True

        # diagonal top left to bottom right
        for row in range(self.rows - 3):
            for column in range(self.columns - 3):
                if state[row,column] == state[row+1,column+1] == state[row+2,column+2] == state[row+3,column+3] == player:
                    return True

        # diagonal bottom left to top right
        for row in range(self.rows - 1,2,-1):
            for column in range(self.columns - 3):
                if state[row,column] == state[row-1,column+1] == state[row-2,column+2] == state[row-3,column+3] == player:
                    return True
                    
        return False
    
    def check_win_and_type(self, state, action, player):
        self.check_params(state=state, player=player, action=action)
        # player = self.get_last_player(state, action)

        for column in range(self.columns):
            for row in range(self.rows - 3):
                if state[row,column] == state[row+1,column] == state[row+2,column] == state[row+3,column] == player:
                    return True, 'vertical'

        # horizontal win
        for row in range(self.rows):
            for column in range(self.columns - 3):
                if state[row,column] == state[row,column+1] == state[row,column+2] == state[row,column+3] == player:
                    return True, 'horizontal'

        # diagonal top left to bottom right
        for row in range(self.rows - 3):
            for column in range(self.columns - 3):
                if state[row,column] == state[row+1,column+1] == state[row+2,column+2] == state[row+3,column+3] == player:
                    return True, 'diagonal'

        # diagonal bottom left to top right
        for row in range(self.rows - 1,2,-1):
            for column in range(self.columns - 3):
                if state[row,column] == state[row-1,column+1] == state[row-2,column+2] == state[row-3,column+3] == player:
                    return True, 'diagonal'
                    
        return False
        
    def get_last_player(self, state, action):
        if action == None:
            return False
        
        self.check_params(state=state, action=action)
        
        if action < self.columns:
            row_idx = sum((state[:, action] == 0).astype(np.int8))
            
            if row_idx>=6:
                return False
            
            col_idx = action
            return state[row_idx, col_idx]
        else:
            col_idx = action - self.columns
            return self.popout[col_idx]

    def get_value_and_terminated(self, state, action, player):
        if self.check_win(state=state, action=action, player=player):
            return 1, True
        if np.sum(self.get_valid_moves(state, self.get_last_player(state, action))) == 0:
            return 0, True
        return 0, False
    
    def get_opponent(self, player):
        self.check_params(player=player)
        return -player

    def get_opponent_value(self, value):
        return -value

    def change_perspective(self, state, player):
        self.check_params(state=state, player=player)
        return state * player
    
    def get_encoded_state(self, state):
        self.check_params(state=state)
        
        encoded_state = np.stack(
            (state == -1, state == 0, state == 1)
        ).astype(np.float32)
        
        if len(state.shape) == 3:
            encoded_state = np.swapaxes(encoded_state, 0, 1)
        
        return encoded_state

    def __draw_board(self, screen, state):
        COLUMN_COUNT = self.columns
        ROW_COUNT = self.rows
        SQUARESIZE = 100
        RADIUS = int(SQUARESIZE/2 - 5)

        RED = (255,0,0)
        BLLUE = (173,216,230)
        WHITE = (255,255,255)
        BLACK = (0,0,0)

        width = COLUMN_COUNT * SQUARESIZE
        height = (ROW_COUNT+1) * SQUARESIZE

        state = np.flip(state,0)
        for c in range(COLUMN_COUNT):
            for r in range(ROW_COUNT):
                pygame.draw.rect(screen, BLLUE, (c*SQUARESIZE, r*SQUARESIZE+SQUARESIZE, SQUARESIZE, SQUARESIZE))
                pygame.draw.circle(screen, WHITE, (int(c*SQUARESIZE+SQUARESIZE/2), int(r*SQUARESIZE+SQUARESIZE+SQUARESIZE/2)), RADIUS)
        
        for c in range(COLUMN_COUNT):
            for r in range(ROW_COUNT):
                if state[r][c] == 1:
                    pygame.draw.circle(screen, RED, (int(c*SQUARESIZE+SQUARESIZE/2), height-int(r*SQUARESIZE+SQUARESIZE/2)), RADIUS)
                elif state[r][c] == -1:
                    pygame.draw.circle(screen, BLACK, (int(c*SQUARESIZE+SQUARESIZE/2), height-int(r*SQUARESIZE+SQUARESIZE/2)), RADIUS)
        pygame.display.update()

    def render(self, state):
        pygame.init()
        screen = pygame.display.set_mode((700,700))

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    sys.exit()
                self.__draw_board(screen, state)
                pygame.display.update()
    
    def render_state(self, state, player):
        valid_moves = self.get_valid_moves(state, player)
        valid_moves_in = valid_moves[:self.columns]
        valid_moves_out = valid_moves[self.columns:]
        
        cell_size = 75

        image_width = self.columns * cell_size
        image_height = self.rows * cell_size

        image = Image.new('RGB', (image_width, image_height), color='grey')
        draw = ImageDraw.Draw(image)

        for row in range(self.rows):
            for col in range(self.columns):
                x0 = col * cell_size + cell_size // 2
                y0 = row * cell_size + cell_size // 2
                radius = cell_size // 2 - 7
                
                outline = "black"
                if player == 1:
                    if row == 0:
                        if valid_moves_in[col] == 1:
                            outline = "green"
                    elif row == (self.rows - 1):
                        if valid_moves_out[col] == 1:
                            outline = "green"

                if state[row, col] == 0:
                    draw.ellipse([(x0 - radius, y0 - radius), (x0 + radius, y0 + radius)], fill='white', outline=outline, width=5)
                elif state[row, col] == 1:
                    draw.ellipse([(x0 - radius, y0 - radius), (x0 + radius, y0 + radius)], fill='red', outline=outline, width=5)
                elif state[row, col] == -1:
                    draw.ellipse([(x0 - radius, y0 - radius), (x0 + radius, y0 + radius)], fill='black', outline=outline, width=5)

        return image


if __name__=="__main__":
    gm = Game_C4()
    print(gm.action_size)
    board = gm.initialize()
    # print(gm.get_valid_moves(board))
    # board = gm.get_next_state(board, 1, 0)
    # board = gm.get_next_state(board, 1, 0)
    # board = gm.get_next_state(board, -1, 0)
    # board = gm.get_next_state(board, 1, 0)
    # board = gm.get_next_state(board, 1, 0)
    # board = gm.get_next_state(board, -1, 0)
    # board = gm.get_next_state(board, 1, 0)
    # print(board)