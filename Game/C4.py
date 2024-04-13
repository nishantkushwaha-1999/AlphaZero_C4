import sys
import pygame
import numpy as np

class Game_C4:
    def __init__(self):
        self.columns = 7
        self.rows = 6
        self.action_size = self.columns * 1 # change action dim to double this
        self.n_similar = 4
    
    def __repr__(self):
        return "Connect_4"
    
    def initialize(self):
        return np.zeros((self.rows, self.columns))

    def check_params(self, state: np.array=None, player=None, action=None):
        if player != None:
            if player not in [1, -1]:
                raise ValueError(f"player: {player} not found. Player must be either -1 or 1")
        
        if str(type(state))=="<class 'numpy.ndarray'>":
                if (state.shape[-1] != self.rows) and (state.shape[-2] != self.rows):
                    raise ValueError(f"Invalid board passed of shape {state.shape}")
        elif state==None:
            pass
        else:
            raise TypeError(f"Ftate must be a {np.array}. state is a {type(state)}")
        
        if action != None:
            if action not in [i for i in range(self.columns)]:
                raise ValueError(f"Action can only be from {[i for i in range(self.columns)]}")

    def get_next_state(self, state, action, player):
        self.check_params(state=state, player=player, action=action)
        
        # Modify for popout rule
        if self.get_valid_moves(state)[action] == 1:
            row_idx = sum(state[:, action] == 0) - 1
            state[row_idx, action] = player
            return state
        else:
            return state
    
    def get_valid_moves(self, state):
        # Modify for popout rule
        return (state[0] == 0).astype(np.int8)
    
    def check_win(self, state, action):
        if action == None:
            return False
        
        self.check_params(state=state, action=action)
        
        row_idx = sum((state[:, action] == 0).astype(np.int8))
        col_idx = action
        player = state[row_idx, col_idx]

        def count(row_offset, col_offset):
            for i in range(1, self.n_similar):
                r_idx = row_idx + row_offset * i
                c_idx = col_idx + col_offset * i
                if (r_idx<0 or r_idx>=self.rows or c_idx<0, c_idx>=self.columns or state[row_idx, col_idx]!= player):
                    return i - 1
        if(
            count(1, 0) >= self.n_similar - 1
            or (count(0, 1) + count(0, -1)) >= self.n_similar -1
            or (count(1, 1) + count(-1, -1)) >= self.n_similar - 1
            or (count(1, -1) + count(-1, 1)) >= self.n_similar - 1
        ):
            return True
        else:
            return False
        
    def get_value_and_terminated(self, state, action):
        if self.check_win(state=state, action=action):
            return 1, True
        if np.sum(self.get_valid_moves(state)) == 0:
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

    def render(self, state=None):
        if state==None:
            state = self.state.copy()
        
        pygame.init()
        screen = pygame.display.set_mode((700,700))

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    sys.exit()
                self.__draw_board(screen, state)
                pygame.display.update()


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