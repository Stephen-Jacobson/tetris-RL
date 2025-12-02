from rotations import PIECES, WALL_KICKS, WALL_KICKS_I
from TetriminoObj import TetriminoObj
from typing import Optional
import numpy as np
import pygame
import gymnasium as gym
from gymnasium.envs.registration import register

CELL_SIZE = 30      # size of each Tetris block in pixels
ROWS, COLS = 20, 10 # visible board size
HIDDEN_ROWS = 4     # extra rows at top for spawning

WINDOW_WIDTH = COLS * CELL_SIZE
WINDOW_HEIGHT = ROWS * CELL_SIZE + 50

STATS_PANEL_WIDTH = 150  
# At the top of the file, change:
WINDOW_WIDTH = STATS_PANEL_WIDTH + COLS * CELL_SIZE + STATS_PANEL_WIDTH  # Add right panel

# Colors for pieces: I,O,T,J,L,S,Z
COLORS = {
    0: (0,0,0),       # empty
    1: (0,255,255),
    2: (255,255,0),
    3: (128,0,128),
    4: (0,255,0),
    5: (255,0,0),
    6: (0,0,255),
    7: (255,165,0)
}


class TetrisEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self):
        super().__init__()

        # Gym Spaces
        self.action_space = gym.spaces.Discrete(8)

        self.observation_space = gym.spaces.Dict(
            {
                "board": gym.spaces.Box(
                    low = 0,
                    high = 7,
                    shape = (24, 10),
                    dtype = np.int32
                ),
                "cur_piece_type": gym.spaces.Discrete(8),
                "cur_piece_rot": gym.spaces.Discrete(4),
                "cur_piece_pos": gym.spaces.Box(
                    low = np.array([0, 0], dtype = np.int32),
                    high = np.array([23, 9], dtype = np.int32),
                    dtype = np.int32
                ),
                "hold_piece": gym.spaces.Discrete(8),
                "next_piece1": gym.spaces.Discrete(8),
                "next_piece2": gym.spaces.Discrete(8),
                "next_piece3": gym.spaces.Discrete(8)
            }
        )

        # Game Logic
        self.board = self.create_board()
        self.reward = 0
        self.gravity = 1/60
        self.gravity_counter = 0
        self.auto_lock = 500
        self.lock = False
        self.lock_timer = 0
        self.lock_moves = 0
        self.rot_index = 0
        self.level = 1
        self.points = 0
        self.clears = 0
        self.shuffle()   #"i", "o", "t", "s", "z", "j", "l"
        self.cur_piece = self.spawn_piece()
        self.held_piece = None
        self.swapped = False
        self.next_pieces = [] 
        self.combo = [0, False]             #stores amount in combo and whether can be b2b or not
        self.last_rotated = False
        self.update_next_pieces()
        self.rl = False

        self.screen = None
        self.clock = None
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        
        self.board = self.create_board()
        self.reward = 0
        self.gravity = 1/60
        self.gravity_counter = 0
        self.lock = False
        self.lock_timer = 0
        self.lock_moves = 0
        self.rot_index = 0
        self.level = 1
        self.points = 0
        self.clears = 0
        self.shuffle()   #"i", "o", "t", "s", "z", "j", "l"
        self.cur_piece = self.spawn_piece()
        self.held_piece = None
        self.swapped = False
        self.next_pieces = []
        self.combo = [0, False]
        self.last_rotated = False
        self.update_next_pieces()

        return self.get_state()
    
    def _get_obs(self):
        # board must be exactly int32
        board_obs = self.board.astype(np.int32).copy()

        # cur_piece_type should be int (1..7)
        cur_type = int(self.cur_piece.type) if self.cur_piece is not None else 0

        # hold_piece: return 0 if None, else the piece type as int
        if self.held_piece is None:
            hold_val = 0
        else:
            # if held_piece is stored as an object
            try:
                hold_val = int(self.held_piece.type)
            except Exception:
                # fallback: if you ever store the type directly
                hold_val = int(self.held_piece)

        next1 = int(self.next_pieces[0]) if len(self.next_pieces) > 0 else 0
        next2 = int(self.next_pieces[1]) if len(self.next_pieces) > 1 else 0
        next3 = int(self.next_pieces[2]) if len(self.next_pieces) > 2 else 0

        return {
            "board": board_obs,                             # np.int32 array
            "cur_piece_type": cur_type,                     # int 0..7
            "cur_piece_rot": int(self.rot_index),           # int
            "cur_piece_pos": np.array(self.cur_piece.pos, dtype=np.int32),
            "hold_piece": hold_val,                         # int 0..7
            "next_piece1": next1,                           # int 0..7
            "next_piece2": next2,
            "next_piece3": next3
        }

    def _get_info(self):
        """Return auxiliary info for debugging/logging."""
        return {
            "level": self.level,                     # current level
            "points": self.points,                   # current score
            "lines_cleared": self.clears,           # total lines cleared
            "combo_count": self.combo[0],           # combo streak
            "b2b_ready": self.combo[1],             # whether back-to-back is active
            "current_piece": self.cur_piece.type,   # type of current piece
            "held_piece": self.held_piece,          # type of held piece
            "next_pieces": self.next_pieces.copy(), # next 3 pieces
            "lock_moves": self.lock_moves           # moves made during lock delay
        }

    def get_state(self):
        observation = self._get_obs()
        info = self._get_info()
        return observation, info
    
    def step(self, action, dt: float = 1000.0/60.0, apply_gravity: bool = True):
        """
        action: 0=left, 1=right, 2=rotatecw, 3=rotateacw, 4=soft down, 5=hard down, 6=hold piece, 7=do nothing
        returns: state, reward, done, info
        """
        old_lines = self.clears
        piece_steps = self.cur_piece.steps

        if self.screen is not None:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    self.screen = None

        old_points = self.points
        done = False
        

        if action == 7:         #do nothing :)
            None
        self.cur_piece.steps += 1

        if action == 6:
            self.hold_piece()
            self.last_rotated = False
        if action == 5:
            done = self.hard_down()
        if action == 4:
            if self.get_lowest_row()[0] < 23 and self.get_lowest_row()[1] == False:
                self.cur_piece.pos = (self.cur_piece.pos[0] + 1, self.cur_piece.pos[1])
                self.cur_piece.pieces = [(r + 1, c) for r, c in self.cur_piece.pieces]
                self.add_points(0, 1)
                self.last_rotated = False
        if action == 0:
            if self.left_and_right()[0] > 0 and not(self.get_side_obstruct(-1)):
                self.cur_piece.pos = (self.cur_piece.pos[0], self.cur_piece.pos[1] - 1)
                self.cur_piece.pieces = [(r, c - 1) for r, c in self.cur_piece.pieces]
                self.last_rotated = False

                if self.lock:
                    self.lock_moves += 1
                    self.lock_timer = 0
        if action == 1:
            if self.left_and_right()[1] < 9 and not(self.get_side_obstruct(1)):
                self.cur_piece.pos = (self.cur_piece.pos[0], self.cur_piece.pos[1] + 1)
                self.cur_piece.pieces = [(r, c + 1) for r, c in self.cur_piece.pieces]
                self.last_rotated = False
                
                if self.lock:
                    self.lock_moves += 1
                    self.lock_timer = 0
        if action == 2:
            self.rotate_cw()
            
        if action == 3:
            self.rotate_acw()

        frames_elapsed = dt / (1000.0 / 60.0)
        
        if apply_gravity:
            self.get_gravity()
            self.gravity_counter += self.gravity * frames_elapsed

            # Handle gravity-based falling
            while self.gravity_counter >= 1:
                self.gravity_counter -= 1
                if self.get_lowest_row()[0] < 23 and self.get_lowest_row()[1] == False:
                    self.cur_piece.pos = (self.cur_piece.pos[0] + 1, self.cur_piece.pos[1])
                    self.cur_piece.pieces = [(r + 1, c) for r, c in self.cur_piece.pieces]

                    self.last_rotated = False

                    self.lock = False
                    self.lock_timer = 0
                    self.lock_moves = 0
                else:
                    self.lock = True
                    self.gravity_counter = 0
                    break
            
            if self.lock:
                self.lock_timer += dt * 1

            if self.lock_timer >= self.auto_lock or self.lock_moves >= 15:
                offsets = PIECES[self.cur_piece.type][self.rot_index]
                for row, col in self.cur_piece.pieces:
                    self.board[row][col] = self.cur_piece.type
                self.swapped = False
                
                temp = self.new_piece()
                if not done:
                    done = temp

            observation = self._get_obs()
            info = self._get_info()

            # Calculate reward as the change in points
            cleared = self.clears - old_lines

            point_delta = self.points - old_points

            bumpiness = self.get_bumpiness()
            holes = self.get_holes()
            agg_height = self.get_aggregate_height()

            # ---- hyperparameters (tune these!) ----
            w_points = 2.0            # keep original point signal
            w_lines = 10.0             # extra positive reward per cleared line (on top of points)
            w_holes = 0.8             # penalty per hole (higher -> avoid holes)
            w_bumpiness = 2       # penalty per bumpiness unit (lower -> flatter)
            w_height = 0.01          # penalty per aggregate height unit (lower -> lower stack)
            per_action_penalty = 0.02 # small penalty for each action on the same piece
            action_threshold = 10     # allow a few actions without big penalty

            extra_action_penalty = 0.05
            extra_actions = max(0, piece_steps - action_threshold)

            # ---- Calculate shaped reward ----
            reward = 0.0
            reward += w_points * point_delta
            
            reward += w_lines * self.clears
            
            reward -= w_holes * holes
            
            reward -= w_bumpiness * bumpiness
            
            reward -= w_height * agg_height
            
            # per-action small penalty (encourages fewer, more decisive moves)
            if 3<=piece_steps<= 10:
                reward += per_action_penalty * 10 * piece_steps
            else:
                if piece_steps > 15:
                    reward -= per_action_penalty * 5 * piece_steps
                else:
                    reward -= per_action_penalty * piece_steps

            
            # extra penalty if the piece used way too many moves
            reward -= extra_action_penalty * extra_actions
            if self.screen is not None:
                print(f"score: {w_points * point_delta}")
                print(f"clears: {w_lines * self.clears}")
                print(f"holes: -{w_holes * holes}")
                print(f"bumps: -{w_bumpiness * bumpiness}")
                print(f"agg height: -{w_height * agg_height}")
                print(f"action penalty: -{per_action_penalty * piece_steps}")
                print(f"extra action: -{extra_action_penalty * extra_actions}")
            #     print(self.reward)
            # self.reward += reward

            # # Get current board metrics
            # holes = self.get_holes()
            # bumpiness = self.get_bumpiness()
            # agg_height = self.get_aggregate_height()
            
            # # ---- Reward hyperparameters ----
            # w_line_clear_base = 100.0     # Base reward per line
            # w_holes = -4.0                # Penalty per hole
            # w_bumpiness = -0.5            # Penalty per bumpiness unit
            # w_height = -0.5               # Penalty for stack height
            # w_game_over = -100.0          # Large penalty for dying
            # w_survival = 0.1              # Small reward for surviving
            # w_max_steps = 30
            
            # # ---- Calculate reward ----
            # reward = 0.0
            
            # # Line clear rewards (exponential to encourage multi-line clears)
            # if cleared == 1:
            #     reward += w_line_clear_base * 1  # 100
            # elif cleared == 2:
            #     reward += w_line_clear_base * 3  # 300
            # elif cleared == 3:
            #     reward += w_line_clear_base * 5  # 500
            # elif cleared == 4:
            #     reward += w_line_clear_base * 8  # 800 (Tetris!)
            
            # if self.cur_piece.steps > w_max_steps:
            #     reward -= (self.cur_piece.steps - w_max_steps) * 0.6

            # # Board state penalties (only apply when piece locks)
            # if self.lock_timer >= self.auto_lock or self.lock_moves >= 15 or action == 5:
            #     reward += w_holes * holes
            #     reward += w_bumpiness * bumpiness
            #     reward += w_height * (agg_height / 100.0)  # Normalize height
            w_survival = 0.1
            w_game_over = -100.0
            # Survival bonus (encourages staying alive)
            reward += w_survival
            

            if done:
                reward += w_game_over
                self.reset()
            return observation, reward, bool(done), False, info
    
    def run(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        self.clock = pygame.time.Clock()
        
        running = True

        das_initial = 130
        if self.auto_lock < das_initial:
            self.auto_lock = das_initial
        das_repeat = 17
        left_held = False
        left_happened = False
        right_held = False
        right_happened = False
        left_timer = 0
        right_timer = 0
        down_held = False

        while running:
            print(self.tslot_exists())
            dt = self.clock.tick(60)
            
            # Update gravity based on current level
            self.get_gravity()
            
            # Increment gravity counter
            self.gravity_counter += self.gravity
            
            if self.lock_timer >= self.auto_lock or self.lock_moves >= 15:
                offsets = PIECES[self.cur_piece.type][self.rot_index]
                for row, col in self.cur_piece.pieces:
                    self.board[row][col] = self.cur_piece.type
                self.swapped = False
                self.new_piece()
            if self.lock:
                self.lock_timer += dt

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        self.step(5, dt, False)
                    if event.key == pygame.K_LEFT:
                        self.step(0, dt, False)

                        left_held = True
                        left_timer = 0
                        left_happened = False

                        # reset opposite side completely
                        right_held = False
                        right_happened = False
                        right_timer = 0

                    if event.key == pygame.K_RIGHT:
                        self.step(1, dt, False)

                        right_held = True
                        right_timer = 0
                        right_happened = False

                        # reset opposite side completely
                        left_held = False
                        left_happened = False
                        left_timer = 0
                    if event.key == pygame.K_UP:
                        self.step(2, dt, False)
                    if event.key == pygame.K_DOWN:
                        self.step(4, dt, False)
                        down_held = True
                    if event.key == pygame.K_z:
                        self.step(3, dt, False)
                    if event.key == pygame.K_c:
                        self.step(6, dt, False)
                elif event.type == pygame.KEYUP:
                    if event.key == pygame.K_LEFT:
                        left_held = False
                        left_happened = False
                        left_timer = 0
                    if event.key == pygame.K_RIGHT:
                        right_held = False
                        right_happened = False
                        right_timer = 0
                    if event.key == pygame.K_DOWN:
                        down_held = False
            
            if left_held:
                if left_happened:
                    if left_timer > das_repeat:
                        self.step(0, dt, False)
                        left_timer = 0
                elif left_timer > das_initial:
                    self.step(0, dt, False)
                    left_happened = True
                    left_timer = 0
                
                left_timer += dt
            
            if right_held:
                if right_happened:
                    if right_timer > das_repeat:
                        self.step(1, dt, False)
                        right_timer = 0
                elif right_timer > das_initial:
                    self.step(1, dt, False)
                    right_happened = True
                    right_timer = 0
                
                right_timer += dt
            if down_held:
                self.step(4, dt, False)

            # Handle gravity-based falling
            while self.gravity_counter >= 1:
                self.gravity_counter -= 1
                if self.get_lowest_row()[0] < 23 and self.get_lowest_row()[1] == False:
                    self.cur_piece.pos = (self.cur_piece.pos[0] + 1, self.cur_piece.pos[1])
                    self.cur_piece.pieces = [(r + 1, c) for r, c in self.cur_piece.pieces]

                    self.last_rotated = False

                    self.lock = False
                    self.lock_timer = 0
                    self.lock_moves = 0
                else:
                    self.lock = True
                    self.gravity_counter = 0
                    break
                    
            self.render()
        pygame.quit()

    def reset_pos(self, piece):
        if piece.type == 1:
            piece.pos = (3, 3)
        else:
            piece.pos = (4, 3)
        self.rot_index = 0
        offsets = PIECES[piece.type][self.rot_index]
        for i in range(len(piece.pieces)):
            piece.pieces[i] = (piece.pos[0] + offsets[i][0], piece.pos[1] + offsets[i][1])

    def add_points(self, type_add, drop):
        #types 0 soft drop, 1 hard drop, 2 single, 3 double, 4 triple, 5 tetris, 6 mini tspin no lines, 7 tspin no lines, 8 mini tspin single, 9 tspin single, 10 mini tspin double, 11 tspin double, 12 tspin triple, 13 combo, b2b is separate to switch
        switch = {
            0: 1 * drop,
            1: 2 * drop,
            2: 100 * self.level,
            3: 300 * self.level,
            4: 500 * self.level,
            5: 800 * self.level,
            6: 100 * self.level,
            7: 400 * self.level,
            8: 200 * self.level,
            9: 800 * self.level,
            10: 400 * self.level,
            11: 1200 * self.level,
            12: 1600 * self.level,
            13: 50 * self.combo[0] * self.level
        }
        if self.combo[1] and 5 <= type_add < 13:
            self.points += 1.5 * switch[type_add]
        else:
            if 5 <= type_add < 13:
                self.combo[1] = True
            elif 2 <= type_add <= 4:
                self.combo[1] = False
            if type_add != 13:
                self.points += switch[type_add]


        if 2 <= type_add < 13:
            self.points += switch[13]
            self.combo[0] += 1
        

    def hold_piece(self):
        if self.swapped:
            return
        self.swapped = True
        self.cur_piece.steps = 0
        if self.held_piece == None:
            self.held_piece = self.cur_piece
            self.new_piece()
        else:
            temp = self.held_piece
            self.held_piece = self.cur_piece
            self.cur_piece = temp
        self.reset_pos(self.cur_piece)

    def update_next_pieces(self):
        """Update the next 3 pieces preview"""
        # Make sure we have enough pieces in the bag
        while len(self.piece_bag) < 3:
            self.shuffle()
            # Append to existing bag instead of replacing
            temp_bag = np.array([1, 2, 3, 4, 5, 6, 7], dtype=int)
            np.random.shuffle(temp_bag)
            self.piece_bag = np.concatenate([self.piece_bag, temp_bag])
        
        # Get the next 3 pieces (from the end of the bag)
        self.next_pieces = self.piece_bag[-3:][::-1]

    def get_lowest_row(self):
        lowest = 0
        piece_obstruct = False
        for i in self.cur_piece.pieces:
            if i[0] > lowest:
                lowest = i[0]
            if i[0] + 1 == 24:
                continue 
            if self.board[i[0] + 1][i[1]] != 0 and not((i[0] + 1, i[1]) in self.cur_piece.pieces):
                piece_obstruct = True
        return lowest, piece_obstruct
    
    
    
    def get_side_obstruct(self, dir):
        #dir = 1 right, -1 left
        for i in self.cur_piece.pieces:
            if self.board[i[0]][i[1] + dir] != 0:
                return True

    def left_and_right(self):
        left = 10
        right = 0
        for i in self.cur_piece.pieces:
            if i[1] < left:
                left = i[1]
            if i[1] > right:
                right = i[1]
        return left, right

    def check_tmini(self):      # returns -1 = no tspin, 0 = mini, 1 = full
        rows = len(self.board)
        cols = len(self.board[0])

        r = self.cur_piece.pos[0]
        c = self.cur_piece.pos[1]

        # diag[0]=top-left, diag[1]=top-right, diag[2]=bot-left, diag[3]=bot-right
        diag = [0, 0, 0, 0]

        # helper to check diagonal and count OOB as filled
        def filled(rr, cc):
            if rr < 0 or rr >= rows or cc < 0 or cc >= cols:
                return True   # out-of-bounds counts as filled
            return self.board[rr][cc] != 0

        # check all 4 corners of the 3×3 T-box
        diag[0] = 1 if filled(r,     c    ) else 0
        diag[1] = 1 if filled(r,     c + 2) else 0
        diag[2] = 1 if filled(r + 2, c    ) else 0
        diag[3] = 1 if filled(r + 2, c + 2) else 0

        # corner count
        corner_count = diag.count(1)

        # must have >= 3 corners AND must have rotated last
        if corner_count < 3 or not self.last_rotated:
            return -1

        # Check "facing direction" T-Spin (full)
        # Your original checks kept exactly as written:
        if self.rot_index == 0:     # facing up
            if diag[0] and diag[1]:
                return 1
        elif self.rot_index == 1:   # facing right
            if diag[1] and diag[3]:
                return 1
        elif self.rot_index == 2:   # facing down
            if diag[2] and diag[3]:
                return 1
        elif self.rot_index == 3:   # facing left
            if diag[0] and diag[2]:
                return 1

        # if T-spin but not full, it's mini
        return 0


    def check_clear(self):
        cleared = 0         #local amount cleared
        type_clear = -1     #type clear to depend points, -1 normal clear, 0 tspin mini, 1 tspin

        if self.cur_piece.type == 3:
            type_clear = self.check_tmini()
        
        for r in range(len(self.board)):
            if all(cell != 0 for cell in self.board[r]):
                self.board = np.delete(self.board, r, axis=0)
                self.board = np.vstack((np.array([[0]*len(self.board[0])]), self.board))
                self.clears += 1
                cleared += 1
                if self.clears % 10 == 0:
                    self.level += 1
                    if self.level > 20:
                        self.auto_lock -= 60
        if cleared == 0:
            if type_clear == 0:
                self.add_points(6, 0)
            elif type_clear == 1:
                self.add_points(7, 0)
            else:
                self.combo = [0, False]
        if cleared == 1:
            if type_clear == -1:
                self.add_points(2, 0)
            if type_clear == 0:
                self.add_points(8, 0)
            if type_clear == 1:
                self.add_points(9, 0)
        if cleared == 2:
            if type_clear == -1:
                self.add_points(3, 0)
            if type_clear == 0:
                self.add_points(10, 0)
            if type_clear == 1:
                self.add_points(11, 0)
        if cleared == 3:
            if type_clear == -1:
                self.add_points(4, 0)
            if type_clear == 1:
                self.add_points(12, 0)
        if cleared == 4:
            self.add_points(5, 0)
        
    def get_ghost_position(self):
        """Calculate where the ghost piece (hard drop preview) should be"""
        if self.cur_piece is None:
            return []
        
        # Find the maximum distance we can drop the piece
        max_drop = len(self.board)
        
        offsets = PIECES[self.cur_piece.type][self.rot_index]
        
        # Check each block of the current piece
        for offset in offsets:
            block_row = self.cur_piece.pos[0] + offset[0]
            block_col = self.cur_piece.pos[1] + offset[1]
            
            # Find how far this block can drop
            drop_distance = -1
            for check_row in range(block_row, len(self.board)):
                if self.board[check_row][block_col] != 0:
                    break
                drop_distance += 1
            
            
            # Use the minimum drop distance across all blocks
            max_drop = min(max_drop, drop_distance)
        
        # Calculate ghost piece positions
        ghost_pieces = []
        for offset in offsets:
            row = self.cur_piece.pos[0] + offset[0] + max_drop
            col = self.cur_piece.pos[1] + offset[1]
            ghost_pieces.append((row, col))
        
        return ghost_pieces
    
    def _positions_for(self, anchor, rot, piece_type=3):
        """Return absolute block cells for a piece_type at (anchor_r,anchor_c) with rotation rot."""
        r, c = anchor
        offsets = PIECES[piece_type][rot]
        return [(r + off[0], c + off[1]) for off in offsets]

    def can_rotate_into_slot(self, anchor, target_rot, piece_type=3):
        """
        Return True if there exists a legal starting placement (row, col, rotation)
        from which a single rotation (cw or acw with wall kicks) will produce the
        exact block cells of the slot defined by (anchor, target_rot).

        NOTE: This looks for a single-rotation move (the usual 'final rotation' for a T-Spin).
        It does not attempt full pathfinding from spawn; it only verifies that a rotation
        + wall-kick can place the piece into the candidate cells from some feasible starting cell.
        """
        rows = len(self.board)
        cols = len(self.board[0])

        # compute the target absolute positions we want to match after rotation+kick
        target_positions = set(self._positions_for(anchor, target_rot, piece_type))

        # helper: is a cell empty (in bounds and zero)
        def is_empty_cell(rr, cc):
            if rr < 0 or rr >= rows or cc < 0 or cc >= cols:
                return False
            return self.board[rr][cc] == 0

        # Search plausible starting rows and columns.
        # We only need to search rows above or at the anchor (pieces drop from above),
        # but allowing a few rows above is safer. You can tighten range for speed.
        min_start_row = 0
        max_start_row = anchor[0]  # inclusive

        for start_r in range(min_start_row, max_start_row + 1):
            for start_c in range(0, cols):
                # try every possible starting rotation
                for rot_start in range(4):
                    # compute absolute cells of the piece at start (without collisions)
                    start_positions = self._positions_for((start_r, start_c), rot_start, piece_type)

                    # ensure start positions are fully in bounds and empty (the piece could be "there" before rotation)
                    ok_start = True
                    for (sr, sc) in start_positions:
                        if sr < 0 or sr >= rows or sc < 0 or sc >= cols or self.board[sr][sc] != 0:
                            ok_start = False
                            break
                    if not ok_start:
                        continue

                    # try rotating one step cw and one step acw (simulate wall-kicks like your wall_kicks())
                    for dir in (0, 1):  # 0 = cw, 1 = acw
                        if dir == 0:
                            temp_in = (rot_start + 1) % 4
                            num_kick = 2 * rot_start + 1
                        else:
                            temp_in = (rot_start - 1) % 4
                            num_kick = 8 - 2 * rot_start

                        offsets_target = PIECES[piece_type][temp_in]

                        # FIXED kick lookup
                        if piece_type == 1:
                            kicks = WALL_KICKS_I[num_kick]
                        else:
                            kicks = WALL_KICKS[num_kick]
                        # (defensive: if WALL_KICKS is shaped differently, the above still indexes)

                        offsets_target = PIECES[piece_type][temp_in]

                        for kick in kicks:
                            # compute positions after applying rotation offsets + this kick
                            result_positions = []
                            collision = False
                            for i in range(len(offsets_target)):
                                rr = start_r + offsets_target[i][0] + kick[0]
                                cc = start_c + offsets_target[i][1] + kick[1]
                                # out of bounds or overlapping existing blocks => collision
                                if rr < 0 or rr >= rows or cc < 0 or cc >= cols:
                                    collision = True
                                    break
                                if self.board[rr][cc] != 0:
                                    collision = True
                                    break
                                result_positions.append((rr, cc))
                            if collision:
                                continue

                            # if result matches target positions exactly -> reachable
                            if set(result_positions) == target_positions:
                                return True
        # no found start pos that rotates into the target slot
        return False

    def detect_t_slots(self):
        """
        Scan the board for potential T-Spin slots (3x3 T-boxes).
        Returns: list of candidates. Each candidate is a dict:
        {
            "anchor": (r, c),            # top-left of the 3x3 box (same convention as check_tmini)
            "corner_count": int,         # how many of the 4 diagonals are filled (walls count)
            "rotation": rot,             # rotation index (0..3) that fits into the slot
            "positions": [(r, c), ...],  # absolute block cells the T would occupy for that rotation
            "is_full": bool,             # heuristic full vs mini (same check as check_tmini)
        }
        """
        rows = len(self.board)
        cols = len(self.board[0])
        piece_type = 3  # T piece type in your code

        def filled(rr, cc):
            # out-of-bounds counts as filled (wall)
            if rr < 0 or rr >= rows or cc < 0 or cc >= cols:
                return True
            return self.board[rr][cc] != 0

        candidates = []

        # iterate every possible 3x3 top-left anchor
        for r in range(0, rows - 2):
            for c in range(0, cols - 2):
                # corners: top-left, top-right, bot-left, bot-right
                corners = [
                    (r,     c),
                    (r,     c + 2),
                    (r + 2, c),
                    (r + 2, c + 2)
                ]
                corner_status = [1 if filled(rr, cc) else 0 for (rr, cc) in corners]
                corner_count = sum(corner_status)
                if corner_count < 3:
                    continue  # not a T-slot candidate

                # Now check if a T can actually fit into the 3x3 for any rotation
                for rot in range(4):
                    offsets = PIECES[piece_type][rot]  # offsets are relative to the anchor (r,c) in your code
                    abs_positions = [(r + off[0], c + off[1]) for off in offsets]

                    # ensure all T block cells are within bounds and empty
                    can_fit = True
                    for (pr, pc) in abs_positions:
                        if pr < 0 or pr >= rows or pc < 0 or pc >= cols:
                            can_fit = False
                            break
                        if self.board[pr][pc] != 0:
                            can_fit = False
                            break

                    if not can_fit:
                        continue

                    # Determine "full" vs "mini" using same facing-direction check as check_tmini()
                    full = False
                    # reuse corner_status list order: [tl, tr, bl, br]
                    if rot == 0:     # facing up
                        if corner_status[0] and corner_status[1]:
                            full = True
                    elif rot == 1:   # facing right
                        if corner_status[1] and corner_status[3]:
                            full = True
                    elif rot == 2:   # facing down
                        if corner_status[2] and corner_status[3]:
                            full = True
                    elif rot == 3:   # facing left
                        if corner_status[0] and corner_status[2]:
                            full = True
                    if not self.can_rotate_into_slot((r, c), rot): continue
                    if self.can_drop_straight_into_slot((r,c), rot): continue
                    candidates.append({
                        "anchor": (r, c),
                        "corner_count": corner_count,
                        "rotation": rot,
                        "positions": abs_positions,
                        "is_full": full
                    })

        return candidates

    def can_drop_straight_into_slot(self, anchor, target_rot, piece_type=3):
        """
        Returns True if the target slot can be reached without rotating
        (just moving horizontally and dropping straight down).
        That makes it NOT a real T-spin slot.
        """
        rows = len(self.board)
        cols = len(self.board[0])

        target_positions = set(self._positions_for(anchor, target_rot, piece_type))

        # Try all plausible horizontal positions
        for start_c in range(0, cols):
            start_r = 0  # start high

            # Simulate straight drop: move down until collision
            while True:
                positions = self._positions_for((start_r, start_c), target_rot, piece_type)

                # check if we match target exactly (success, no rotate needed)
                if set(positions) == target_positions:
                    return True

                # try dropping further
                next_positions = [(r+1, c) for (r,c) in positions]

                # if collision or outside board -> stop this column
                for nr, nc in next_positions:
                    if nr >= rows or nc < 0 or nc >= cols:
                        break
                    if self.board[nr][nc] != 0:
                        break

                else:
                    # still empty -> apply drop and continue
                    start_r += 1
                    continue

                break

        return False


    def tslot_exists(self):
        """Boolean quick-check: True if any T-slot candidate exists on board."""
        return len(self.detect_t_slots()) > 0


    
    def hard_down(self):
        # Find the maximum distance we can drop the piece
        max_drop = len(self.board)
        
        offsets = PIECES[self.cur_piece.type][self.rot_index]
        # Check each block of the current piece
        for offset in offsets:
            block_row = self.cur_piece.pos[0] + offset[0]
            block_col = self.cur_piece.pos[1] + offset[1]
            
            # Find how far this block can drop
            drop_distance = 0
            for check_row in range(block_row + 1, len(self.board)):
                if self.board[check_row][block_col] != 0:
                    break
                drop_distance += 1
            else:
                # Reached the bottom
                drop_distance = len(self.board) - block_row - 1
            
            # Use the minimum drop distance across all blocks
            max_drop = min(max_drop, drop_distance)
        self.add_points(1, max_drop)
        # Move the piece's anchor position down by max_drop
        self.cur_piece.pos = (self.cur_piece.pos[0] + max_drop, self.cur_piece.pos[1])
        
        # Place all blocks on the board using the new position
        for offset in offsets:
            row = self.cur_piece.pos[0] + offset[0]
            col = self.cur_piece.pos[1] + offset[1]
            self.board[row][col] = self.cur_piece.type
        self.swapped = False
        if self.new_piece():
            return True
        else:
            return False
            

    def check_end(self):
        dont_end = False
        for i in self.cur_piece.pieces:
            if (i[0] > 3):
                dont_end = True
        if not(dont_end):
            self.reset()
            return True
        return False

    def new_piece(self):
        game_over = self.check_end()
        if game_over:
            return True
        self.check_clear()
        temp = self.spawn_piece()
        if temp == None:
            return True
        self.cur_piece = temp
        self.lock = False
        self.lock_timer = 0
        self.lock_moves = 0
        self.rot_index = 0
        return False
    
    def get_bumpiness(self):
        """Calculate the bumpiness of the board (sum of absolute height differences between adjacent columns)"""
        heights = self.get_column_heights()
        bumpiness = 0
        for i in range(len(heights) - 1):
            bumpiness += abs(heights[i] - heights[i + 1])
        return bumpiness

    def get_column_heights(self):
        """Get the height of each column (distance from bottom to highest filled cell)"""
        heights = []
        for col in range(COLS):
            height = 0
            for row in range(len(self.board) - 1, -1, -1):  # Start from bottom
                if self.board[row][col] != 0:
                    height = len(self.board) - row
            heights.append(height)
        return heights
    
    def get_aggregate_height(self):
        """Calculate the sum of all column heights"""
        heights = self.get_column_heights()
        return sum(heights)

    def get_holes(self):
        """Calculate the number of holes (empty cells with a filled cell directly above them)"""
        holes = 0
        for col in range(COLS):
            block_found = False
            for row in range(len(self.board)):
                cell = self.board[row][col]
                if cell != 0:
                    block_found = True
                elif block_found and cell == 0:
                    holes += 1
        return holes

    def wall_kicks(self, dir):           #use when rotating, direction: 0 clockwise, 1 anticlockwise
        num_kick = 0
        if dir == 0:
            temp_in = (self.rot_index + 1) % 4
            num_kick = 2*self.rot_index + 1         #corresponds correctly to wall kicks tables
        elif dir == 1:
            temp_in = (self.rot_index - 1) % 4
            num_kick = 8 - 2*self.rot_index
        
        offsets = PIECES[self.cur_piece.type][temp_in]

        for j in range(len(WALL_KICKS[1])):
            fit = True
            for i in range(len(self.cur_piece.pieces)):
                if self.cur_piece.type == 1:
                    row = self.cur_piece.pos[0] + offsets[i][0] + WALL_KICKS_I[num_kick][j][0]
                    col = self.cur_piece.pos[1] + offsets[i][1] + WALL_KICKS_I[num_kick][j][1]
                else:
                    row = self.cur_piece.pos[0] + offsets[i][0] + WALL_KICKS[num_kick][j][0]
                    col = self.cur_piece.pos[1] + offsets[i][1] + WALL_KICKS[num_kick][j][1]
    
                if not (0 <= row < 24 and 0 <= col < 10):
                    fit = False
                    break
                if self.board[row][col] != 0:
                    fit = False
                    break
                
            if fit == True:
                self.last_rotated = True
                if self.lock:
                    self.lock_moves += 1
                    self.lock_timer = 0

                if dir == 0:
                    self.rot_index = (self.rot_index + 1) % 4
                elif dir == 1:
                    self.rot_index = (self.rot_index - 1) % 4
                
                for k in range(len(self.cur_piece.pieces)):
                    
                    if self.cur_piece.type == 1:
                        self.cur_piece.pieces[k] = (self.cur_piece.pos[0] + offsets[k][0] + WALL_KICKS_I[num_kick][j][0], self.cur_piece.pos[1] + offsets[k][1] + WALL_KICKS_I[num_kick][j][1])
                        
                    else:
                        self.cur_piece.pieces[k] = (self.cur_piece.pos[0] + offsets[k][0] + WALL_KICKS[num_kick][j][0], self.cur_piece.pos[1] + offsets[k][1] + WALL_KICKS[num_kick][j][1])
                if self.cur_piece.type == 1:
                    self.cur_piece.pos = (self.cur_piece.pos[0] + WALL_KICKS_I[num_kick][j][0], self.cur_piece.pos[1] + WALL_KICKS_I[num_kick][j][1])
                else:
                    self.cur_piece.pos = (self.cur_piece.pos[0] + WALL_KICKS[num_kick][j][0], self.cur_piece.pos[1] + WALL_KICKS[num_kick][j][1])  
                break
            
    def get_gravity(self):
        gravity = [1/60, 1/40, 1/30, 1/20, 1/15, 1/12, 1/10, 1/8, 1/7, 1/6, 1/5, 1/4, 1/3, 1/2, 1, 1.5, 2, 3, 4, 5, 20]
        if self.level <= 20:
            self.gravity = gravity[self.level - 1]
        else:
            self.gravity = 20

    def render(self):
        if self.screen is None:
            self.rl = True
            pygame.init()
            self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
            self.clock = pygame.time.Clock()
        
        pygame.display.set_caption("Tetris")

        self.screen.fill((40, 40, 40))
        
        # Draw stats panel on the left
        pygame.draw.rect(self.screen, (40, 40, 40), pygame.Rect(0, 0, STATS_PANEL_WIDTH, WINDOW_HEIGHT))
        
        # Draw right panel for next pieces
        right_panel_x = STATS_PANEL_WIDTH + COLS * CELL_SIZE
        pygame.draw.rect(self.screen, (40, 40, 40), pygame.Rect(right_panel_x, 0, STATS_PANEL_WIDTH, WINDOW_HEIGHT))
        
        font = pygame.font.Font(None, 36)
        small_font = pygame.font.Font(None, 48)
        
        # Draw HOLD section
        hold_label = font.render("HOLD", True, (255, 255, 255))
        self.screen.blit(hold_label, (20, 480))
        
        # Draw hold piece box
        hold_box_y = 520
        pygame.draw.rect(self.screen, (60, 60, 60), pygame.Rect(20, hold_box_y, 110, 110), 2)
        
        if self.held_piece is not None:
            offsets = PIECES[self.held_piece.type][0]  # Always show rotation 0
            color = COLORS[self.held_piece.type]
            # Center the piece in the box
            center_x = 75
            center_y = hold_box_y + 55
            for r, c in offsets:
                pygame.draw.rect(
                    self.screen,
                    color,
                    pygame.Rect(center_x + c*20 - 20, center_y + r*20 - 20, 18, 18)
                )
        
        # Draw NEXT section
        next_label = font.render("NEXT", True, (255, 255, 255))
        self.screen.blit(next_label, (right_panel_x + 20, 50))
        
        # Draw next 3 pieces
        for i, piece_type in enumerate(self.next_pieces):
            box_y = 100 + i * 130
            pygame.draw.rect(self.screen, (60, 60, 60), pygame.Rect(right_panel_x + 20, box_y, 110, 110), 2)
            
            offsets = PIECES[piece_type][0]
            color = COLORS[piece_type]
            center_x = right_panel_x + 75
            center_y = box_y + 55
            for r, c in offsets:
                pygame.draw.rect(
                    self.screen,
                    color,
                    pygame.Rect(center_x + c*20 - 20, center_y + r*20 - 20, 18, 18)
                )
        
        # Draw "SCORE" label and value
        score_label = font.render("SCORE", True, (255, 255, 255))
        score_value = small_font.render(str(self.points), True, (255, 255, 255))
        self.screen.blit(score_label, (20, 50))
        self.screen.blit(score_value, (20, 90))
        
        # Draw "LEVEL" label and value
        level_label = font.render("LEVEL", True, (255, 255, 255))
        level_value = small_font.render(str(self.level), True, (255, 255, 255))
        self.screen.blit(level_label, (20, 200))
        self.screen.blit(level_value, (20, 240))
        
        # Draw "LINES" label and value
        lines_label = font.render("LINES", True, (255, 255, 255))
        lines_value = small_font.render(str(self.clears), True, (255, 255, 255))
        self.screen.blit(lines_label, (20, 350))
        self.screen.blit(lines_value, (20, 390))
        
        # Draw board (offset by STATS_PANEL_WIDTH)
        for r in range(HIDDEN_ROWS, HIDDEN_ROWS + ROWS):
            for c in range(COLS):
                value = self.board[r][c]
                color = COLORS[value] if value in COLORS else (128,128,128)
                pygame.draw.rect(
                    self.screen,
                    color,
                    pygame.Rect(STATS_PANEL_WIDTH + c*CELL_SIZE, (r-HIDDEN_ROWS)*CELL_SIZE + 25, CELL_SIZE, CELL_SIZE)
                )
                pygame.draw.rect(
                    self.screen,
                    (50,50,50),
                    pygame.Rect(STATS_PANEL_WIDTH + c*CELL_SIZE, (r-HIDDEN_ROWS)*CELL_SIZE + 25, CELL_SIZE, CELL_SIZE),
                    1
                )
        
        # Draw ghost piece
        ghost_positions = self.get_ghost_position()
        for r, c in ghost_positions:
            if r >= HIDDEN_ROWS:
                color = COLORS[self.cur_piece.type]
                pygame.draw.rect(
                    self.screen,
                    (color[0]//3, color[1]//3, color[2]//3),
                    pygame.Rect(STATS_PANEL_WIDTH + c*CELL_SIZE, (r-HIDDEN_ROWS)*CELL_SIZE + 25, CELL_SIZE, CELL_SIZE),
                    2
                )
        
        # Draw current piece on top
        if self.cur_piece is not None:
            for r, c in self.cur_piece.pieces:
                if r >= HIDDEN_ROWS:
                    color = COLORS[self.cur_piece.type]
                    pygame.draw.rect(
                        self.screen,
                        color,
                        pygame.Rect(STATS_PANEL_WIDTH + c*CELL_SIZE, (r-HIDDEN_ROWS)*CELL_SIZE + 25, CELL_SIZE, CELL_SIZE)
                    )
                    pygame.draw.rect(
                        self.screen,
                        (50,50,50),
                        pygame.Rect(STATS_PANEL_WIDTH + c*CELL_SIZE, (r-HIDDEN_ROWS)*CELL_SIZE + 25, CELL_SIZE, CELL_SIZE),
                        1
                    )

        pygame.display.flip()
        self.clock.tick(60)

    def create_board(self):
        rows = 20 + 4       # +4 to top of grid to account for pieces which rotate at the top
        cols = 10
        board = np.zeros((rows, cols), dtype=np.int32)
        return board
    
    def shuffle(self):
        self.piece_bag = np.array([1, 2, 3, 4, 5, 6, 7], dtype = int)
        np.random.shuffle(self.piece_bag)
        return self.piece_bag

    def get_type(self):
        if len(self.piece_bag) == 0:
            self.shuffle()
        type, self.piece_bag = self.piece_bag[-1], self.piece_bag[:-1]
        self.rot_index = 0
        self.update_next_pieces()
        return type

    def spawn_piece(self):
        type = self.get_type()
        if type == 1:
            cur_piece = TetriminoObj((3,3), [], 0)
        else:
            cur_piece = TetriminoObj((4,3), [], 0)
        cur_piece.type = type

        offsets = PIECES[cur_piece.type][self.rot_index]
        top_row = 4
        left_col = 3
        if cur_piece.type == 1:
            top_row = 3
        cur_piece.pieces = [(top_row + r, left_col + c) for r, c in offsets]

        for i in cur_piece.pieces:
            if self.board[i[0]][i[1]] != 0:
                if not(self.rl):
                    self.reset()
                else:
                    return None

        return cur_piece

    def rotate_cw(self):
        self.wall_kicks(0)

    def rotate_acw(self):
        self.wall_kicks(1)
    
register(
    id='Tetris-v0',
    entry_point='Tetris:TetrisEnv',
    max_episode_steps=10000,
)

if __name__ == "__main__":
    env = TetrisEnv()  # create environment
    env.run()          # start main loop