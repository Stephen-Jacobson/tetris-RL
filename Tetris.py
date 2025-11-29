from rotations import PIECES, WALL_KICKS, WALL_KICKS_I
import random
from TetriminoObj import TetriminoObj
import numpy as np
import pygame

pygame.init()

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

screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Tetris")
clock = pygame.time.Clock()

class TetrisEnv:
    def __init__(self):
        self.board = self.create_board()
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
    
    def reset(self):
        self.board = self.create_board()
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

    def get_state(self):
        # return board + piece info as numpy array
        pass
    
    def step(self, action):
        """
        action: 0=left, 1=right, 2=rotatecw, 3=rotateacw, 4=soft down, 5=hard down, 6=hold piece
        returns: state, reward, done, info
        """
        if action == 6:
            self.hold_piece()
            self.last_rotated = False
        if action == 5:
            self.hard_down()
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
    
    def run(self):
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
            dt = clock.tick(60)
            
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
                        self.step(5)
                    if event.key == pygame.K_LEFT:
                        self.step(0)

                        left_held = True
                        left_timer = 0
                        left_happened = False

                        # reset opposite side completely
                        right_held = False
                        right_happened = False
                        right_timer = 0

                    if event.key == pygame.K_RIGHT:
                        self.step(1)

                        right_held = True
                        right_timer = 0
                        right_happened = False

                        # reset opposite side completely
                        left_held = False
                        left_happened = False
                        left_timer = 0
                    if event.key == pygame.K_UP:
                        self.step(2)
                    if event.key == pygame.K_DOWN:
                        self.step(4)
                        down_held = True
                    if event.key == pygame.K_z:
                        self.step(3)
                    if event.key == pygame.K_c:
                        self.step(6)
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
                        self.step(0)
                        left_timer = 0
                elif left_timer > das_initial:
                    self.step(0)
                    left_happened = True
                    left_timer = 0
                
                left_timer += dt
            
            if right_held:
                if right_happened:
                    if right_timer > das_repeat:
                        self.step(1)
                        right_timer = 0
                elif right_timer > das_initial:
                    self.step(1)
                    right_happened = True
                    right_timer = 0
                
                right_timer += dt
            if down_held:
                self.step(4)

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
        self.new_piece()

    def check_end(self):
        dont_end = False
        for i in self.cur_piece.pieces:
            if (i[0] > 3):
                dont_end = True
        if not(dont_end):
            self.reset()

    def new_piece(self):
        self.check_end()
        self.check_clear()
        self.cur_piece = self.spawn_piece()
        self.lock = False
        self.lock_timer = 0
        self.lock_moves = 0
        self.rot_index = 0

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
        screen.fill((40, 40, 40))
        
        # Draw stats panel on the left
        pygame.draw.rect(screen, (40, 40, 40), pygame.Rect(0, 0, STATS_PANEL_WIDTH, WINDOW_HEIGHT))
        
        # Draw right panel for next pieces
        right_panel_x = STATS_PANEL_WIDTH + COLS * CELL_SIZE
        pygame.draw.rect(screen, (40, 40, 40), pygame.Rect(right_panel_x, 0, STATS_PANEL_WIDTH, WINDOW_HEIGHT))
        
        font = pygame.font.Font(None, 36)
        small_font = pygame.font.Font(None, 48)
        
        # Draw HOLD section
        hold_label = font.render("HOLD", True, (255, 255, 255))
        screen.blit(hold_label, (20, 480))
        
        # Draw hold piece box
        hold_box_y = 520
        pygame.draw.rect(screen, (60, 60, 60), pygame.Rect(20, hold_box_y, 110, 110), 2)
        
        if self.held_piece is not None:
            offsets = PIECES[self.held_piece.type][0]  # Always show rotation 0
            color = COLORS[self.held_piece.type]
            # Center the piece in the box
            center_x = 75
            center_y = hold_box_y + 55
            for r, c in offsets:
                pygame.draw.rect(
                    screen,
                    color,
                    pygame.Rect(center_x + c*20 - 20, center_y + r*20 - 20, 18, 18)
                )
        
        # Draw NEXT section
        next_label = font.render("NEXT", True, (255, 255, 255))
        screen.blit(next_label, (right_panel_x + 20, 50))
        
        # Draw next 3 pieces
        for i, piece_type in enumerate(self.next_pieces):
            box_y = 100 + i * 130
            pygame.draw.rect(screen, (60, 60, 60), pygame.Rect(right_panel_x + 20, box_y, 110, 110), 2)
            
            offsets = PIECES[piece_type][0]
            color = COLORS[piece_type]
            center_x = right_panel_x + 75
            center_y = box_y + 55
            for r, c in offsets:
                pygame.draw.rect(
                    screen,
                    color,
                    pygame.Rect(center_x + c*20 - 20, center_y + r*20 - 20, 18, 18)
                )
        
        # Draw "SCORE" label and value
        score_label = font.render("SCORE", True, (255, 255, 255))
        score_value = small_font.render(str(self.points), True, (255, 255, 255))
        screen.blit(score_label, (20, 50))
        screen.blit(score_value, (20, 90))
        
        # Draw "LEVEL" label and value
        level_label = font.render("LEVEL", True, (255, 255, 255))
        level_value = small_font.render(str(self.level), True, (255, 255, 255))
        screen.blit(level_label, (20, 200))
        screen.blit(level_value, (20, 240))
        
        # Draw "LINES" label and value
        lines_label = font.render("LINES", True, (255, 255, 255))
        lines_value = small_font.render(str(self.clears), True, (255, 255, 255))
        screen.blit(lines_label, (20, 350))
        screen.blit(lines_value, (20, 390))
        
        # Draw board (offset by STATS_PANEL_WIDTH)
        for r in range(HIDDEN_ROWS, HIDDEN_ROWS + ROWS):
            for c in range(COLS):
                value = self.board[r][c]
                color = COLORS[value] if value in COLORS else (128,128,128)
                pygame.draw.rect(
                    screen,
                    color,
                    pygame.Rect(STATS_PANEL_WIDTH + c*CELL_SIZE, (r-HIDDEN_ROWS)*CELL_SIZE + 25, CELL_SIZE, CELL_SIZE)
                )
                pygame.draw.rect(
                    screen,
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
                    screen,
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
                        screen,
                        color,
                        pygame.Rect(STATS_PANEL_WIDTH + c*CELL_SIZE, (r-HIDDEN_ROWS)*CELL_SIZE + 25, CELL_SIZE, CELL_SIZE)
                    )
                    pygame.draw.rect(
                        screen,
                        (50,50,50),
                        pygame.Rect(STATS_PANEL_WIDTH + c*CELL_SIZE, (r-HIDDEN_ROWS)*CELL_SIZE + 25, CELL_SIZE, CELL_SIZE),
                        1
                    )

        pygame.display.flip()
        clock.tick(60)

    def create_board(self):
        rows = 20 + 4       # +4 to top of grid to account for pieces which rotate at the top
        cols = 10
        board = np.zeros((rows, cols), dtype = int)
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
                self.reset()

        return cur_piece

    def rotate_cw(self):
        self.wall_kicks(0)

    def rotate_acw(self):
        self.wall_kicks(1)
    
if __name__ == "__main__":
    env = TetrisEnv()  # create environment
    env.run()          # start main loop