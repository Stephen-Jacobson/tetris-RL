class TetriminoObj:
    def __init__(self, pos, pieces, type):
        self.pos = pos
        self.pieces = pieces
        self.type = type

    # def set_space(self, space):
    #     self.space = space

    def set_pieces(self, pieces):
        self.pieces = pieces
        

    def set_type(self, type):
        self.type = type