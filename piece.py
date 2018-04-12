import json

class Piece():
    def __init__(self, x, y, npix):
        self.src_x = x
        self.src_y = y
        self.src_n_pix = npix
        self.src_n_bline_pix = 0
        self.src_n_rline_pix = 0
        self.src_n_bink_pix = 0
        self.src_n_rink_pix = 0
        self.src_n_bg_pix = 0
        self.src_n_other_pix = 0
        self.dst_b_angle = False
        self.dst_angle = 0
        self.dst_b_polarity = False
        self.dst_polarity = 0
        #self.ghost

    def __repr__(self):
        return json.dumps(self.__dict__)

if __name__ == "__main__":
    p = Piece(10, 10, 1000)
    print(repr(p))
