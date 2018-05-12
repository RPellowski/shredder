import json
import copy

HSL_PARAMS = {
        "paper" :     [0.15, 0.70, 0.96, 0.13, 0.17, 0.15],
        "bluelines" : [0.18, 0.80, 0.62, 0.02, 0.31, 0.15],
        "redlines" :  [0.10, 0.98, 0.93, 0.02, 0.15, 0.01],
        "pieces" :    [0.40, 0.50, 0.50, 0.401, 0.501, 0.501],
        "blackink" :  [0.01, 0.01, 0.01, 0.10, 0.20, 0.40]
    }

def jdefault(o):
    if isinstance(o, set):
        return list(o)
    return {o.__class__.__name__: o.__dict__}

class basic():
    def __repr__(self):
        return (json.dumps(self, default=jdefault))

N_SEQ_VIEWS=8
COARSEST_SEQ_VIEW=2**(N_SEQ_VIEWS-1)
SEQ_VIEWS=[2**x for x in range(N_SEQ_VIEWS)]

class Sequence(basic):
    '''
    Provide multiple scaled views on a series of data points. Coarse granularity
    provides for matching a larger number of candidates while allowing more
    false positives. Finer granularity reduces the number of false positives.

    Input data points is a list of integers. A view is calculated only when
    needed and saved so the calculation is done once. However, the coarsest view
    is calculated on instantiation.

    The class enables tracking of matches by granularity.
    '''
    def __init__(self, indata, use_absolute = True):
        self.indata = copy.copy(indata)
        self.views = {}
        self.use_absolute = use_absolute
        for v in SEQ_VIEWS:
            self.views[v] = None
        self.views[1] = copy.copy(self.indata)
        for i, item in enumerate(self.views[1]):
            self.views[1][i] = abs(item) if use_absolute else item

    def get_view(self, view):
        '''
        Returns an average for "view" number of samples.
        '''
        if self.views[view] is None:
            #print("View", view, "not in views")
            self.views[view] = []
            divisor = view
            l = len(self.views[1])
            for i in range(0, l, view):
                #print("1",i,divisor,l)
                if i + divisor > l:
                    divisor = l - i
                    #print("new divisor", divisor)
                item = 0
                for j in range(divisor):
                    item += abs(self.views[1][i + j] if self.use_absolute else item)
                self.views[view].append((item + divisor/2)/divisor)
                #print("2",i,divisor,l)
        return(self.views[view])

class Edge(basic):
    '''
    Represents a side of a piece in a way that can be compared to other pieces,
    with high probability of match and with low probability of false positive.
    Assumptions:
        angle is known
        polarity is unknown
        metric
            same for both sides of an edge
            depends on absolute horizontal or vertical piece orientation
            contains relevance to nearby colors
    Investigations
        blue lines as anchors
    '''
    def __init__(self):
        pass

    @classmethod
    def make_edge_set(cls, blob):
        '''
        Given an image blob as input, create all the edges that describe the
        piece and return them as a list of Sequences.
        '''
        edge_set = {}
        #edge_set["blue"] = []
        #while (blue-related edges to add)
        #    edge_set["blue"].append(cls(create sequence based on blue)))
        return edge_set

# This list grows as pieces are assembled into larger sized aggregations. The
# list shrinks over time until, ideally, one assembly remains, consisting of the
# solved puzzle.
assemblies = []

class Assembly(basic):
    '''
    At a minimum, contains:
    - list of Pieces that make up the assembly
    - layer (1:1 layer:Assembly)
    - translation vector (used in the process of moving to another layer)
    - representation of its edges
    '''

    def __init__(self):
        self.pieces = []
        self.layer = 0
        self.xlate_x = 0
        self.xlate_y = 0

# This list is at least as large as the number of blobs. It can grow as
# candidate blobs are created for the pieces with unknwon orientation angles and
# polarity. When a piece is matched edgewise, its orientation angle and polarity
# becomes known and sibling candidates can be removed from the list.
Pieces = {}

class Piece(basic):
    '''
    A Piece contains the metadata that represents the image blob of a physical
    item. Statistics on color will tell us what algorithms to apply for angle
    and polarity. A definite angle and polarity moves the piece into analysis
    of edges.

    If we cannot determine angle and polarity immediately, generating candidate
    rotations could be useful as candidates will allow further processing on
    edges. However, it expands the space so this generation must be done
    properly.

    Edge metadata will be used for matching Pieces into Assemblies (and
    placement on a temporary or permanent layer). Once a Piece is matched, its
    angle and polarity are known and candidates can be released.

    When final assembly is complete, all Pieces will be on the same layer at the
    destination coordinates.
    '''
    def __init__(self, label, x, y, w, h):
        self.label = label
        self.src_x = x
        self.src_y = y
        self.src_w = w
        self.src_h = h
        self.src_n_pix = w * h # area of bounding box, sum of all pix
        self.src_n_bline_pix = 0
        self.src_n_rline_pix = 0
        self.src_n_bink_pix = 0
        self.src_n_rink_pix = 0
        self.src_n_paper_pix = 0
        self.src_n_misc_pix = 0
        self.src_n_bg_pix = 0
        self.dst_b_angle = False
        self.dst_b_polarity = False
        self.siblings = []  # Pieces, variations on angle

        # Features that might change across the set of temp candidates
        self.candidate = 0
        self.layer = 1
        self.dst_angle = 0.0
        self.dst_polarity = 0.0
        self.dst_result = 0
        self.dst_x = 0
        self.dst_y = 0
        self.dst_w = w
        self.dst_h = h
        self.edges = []  # Edges

    def set_b_angle(self, b_angle):
        self.dst_b_angle = b_angle
        self.set_result()

    def set_angle(self, angle):
        self.dst_angle = angle
        self.set_result()

    def set_b_polarity(self, b_polarity):
        self.dst_b_polarity = b_polarity
        self.set_result()

    def set_polarity(self, polarity):
        self.dst_polarity = polarity
        self.set_result()

    def set_result(self):
        self.dst_result = (45.0 +
            int(self.dst_b_angle) * self.dst_angle +
            int(self.dst_b_polarity) * self.dst_polarity) % 360.0 - 45.0

if __name__ == "__main__":
    p = Piece(10, 10, 30, 40, 1000)
    print(str(p))
    a = Assembly()
    a.pieces.append(p)
    print(a)
    e = Edge()
    print(e)

    '''
    import numpy as np
    s = Sequence(range(16))
    #list(np.array([range(17)])*np.array([-1,1,-1,1])), use_absolute = False)
    print(s)
    for i in SEQ_VIEWS:
        s.get_view(i)
    for i in SEQ_VIEWS:
        print(s.get_view(i))
    print(s)
    print(SEQ_VIEWS)
    print(N_SEQ_VIEWS)
    print(COARSEST_SEQ_VIEW)
    '''
