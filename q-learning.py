

from collections import namedtuple
from copy import deepcopy

Point = namedtuple('Point', 'x y')
Experience = namedtuple('Experience', 's0 a0 r0 s1')

class Game:
    def __init__(self, Board, start, end):  
    # 0 - path, 
    # 1 - happy path, 
    #-1 - bounce back, 
    #99 - goal
        height = len(Board)         
        width = len(Board[0])
        Board = [[-1] * width] + map(lambda l: [-1] + l + [-1], Board) + [[-1] * width]
        self.start = Point(start.x + 1, start.y + 1)
        self.end = Point(end.x + 1, end.y + 1)
        Board[self.start.y][self.start.x] = 80
        Board[self.end.y][self.end.x] = 99
        self.player = self.start
        self.Board = Board
        self.cum_reward = 0
        self.finished = False

    def input(self, ch):
      if self.finished:
        raise ValueError("Game already finished!")
      if ch == 'l':
        self._move(-1, 0)
      if ch == 'r':
        self._move(1, 0)
      if ch == 'u':
        self._move(0, -1)
      if ch == 'd':
        self._move(0, 1)
      return self  

    def _move(self, dx, dy):
      p = Point(self.player.x + dx, self.player.y + dy)
      if (self._get_value(p) >= 0):
        if (self._get_value(p) == 99):
            self.finished = True
        self.player = p        
        self.cum_reward += self._get_value(p)
        self._set_value(p, 0)
        

    def _get_value(self, p):
      return self.Board[p.y][p.x]

    def _set_value(self, p, v):
      self.Board[p.y][p.x] = v    

    def __str__(self):
        b = deepcopy(self.Board)
        b[self.player.y][self.player.x] = 'x'
        return reduce(lambda x, y: str(x) + ('\n') + str(y), b)

board = Game([[0, 0, 0, 0, 0],
              [0, 1, 1,-1,0],
              [0,-1, 0,10,-9]], Point(0, 2), Point(4, 1))

def single_play(_Board, Algo):
    Board = deepcopy(_Board)
    algo_input = Algo.next_input()

    history = []

    while not Board.finished:
        b = Board
        Board = deepcopy(Board)
        i = next(algo_input)
        Board.input(i)
        exp = Experience(b, i, Board.cum_reward, Board)
        Algo.feedback(exp)
        history = [exp] + history
        print str(exp.s0), '\n', exp.a0, exp.r0, '\n', str(exp.s1), '\n'

    print "Finished."    
   

class StupidAlgo:
    def next_input(self):
        yield 'u'
        yield 'r'
        yield 'u'
        yield 'r'
        yield 'r'
        yield 'r'
        yield 'd'

    def feedback(self, exp):
        pass    

class QLearningAlgo:    

    def __init__(self, width, height):
        self.gamma = 0.01
        self.alpha = 1
        self.Q = dict([(Point(i,j), random.random() * 20) for i in range(1, width+1) for j in range(1, height + 1)])
        


    def next_input(self):
        while True:
            yield 'u'

    def feedback(self, exp):
        pass
        #self.Q[]



