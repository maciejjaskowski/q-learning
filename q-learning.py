

from collections import namedtuple
from copy import deepcopy
from random import random, choice

Point = namedtuple('Point', 'x y')
Experience = namedtuple('Experience', 's0 a0 r0 s1')

def getOrElse(d, k, el):
    if (k in d):
        return d[k]
    else:
        return el

class Game:
    def __init__(self, Board, start, end):  
    # 0 - path, 
    # 1 - happy path, 
    #-1 - bounce back, 

        height = len(Board) + 2        
        width = len(Board[0]) + 2

        Board = [[-1] * width] + map(lambda l: [-1] + l + [-1], Board) + [[-1] * width]
        self.end = Point(end.x + 1, end.y + 1)
        self.player = Point(start.x + 1, start.y + 1)

        self.Board = Board
        self.cum_reward = 0
        self.finished = False

    def get_state(self):
        return tuple([item for sublist in self.Board for item in sublist] + [self.player])

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
        if (p == self.end):
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
        b[self.end.y][self.end.x] = 'E'
        return reduce(lambda x, y: str(x) + ('\n') + str(y), b)

board = Game([[0, 0, 0, 0, 0],
              [0, 1, 1,-1,99],
              [0,-1, 0,10,-9]], Point(0, 2), Point(4, 1))

def single_play(_Board, Algo, n_steps = float("inf")):
    Board = deepcopy(_Board)
    algo_input = Algo.next_input()

    history = []

    i_steps = 0

    while not Board.finished and i_steps < n_steps:
        i_steps += 1
        print Board.player
        old_state = Board.get_state()
        old_cum_reward = Board.cum_reward

        action = next(algo_input)
        Board.input(action)

        exp = Experience(old_state, action, Board.cum_reward - old_cum_reward, Board.get_state())
        Algo.feedback(exp)

        history = [exp] + history        

    print Board.get_state()
    print Algo.Q

    if Board.finished:
      print "Finished after ", i_steps, " steps"    


    return (i_steps, Board.cum_reward)
   

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

    def __init__(self, actions, init_state):
        self.gamma = 0.01
        self.alpha = 1
        self.epsilon = 0.1
        self.Q = {}
        self.actions = actions
        self.state = init_state

    def _get_q(self, k):
        v = getOrElse(self.Q, k, random())
        self.Q[k] = v
        return v


    def next_input(self):
        while True:
          options = map(lambda action: (action, self._get_q((self.state, action))), 
            self.actions)  
          if (random() < self.epsilon):
            yield choice(options)[0]
          else:
            yield max(options, key = lambda opt: opt[1])[0]
          


    def feedback(self, exp):
        est = max(map(lambda action: self._get_q((exp.s1, action)), self.actions))
        prev_Q = self._get_q((exp.s0, exp.a0))
        self.Q[(exp.s0, exp.a0)] = prev_Q + self.alpha * (exp.r0 - 1 + self.gamma * est)
        self.state = exp.s1


class Teacher:
    def __init__(self, board, algo):
      self.board = board
      self.algo = algo

    def teach(self, epochs):
      return [single_play(self.board, self.algo, 10000) for i in range(epochs)]
        
