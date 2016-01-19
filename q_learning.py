from __future__ import division

from collections import namedtuple
from copy import deepcopy
from random import random, choice, uniform, sample, randrange
from time import sleep

from bisect import bisect
import itertools
import numpy as np
import pandas as pd

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
        self.valid_player_locations = [(x,y) for row, y in zip(self.Board, range(len(self.Board))) 
                                             for x in range(len(row)) if row[x]>=0 ]

        self.last_action = None                                 

    def get_actions(self):
      return ['l', 'r', 'u', 'd']    

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
      self.last_action = ch  
      return self  

    def _move(self, dx, dy):
      p = Point(self.player.x + dx, self.player.y + dy)
      if (self._get_value(p) >= 0):
        if (p == self.end):
            self.finished = True
            self.cum_reward += 100
        self.player = p        
        self.cum_reward += self._get_value(p) - 1
        self._set_value(p, 0)
      else:
        self.cum_reward += self._get_value(p) - 1
        

    def _get_value(self, p):
      return self.Board[p.y][p.x]

    def _set_value(self, p, v):
      self.Board[p.y][p.x] = v    

    def __str__(self):
        b = deepcopy(self.Board)
        b[self.player.y][self.player.x] = '_'
        b[self.end.y][self.end.x] = 'E'
        sub = {0: ' . ', -1: '[.]', -9: '[*]', 9: ' + ', '_': ' _ ', 'E': ' E '}

        return '\n'.join( 
            map(lambda l: "".join(map(lambda x: "%03s"%sub[x], l)), b))

class GameVisualizer:
  def show(self, game):
    print "Cumulative Reward: ", game.cum_reward
    print "Action: ", game.last_action
    print game
    #print map(lambda action: (action, self.algo._get_q(Board.get_state(), action)), self.algo.actions)          
    print "\n"
    sleep(0.1)   

  def next_game(self):         
    pass

class CollectAllGame:
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

      self.to_gather = len(filter(lambda x: x == 9, self.get_state()))                                 
      self.last_action = None

    def get_state(self):
        return tuple([item for sublist in self.Board for item in sublist] + [self.player])

    def get_actions(self):
      return ['l', 'r', 'u', 'd']    

    def input(self, ch):
      if self.finished:
        raise ValueError("Game already finished!")
      d = { 'l': (-1, 0), 'r': (1, 0), 'u': (0, -1), 'd': (0, 1) }  
      self._move(d[ch][0], d[ch][1])      
      self.last_action = ch
      return self  

    def _move(self, dx, dy):
      p = Point(self.player.x + dx, self.player.y + dy)
      if (self._get_value(p) >= 0):
        if (p == self.end and self.to_gather == 0):
            self.finished = True
            self.cum_reward += 20
        self.player = p        
        self.cum_reward += self._get_value(p) - 0.01
        self._set_value(p, 0)
      else:
        self.cum_reward += self._get_value(p) - 0.01
        

    def _get_value(self, p):
      return self.Board[p.y][p.x]

    def _set_value(self, p, v):
      self.Board[p.y][p.x] = v    

    def __str__(self):
      b = deepcopy(self.Board)
      b[self.player.y][self.player.x] = 'X'
      b[self.end.y][self.end.x] = 'E'
      sub = {0: ' . ', -1: '[.]', -9: '[*]', 9: ' + ', 'X': ' X ', 'E': ' E '}

      return '\n'.join( 
          map(lambda l: "".join(map(lambda x: "%03s"%sub[x], l)), b))        


class CollectAllGameVisualizer:
  
  def show(self, game):
    print "Cumulative Reward: ", game.cum_reward
    print "Action: ", game.last_action
    print game
    #print map(lambda action: (action, self.algo._get_q(Board.get_state(), action)), self.algo.actions)          
    print "\n"
    sleep(0.1)   

  def next_game(self):         
    pass

class PcManGameVisualizer:
  def __init__(self):
    pass

  def show(self, game):
    print "Cumulative Reward: ", game.cum_reward
    print "Action: ", game.last_action
    print game
    #print map(lambda action: (action, self.algo._get_q(Board.get_state(), action)), self.algo.actions)          
    print "\n"
    sleep(0.1)

  def next_game(self):
    pass    

class GameNoVisualizer:
  def show(self, game):
    pass

  def next_game(self):
    pass        

class EveryNVisualizer:
  def __init__(self, n, visualizer):
    self.n = n
    self.right_visualizer = visualizer
    self.visualizer = GameNoVisualizer()
    self.i = 0

  def show(self, game):
    self.visualizer.show(game)
    
  def next_game(self):
    self.i += 1    
    if (self.i % self.n == self.n - 1):
      self.visualizer = self.right_visualizer
    else:
      self.visualizer = GameNoVisualizer()

class PcManGame:
    def __init__(self, Board, start, ghost_start):  
    # 0 - path, 
    # 1 - happy path, 
    #-1 - bounce back, 

        height = len(Board) + 2        
        width = len(Board[0]) + 2

        Board = [[-1] * width] + map(lambda l: [-1] + l + [-1], Board) + [[-1] * width]
        self.player = Point(start.x + 1, start.y + 1)

        self.Board = Board
        self.cum_reward = 0
        self.finished = False
        self.ghost = Point(ghost_start.x + 1, ghost_start.y + 1)
        self.ghost_direction = choice(self._valid_direction(self.ghost))
        self.to_gather = len(filter(lambda x: x == 9, self.get_state()))
        self.last_action = None

    def get_state(self):
        return tuple([item for sublist in self.Board for item in sublist] + [self.player])

    def get_actions(self):
      return ['l', 'r', 'u', 'd']    

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
      if ch == ' ':
        pass  
      if (self.player == self.ghost):
        self.finished = True
      else:    
        self._move_ghost()

      self.last_action = ch  

      if (self.player == self.ghost):
        self.finished = True

      
      if (self._get_value(self.player) > 0):
        self.to_gather -= 1
        self.cum_reward += self._get_value(self.player)
    

      if (self.to_gather == 0):
            self.finished = True
            self.cum_reward += 5000  


      self._set_value(self.player, 0)

      return self  

    def _valid_direction(self, pos):
      def dire(dx, dy):
        return lambda p: Point(p.x + dx, p.y + dy)
      return filter(lambda d: self._get_value(d(pos)) >= 0, [dire(0, 1), dire(0, -1), dire(1, 0), dire(-1, 0)])


    def _move_ghost(self):
       next_ghost_pos = self.ghost_direction(self.ghost)
       if (self._get_value(next_ghost_pos) < 0):
          self.ghost_direction = choice(self._valid_direction(self.ghost))
          next_ghost_pos = self.ghost_direction(self.ghost)
       self.ghost = next_ghost_pos 

    def _move(self, dx, dy):
      p = Point(self.player.x + dx, self.player.y + dy)
      if (self._get_value(p) >= 0):
        self.player = p        
      
      
        

    def _get_value(self, p):
      return self.Board[p.y][p.x]

    def _set_value(self, p, v):
      self.Board[p.y][p.x] = v    

    def __str__(self):
        b = deepcopy(self.Board)
        b[self.player.y][self.player.x] = 'X'
        b[self.ghost.y][self.ghost.x] = 'G'
        sub = {0: ' . ', -1: '[.]', -9: '[*]', 9: ' + ', 'X': ' X ', 'E': ' E ', 'G': ' G '}

        return '\n'.join( 
            map(lambda l: "".join(map(lambda x: "%03s"%sub[x], l)), b))



class MountainCarGame:

  def __init__(self):
    self.position = -0.5 #random() * ( 0.5 - (-1.2)) + (-1.2)
    self.throttle = 0
    self.velocity = 0 #random() * (0.07 - (-0.07)) + (-0.07)
    self.cum_reward = 0
    self.finished = False


  def input(self, ch):
    import math
    if ch in self.get_actions():
      self.throttle = ch
      

    
    self.velocity = self.velocity + 0.001 * self.throttle - 0.0025 * math.cos(3 * self.position)
    if (self.velocity < -0.07):
      self.velocity = -0.07
    if (self.velocity > 0.07):
      self.velocity = 0.07  

    self.position = self.position + self.velocity
    if self.position > 0.5:
      self.finished = True      
    if self.position < -1.2:
      self.position = -1.2
      self.velocity = 0  
        
    self.cum_reward = self.cum_reward - 1 + abs(self.position + 0.5)
    return self

  

  def get_actions(self):
    return [-1, 0, 1]  

  def get_state(self):    
    return (self.position, self.velocity)

class MountainCarGameVisualizer:
  def __init__(self, algo, print_every_n = 10):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    self.fig = plt.figure()
    self.velocity_position = self.fig.add_subplot(2,2,1)
    self.expected_reward = self.fig.add_subplot(2, 2, 2, projection = '3d')
    self.direction = self.fig.add_subplot(2, 2, 3)

    self.hl, = self.velocity_position.plot([], [], color='k', linestyle='-')
    self.mi, = self.velocity_position.plot([], [], color='red', linestyle='-')
    self.ma, = self.velocity_position.plot([], [], color='red', linestyle='-')
    self.dir = [0,0,0]
    self.dir[0], = self.direction.plot([], [], color = 'red', linestyle='', marker = 'x', ms = 1)
    self.dir[1], = self.direction.plot([], [], color = 'blue', linestyle='', marker = 'x', ms = 1)
    self.dir[2], = self.direction.plot([], [], color = 'green', linestyle='', marker = 'x', ms = 1)
    self.velocity_position.set_xlim([-1.3, 0.6])
    self.velocity_position.set_ylim([-0.1, 0.1])
    self.expected_reward.set_xlim([-1.3, 0.6])
    self.expected_reward.set_ylim([-0.1, 0.1])
    self.direction.set_xlim([-1.3, 0.6])
    self.direction.set_ylim([-0.1, 0.1])
    self.history_x = []
    self.history_y = []
    self.i = -1
    self.print_every_n = print_every_n
    self.xlim = [0,0]
    self.algo = algo


  def show(self, game):
    self.i += 1
    import matplotlib.pyplot as plt
    import matplotlib

    if len(self.history_x) > 100:
      self.history_x = self.history_x[1:]
      self.history_y = self.history_y[1:]

    newx, newy = game.get_state()
    self.history_x.append(newx)
    self.history_y.append(newy)
    self.xlim[0] = min(self.xlim[0], newx)
    self.xlim[1] = max(self.xlim[1], newx)

    


    if self.i % self.print_every_n == 0:
      self.hl.set_xdata(self.history_x)
      self.hl.set_ydata(self.history_y)
      self.mi.set_xdata([self.xlim[0]]*2)
      self.mi.set_ydata([-0.08, 0.08])
      self.ma.set_xdata([self.xlim[1]]*2)
      self.ma.set_ydata([-0.08, 0.08])

      pos = np.arange(-0.9, 0.2, 0.03)
      vel = np.arange(-0.04, 0.04, 0.004)
      Pos,Vel = np.meshgrid(pos, vel)
      #expected_reward = np.reshape([ self.algo.pi_value((_pos, _vel)) for _vel in vel for _pos in pos  ], np.shape(Pos))

      direction = pd.DataFrame([ (_pos, _vel, self.algo.pi((_pos, _vel))) for _vel in vel for _pos in pos  ])
      direction.columns = ["pos", "vel", "throttle"]
      #direction = np.reshape([ self.algo.pi((_pos, _vel)) for _vel in vel for _pos in pos  ], np.shape(Pos))

      #self.direction.contour(Pos, Vel, direction, 6,
      #           colors='k')

      col = {-1: 'red', 0: 'blue', 1: 'green'}
      for name, group in direction.groupby('throttle'):
        if (name >= 0):
          self.dir[name + 1].set_xdata(group['pos'])
          self.dir[name + 1].set_ydata(group['vel'])
      #self.expected_reward.plot_surface(Pos, Vel, expected_reward)

      self.fig.canvas.draw()
      self.fig.canvas.flush_events()

      import time
      time.sleep(0.01)




    
    #print(game.get_state(), game.throttle)
    
  def next_game(self):
    pass

class Tilings: # n_tilings = 5, n = 9

  def __init__(self, x, y, n, n_tilings):
    self.dx = (x[1] - x[0])
    self.dy = (y[1] - y[0])
    self.range_x = np.array(range(n-1)) / (n-1) * self.dx + x[0] 
    self.range_y = np.array(range(n-1)) / (n-1) * self.dy + y[0] 
    self.n = n
    self.n_tilings = n_tilings

  def tiling(self):
    distortion_x = [ x + random() * self.dx / (self.n-1) for x in self.range_x]
    distortion_y = [ y + random() * self.dy / (self.n-1) for y in self.range_y]
    return distortion_x, distortion_y

  def tilings(self): 
    return [ self.tiling() for i in range(self.n_tilings)]

  def one_at(self, i):
    result = np.zeros(self.n * self.n)
    result[i] = 1
    return result

  def calc(self):
    ts = self.tilings()
    return (lambda p: tuple([(bisect(tiling[0], p[0]), bisect(tiling[1], p[1])) for tiling in ts]), ts)
    #return [one_at(y * n + x) for x, y in locations]
    # flatten = reduce(lambda x, y: x+y, result)
   

class StupidAlgo:
    def action(self):
        yield 'u'
        yield 'r'
        yield 'u'
        yield 'r'
        yield 'r'
        yield 'r'
        yield 'd'

    def feedback(self, exp):
        pass   

class RandomAlgo:
    def __init__(self, legal_actions):
      self.legal_actions = legal_actions

    def action(self):
      while True:
        yield self.legal_actions[randrange(len(self.legal_actions))]

    def feedback(self, x):
        pass    

class SARS:    

    def __init__(self, actions, init_state):
        self.gamma = 0.5
        self.alpha = 1
        self.epsilon = 0.2
        self.Q = {}
        self.actions = actions
        self.state = init_state

    def _get_q(self, state, action):
        k = (state, action)
        v = getOrElse(self.Q, k, 99)
        self.Q[k] = v
        return v


    def action(self):
        while True:
          if (random() < self.epsilon):
            yield choice(self.actions)
          else:
            yield self.pi(self.state)
          
    def pi(self, state):
      return max(map(lambda action: (action, self._get_q(state, action)), 
            self.actions), key = lambda opt: opt[1])[0]


    def feedback(self, exp):
        q = self._get_q
        qB = q(exp.s1, self.pi(self.state))
        q0 = q(exp.s0, exp.a0)
        self.Q[(exp.s0, exp.a0)] = (
            (1-self.alpha) * q0 + self.alpha * (exp.r0 + self.gamma * qB)
          )
        self.state = exp.s1    

class SARSRepeat: #Dyna-Q

    def __init__(self, actions, init_state, sample_size = 5, history_length = 20, initial_q = 0):
        self.gamma = 0.5
        self.alpha = 1
        self.epsilon = 0.2
        self.Q = {}
        self.actions = actions
        self.state = init_state
        self.memory = []
        self.sample_size = sample_size
        self.history_length = history_length
        self.initial_q = initial_q

    def _get_q(self, state, action):
        k = (state, action)
        #print "_---------------"
        #print(k, self.Q, self.initial_q)
        v = getOrElse(self.Q, k, self.initial_q)
        self.Q[k] = v
        return v


    def action(self):
        while True:
          if (random() < self.epsilon):
            yield choice(self.actions)
          else:
            yield self.pi(self.state)
          
    def pi(self, state):
      return max(map(lambda action: (action, self._get_q(state, action)), 
            self.actions), key = lambda opt: opt[1])[0]


    def feedback(self, exp):
        q = self._get_q

        self.memory += [(exp.s0, exp.a0, exp.r0, q(exp.s1, self.pi(self.state)))]
        if (len(self.memory) > self.history_length):
          self.memory = self.memory[1:]

        for s0, a0, r0, qB in sample(self.memory, min(len(self.memory), self.sample_size)):
          self.Q[(s0, a0)] = (1-self.alpha) * q(s0, a0) + self.alpha * (r0 + self.gamma * qB)

        self.state = exp.s1  


    

class SARSA:    # or SARSA

    def __init__(self, actions, init_state, initial_q):
        self.gamma = 0.5
        self.alpha = 1
        self.epsilon = 0.2
        self.Q = {}
        self.actions = actions
        self.state = init_state
        self.initial_q = initial_q

    def _get_q(self, state, action):
        k = (state, action)
        v = getOrElse(self.Q, k, self.initial_q)
        self.Q[k] = v
        return v

    def action(self):
      self.next_action = self._action(self.state)
      while True:
        yield self.next_action

    def _action(self, state):     
      if (random() < self.epsilon):
        return choice(self.actions)
      else:
        return self.pi(state)
          
    def pi(self, state):
      return max(map(lambda action: (action, self._get_q(state, action)), 
            self.actions), key = lambda opt: opt[1])[0]

    def feedback(self, exp):
        q = self._get_q
        a1 = self._action(exp.s1)
        q1 = q(exp.s1, a1)
        q0 = q(exp.s0, exp.a0)
        self.Q[(exp.s0, exp.a0)] = (
          (1-self.alpha) * q0 + self.alpha * (exp.r0 + self.gamma * q1)
        )
        self.state = exp.s1
        self.next_action = a1

class SARSALambda: 

    def __init__(self, actions, init_state, initial_q, memory_size):
      self.lmbda = 0.8
      self.gamma = 0.5
      self.alpha = 1
      self.epsilon = 0.2
      self.Q = {}
      self.actions = actions
      self.state = init_state
      self.initial_q = initial_q
      self.memory = []
      self.memory_size = memory_size

    def _get_q(self, state, action):
        k = (state, action)
        v = getOrElse(self.Q, k, self.initial_q)
        self.Q[k] = v
        return v

    def action(self):
      self.next_action = self._action(self.state)
      while True:
        yield self.next_action

    def _action(self, state):     
      if (random() < self.epsilon):
        return choice(self.actions)
      else:
        return self.pi(state)
          
    def pi(self, state):
      return max(map(lambda action: (action, self._get_q(state, action)), 
            self.actions), key = lambda opt: opt[1])[0]

    def feedback(self, exp):
        q = self._get_q
        a1 = self._action(exp.s1)
        q1 = q(exp.s1, a1)
        q0 = q(exp.s0, exp.a0)

        delta = exp.r0 + self.gamma * q1 - q0
        self.memory.append((exp.s0, exp.a0, delta))

        if len(self.memory) > self.memory_size:
          self.memory = self.memory[1:]

        for (s, a, delta), d in zip(reversed(self.memory), range(len(self.memory))):
          self.Q[(s, a)] = self.Q[(s, a)] + self.alpha * delta * (self.lmbda ** (-d))
        
        self.state = exp.s1
        self.next_action = a1     


class SARSALambdaGradientDescent: 

    def __init__(self, actions, init_state, initial_q, memory_size, tilings, initial_theta):
      self.lmbda = 0.8
      self.gamma = 0.7
      self.alpha = 0.1
      self.epsilon = 0.1
      
      self.actions = actions
      self.action_ind = dict(zip(self.actions, range(len(self.actions))))
      self.state = init_state
      self.initial_q = initial_q
      self.visited = set()
      self.memory = []
      self.memory_size = memory_size

      self.theta = np.concatenate([np.array([initial_theta]), np.array([initial_theta]), np.array([initial_theta])], axis = 0)
      self.tilings = tilings
      
      self.e = np.concatenate([np.array([[0] * len(initial_theta)]), np.array([[0] * len(initial_theta)]), np.array([[0] * len(initial_theta)])], axis = 0)

    def phi(self, state):
      return np.array(self.tilings(state))

    def q(self, state, action):
      return sum(self.theta[self.action_ind[action]][self.phi(state)])  

    def q_positive(self, state, action):
      if (tuple(self.phi(state)), action) in self.visited:
        return self.q(state, action)
      else:
        return self.initial_q  

    def action(self):
      self.next_action = self._action(self.state)
      while True:
        yield self.next_action

    def _action(self, state):     
      if (random() < self.epsilon):
        return choice(self.actions)
      else:

        return self.pi(state)
          
    def pi(self, state):      
      return max(map(lambda action: (action, self.q_positive(state, action)), self.actions), 
                 key = lambda opt: opt[1])[0]

    def pi_value(self, state):
      return max(map(lambda action: (action, self.q_positive(state, action)), self.actions), 
                 key = lambda opt: opt[1])[1]

    def feedback(self, exp):
        #print(exp)
        a1 = self._action(exp.s1)
        print ("first visit? ", tuple(self.phi(exp.s0)), exp.a0, \
            ((tuple(self.phi(exp.s0)), exp.a0) not in self.visited))
        self.visited.add((tuple(self.phi(exp.s0)), exp.a0))        
        
        delta = exp.r0 + self.gamma * self.q(exp.s1, a1) - self.q(exp.s0, exp.a0)
        print("delta: ", delta, "sarsa: ", exp.s0, exp.a0, exp.r0, exp.s1, a1)


        
        self.theta = self.theta + self.alpha * delta * self.e
        self.e = self.gamma * self.lmbda * self.e
          
        
        for a in self.actions:
            self.e[self.action_ind[a]][self.phi(exp.s1)] = 0
        self.e[self.action_ind[a1]][self.phi(exp.s1)] = 1      
        
        print("phi1: ", self.phi(exp.s1))
        
        self.state = exp.s1
        self.next_action = a1                       
        

class Teacher:
    def __init__(self, game, algo, game_visualizer, state_vectorizer = lambda x: x):
      self.game = game
      self.algo = algo
      self.game_visualizer = game_visualizer
      self.state_vectorizer = state_vectorizer


    def get_state(self, Game): 
      return self.state_vectorizer(Game.get_state())

    def teach(self, episodes):
      return [self.single_play(15000) for i in range(episodes)]

    def single_play(self, n_steps = float("inf")):
      Game = deepcopy(self.game)
      algo_input = self.algo.action()

      history = []

      i_steps = 0

      while not Game.finished and i_steps < n_steps:
          i_steps += 1
          
          old_state = self.get_state(Game)
          old_cum_reward = Game.cum_reward

          action = next(algo_input)
          Game.input(action)

          exp = Experience(old_state, action, Game.cum_reward - old_cum_reward, self.get_state(Game))
          self.algo.feedback(exp)

          history = [exp] + history
          self.game_visualizer.show(Game)  

      if Game.finished:
        print "Finished after ", i_steps, " steps"    
      else:
        print "Failure."  

      print Game.cum_reward  

      self.game_visualizer.next_game()

      return (i_steps, Game.cum_reward)

#Todo: 
