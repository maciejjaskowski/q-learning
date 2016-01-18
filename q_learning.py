

from collections import namedtuple
from copy import deepcopy
from random import random, choice, uniform, sample, randrange
from time import sleep
import itertools

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
                                         for x in range(len(row)) if row[x]>=0]

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
        self.cum_reward += self._get_value(p) - 0.1
        self._set_value(p, 0)
      else:
        self.cum_reward += self._get_value(p) - 0.1
        

    def _get_value(self, p):
      return self.Board[p.y][p.x]

    def _set_value(self, p, v):
      self.Board[p.y][p.x] = v    

    def __str__(self):
      b = deepcopy(self.Board)
      b[self.player.y][self.player.x] = 'X'
      b[self.end.y][self.end.x] = 'E'
      sub = {0: ' . ', -1: '[.]', -9: '[*]', 9: ' + ', '_': ' _ ', 'E': ' E '}

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

class QLearningOffPolicyTDControlAlgo:    

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

class QLearningOffPolicyWithRepeatAlgo: #Dyna-Q

    def __init__(self, actions, init_state, sample_size = 5, history_length = 20):
        self.gamma = 0.5
        self.alpha = 1
        self.epsilon = 0.2
        self.Q = {}
        self.actions = actions
        self.state = init_state
        self.memory = []
        self.sample_size = sample_size
        self.history_length = history_length

    def _get_q(self, state, action):
        k = (state, action)
        v = getOrElse(self.Q, k, 9)
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

class QLearningOnPolicyTDControlAlgo:    # or SARSA

    def __init__(self, actions, init_state):
        self.gamma = 0.5
        self.alpha = 1
        self.epsilon = 0.2
        self.Q = {}
        self.actions = actions
        self.state = init_state

    def _get_q(self, state, action):
        k = (state, action)
        v = getOrElse(self.Q, k, 9)
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
        

class Teacher:
    def __init__(self, game, algo, game_visualizer):
      self.game = game
      self.algo = algo
      self.game_visualizer = game_visualizer

    def teach(self, episodes):
      return [self.single_play(50000) for i in range(episodes)]

    def single_play(self, n_steps = float("inf")):
      Game = deepcopy(self.game)
      algo_input = self.algo.action()

      history = []

      i_steps = 0

      while not Game.finished and i_steps < n_steps:
          i_steps += 1
          
          old_state = Game.get_state()
          old_cum_reward = Game.cum_reward

          action = next(algo_input)
          Game.input(action)

          exp = Experience(old_state, action, Game.cum_reward - old_cum_reward, Game.get_state())
          self.algo.feedback(exp)

          history = [exp] + history
          self.game_visualizer.show(Game)  

      if Game.finished:
        print "Finished after ", i_steps, " steps"    
      else:
        print "Failure."  

      self.game_visualizer.next_game()

      return (i_steps, Game.cum_reward)

#Todo: 
