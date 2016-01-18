#!/usr/bin/env python
# python_example.py
# Author: Ben Goodrich
#
# This is a direct port to python of the shared library example from
# ALE provided in doc/examples/sharedLibraryInterfaceExample.cpp
import sys
from random import randrange
from ale_python_interface import ALEInterface
from itertools import groupby
import numpy as np



# Set USE_SDL to true to display the screen. ALE must be compilied
# with SDL enabled for this to work. On OSX, pygame init is used to
# proxy-call SDL_main.

#USE_SDL = False
#if USE_SDL:
#  if sys.platform == 'darwin':   
#    import pygame
#    pygame.init() 
#    ale.setBool('sound', False) # Sound doesn't work on OSX    
#  elif sys.platform.startswith('linux'):
#    ale.setBool('sound', True)
  

# ale.setString("record_screen_dir", "record")

# Load the ROM file
#ale.setBool('display_screen', True)  

def init_ale():
  ale = ALEInterface()
  ale.setInt('random_seed', 123)
  ale.setBool('frame_skip', 1)
  ale.loadROM(rom_path + '/space_invaders.bin')
  return ale

class SpaceInvadersGameVisualizer:
  def __init__(self):
    import pygame
    pygame.init()
    # Get & Set the desired settings
    w_orig = 160
    h_orig = 210

    self.arr = {0: (0,0,0),
           6: (200, 200, 0),
           20: (0, 200, 200),
           52: (200, 0, 200),
           196: (196, 0, 0),
           226: (0, 226, 0), 
           246: (146, 0, 0)}

    self.desired_width = 14
    self.desired_height = 20
    self.screen = pygame.display.set_mode((self.desired_height * 16, self.desired_width * 16))
    self.colors = sorted(self.arr.keys())

  def show_vectorized(self, vec):
    import pygame
    rect = pygame.Surface((2, 14))
    border = pygame.Surface((16, 16))
    for y in range(0, self.desired_width):
      for x in range(0, self.desired_height):
        border_rect = pygame.Rect(x, y, 16, 16)
        border.fill((255, 255, 255))
        self.screen.blit(border, (x*16, y*16))  

        for i_color in range(len(self.arr)):
          if (vec[y][x][i_color]):
            rect.fill(self.arr[self.colors[i_color]])
          else:
            rect.fill((0, 0, 0))
          self.screen.blit(rect, (x * 16 + 1 + i_color*2, y * 16 + 1))
        
    pygame.display.flip()   

  def vectorize_single_group(self, vec):
    return map(lambda e: e in vec, self.colors)

  def vectorized(self, scr):
    grouped = \
      np.reshape(
        np.swapaxes(
          np.reshape(scr, 
            (self.desired_width, 210 / self.desired_width, 
              self.desired_height, 160 / self.desired_height)), 
          1,2), 
        (self.desired_width, self.desired_height, 
          160 * 210 / self.desired_width / self.desired_height))
    return np.apply_along_axis(self.vectorize_single_group, axis = 2, arr = grouped)  

  def show(self, game):
    self.show_vectorized(self.vectorized(game.get_state()))

class SpaceInvadersGame:

  def __init__(self, ale):
    self.ale = ale
    self.finished = False    
    self.cum_reward = 0

  def get_actions(self):
    return self.ale.getMinimalActionSet()  

  def input(self, action):
    self.cum_reward = self.ale.act(action)
    if (self.ale.game_over()):
      self.finished = True
      self.ale.reset_game()

    return self

  def get_state(self):
    return self.ale.getScreen()

rom_path = '/Users/maciej/Development/atari-roms'


def test():
  %run q_learning.py
  %run ale-example.py
  ale = init_ale()
  game = SpaceInvadersGame(rom_path, ale)
  #game.show_vectorized(game.vectorized(ale.getScreen()))
  teacher = Teacher(game, RandomAlgo(game.get_actions()), SpaceInvadersGameVisualizer())
  teacher.teach(1)

    #def show(sqr):  
    #  square=pygame.Surface(1, 1)
    #  for y in range(0, 210, reduceby):
    #    for x in range(0, 160, reduceby):
    #        square.fill(arr[sqr[y][x]])
    #        draw_me=pygame.Rect((x*aspect+1), (y*aspect+1), reduceby * aspect, reduceby * aspect)
    #        screen.blit(square,draw_me)
    #  screen.blit(square,(0,0))
    #  pygame.display.flip()    


#def play_randomly(n):
#  screens = []
#  for episode in xrange(n):
#    total_reward = 0


#    while not ale.game_over():         
#      a = legal_actions[randrange(len(legal_actions))]
#      ale.act(a)
#      screens.append(ale.getScreen())
#    ale.reset_game()  
#  return screens  

#def define_background(screens):
#  import scipy.stats
#  return scipy.stats.mode(screens).mode    

#def pickle_background(background, file_name):
#  import pickle
#  with open(file_name, 'wb') as pfile:
#    pickle.dump(background, pfile, protocol = pickle.HIGHEST_PROTOCOL)  

#def unpickle_backround(file_name):
#  import pickle
#  with open(file_name, 'r') as pfile:
#    return pickle.load(pfile)    

# Play 10 episodes
#def play():
# for episode in xrange(10):
#  total_reward = 0


#  while not ale.game_over():     
#    import time
    #time.sleep(0.05)
    #show(sqr = np.reshape(ale.getScreen(), (210, 160)))
#    show_vectorized(vectorized(ale.getScreen()))
#    a = legal_actions[randrange(len(legal_actions))]
    # Apply an action and get the resulting reward
#    reward = ale.act(a);
#    total_reward += reward
#  print 'Episode', episode, 'ended with score:', total_reward
#  ale.reset_game()