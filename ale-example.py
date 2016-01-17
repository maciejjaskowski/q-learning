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


if len(sys.argv) < 2:
  sys.argv.append('/Users/maciej/Development/atari-roms/space_invaders.bin')
  #print 'Usage:', sys.argv[0], 'rom_file'
  #sys.exit()

ale = ALEInterface()

# Get & Set the desired settings
ale.setInt('random_seed', 123)

# Set USE_SDL to true to display the screen. ALE must be compilied
# with SDL enabled for this to work. On OSX, pygame init is used to
# proxy-call SDL_main.

USE_SDL = False
if USE_SDL:
  if sys.platform == 'darwin':   
    import pygame
    pygame.init() 
    ale.setBool('sound', False) # Sound doesn't work on OSX    
  elif sys.platform.startswith('linux'):
    ale.setBool('sound', True)
  

# ale.setString("record_screen_dir", "record")

# Load the ROM file
#ale.setBool('display_screen', True)  

import pygame
pygame.init()
w_orig = 160
h_orig = 210

reduceby = 1

aspect = 1


square=pygame.Surface((reduceby * aspect, reduceby * aspect))
arr = {0: (0,0,0),
       6: (200, 200, 0),
       20: (0, 200, 200),
       52: (200, 0, 200),
       196: (196, 0, 0),
       226: (0, 226, 0), 
       246: (146, 0, 0)}


#background = pygame.image.load('ms_pacman_background.bmp').convert_alpha()
#background_pxls = pygame.surfarray.pixels2d(background)

#background_pxls = background_pxls[range(0, len(background_pxls), 2)]



board = pygame.Surface((210, 160))

desired_width = 14
desired_height = 20

screen = pygame.display.set_mode((desired_height * 16, desired_width * 16))


colors = [0, 6, 20, 52, 196, 226, 246]

#scr[scr - background == 0] = 0

def vectorize_single_group(vec):
    return map(lambda e: e in vec, colors)

def vectorized(scr):
  grouped = np.reshape(np.swapaxes(np.reshape(scr, (desired_width, 210 / desired_width, desired_height, 160 / desired_height)), 1,2), (desired_width, desired_height, 160 * 210 / desired_width / desired_height))
  return np.apply_along_axis(vectorize_single_group, axis = 2, arr = grouped)

def show_vectorized(vec):
  rect=pygame.Surface((2, 14))
  border = pygame.Surface((16, 16))
  for y in range(0, desired_width):
    for x in range(0, desired_height):
      border_rect = pygame.Rect(x, y, 16, 16)
      border.fill((255, 255, 255))
      screen.blit(border, (x*16, y*16))      
      for i_color in range(len(colors)):
        if (vec[y][x][i_color]):
          rect.fill(arr[colors[i_color]])
        else:
          rect.fill((0, 0, 0))
        screen.blit(rect, (x * 16 + 1 + i_color*2, y * 16 + 1))
      
  pygame.display.flip()          


def show(sqr):  

  
  
#  print (scr - background_pxls == 0)
  

  #sqr = [scr[i:i+w_orig] for i in xrange(0, w_orig * h_orig, w_orig)]

  #shape = sqr.shape
  #m = np.reshape(map(lambda x: arr_s[x], np.reshape(sqr, (shape[0] * shape[1],))), shape)
  #sqr[(sqr - m == 0)] = 0
  #sums = [reduce(lambda c, x: c + [x*(c[-1] + x)], row, [0]) for row in (sqr == blue).astype(int)]
  #sqr[sums == 8]

  for y in range(0, 210, reduceby):
    for x in range(0, 160, reduceby):
        square.fill(arr[sqr[y][x]])
        draw_me=pygame.Rect((x*aspect+1), (y*aspect+1), reduceby * aspect, reduceby * aspect)
        screen.blit(square,draw_me)
  screen.blit(square,(0,0))
  pygame.display.flip()    



ale.setBool('frame_skip', 1)
ale.loadROM(sys.argv[1])
print "--"
# Get the list of legal actions
legal_actions = ale.getMinimalActionSet()

#show(sqr = np.reshape(ale.getScreen(), (210, 160)))



def play_randomly(n):
  screens = []
  for episode in xrange(n):
    total_reward = 0


    while not ale.game_over():         
      #show(sqr = np.reshape(ale.getScreen(), (210, 160)))
      a = legal_actions[randrange(len(legal_actions))]
      ale.act(a)
      screens.append(ale.getScreen())
    ale.reset_game()  
  return screens  

def define_background(screens):
  import scipy.stats
  return scipy.stats.mode(screens).mode    

def pickle_background(background, file_name):
  import pickle
  with open(file_name, 'wb') as pfile:
    pickle.dump(background, pfile, protocol = pickle.HIGHEST_PROTOCOL)  

def unpickle_backround(file_name):
  import pickle
  with open(file_name, 'r') as pfile:
    return pickle.load(pfile)    

# Play 10 episodes
def play():
 for episode in xrange(10):
  total_reward = 0


  while not ale.game_over():     
    import time
    #time.sleep(0.05)
    #show(sqr = np.reshape(ale.getScreen(), (210, 160)))
    show_vectorized(vectorized(ale.getScreen()))
    a = legal_actions[randrange(len(legal_actions))]
    #a = legal_actions[i % 2]
    # Apply an action and get the resulting reward
    reward = ale.act(a);
    total_reward += reward
  print 'Episode', episode, 'ended with score:', total_reward
  ale.reset_game()