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


if len(sys.argv) < 2:
  print 'Usage:', sys.argv[0], 'rom_file'
  sys.exit()

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

reduceby = 4

aspect = 2

screen = pygame.display.set_mode((160 * aspect, 160 * aspect))
square=pygame.Surface((reduceby * aspect, reduceby * aspect))
arr = {0: (255, 255, 255),
       74: (0, 0, 200), #sciany i punkty
       144: (255, 0, 0),
       68: (120, 120, 120),
       70: (200, 200, 200),
       38: (0, 200, 200),
       40: (0, 0, 0),
       24: (255, 255, 0),
       42: (255, 200, 255), #pcman
       88: (100, 100, 0),
       184: (100, 0, 200)}


def show(scr):
  
  sqr = [scr[i:i+w_orig] for i in xrange(0, w_orig * h_orig, w_orig)]
  for y in range(0, 160, reduceby):
    for x in range(0, 160, reduceby):                
        square.fill(arr[sqr[y+8][x]])
        draw_me=pygame.Rect((x*aspect+1), (y*aspect+1), reduceby * aspect, reduceby * aspect)
        screen.blit(square,draw_me)
  #screen.blit(square,(0,0))
  pygame.display.flip()    

ale.setBool('frame_skip', 1)
ale.loadROM(sys.argv[1])
print "--"
# Get the list of legal actions
legal_actions = ale.getMinimalActionSet()

show(scr = ale.getScreen())

# Play 10 episodes
for episode in xrange(10):
  total_reward = 0
  if (episode % 2 == 1):
    ale.setBool('display_screen', False)
  else:
    ale.setBool('display_screen', True)  

  while not ale.game_over():     
    show(scr = ale.getScreen()) 
    a = legal_actions[randrange(len(legal_actions))]
    # Apply an action and get the resulting reward
    reward = ale.act(a);
    total_reward += reward
  print 'Episode', episode, 'ended with score:', total_reward
  ale.reset_game()