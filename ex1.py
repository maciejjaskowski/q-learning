
import q_learning as q
import numpy as np

board_simple = q.Game([[0, 0, 0, 0, 0],
                     [0, 1, 1,-1,99],
                     [0,-1, 0,10,-9]], q.Point(0, 2), q.Point(4, 1))

board_big = q.Game([[ 0, 0, 0, 0, 0,-9, 0, 0, 0, 9],
                  [-1, 9,-1,-1, 0, 0, 0, 0,-9, 0],
                  [-1, 9,-1, 0,-1,-1, 0, 0, 0, 0],
                  [-1, 9, 9, 9, 9,-1, 0, 0, 0, 0],
                  [-1,-1,-1,-1,-1,-1, 0, 0, 0, 0],
                  [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], 
                 q.Point(0, 5),
                 q.Point(7, 5))

game_big2 = q.Game([[ 0, 0, 0, 0, 0,-9, 0, 0, 0, 9],
                  [-1, 9,-1,-1, 0, 0, 0, 0,-9, 0],
                  [-1, 9,-1, 0,-1,-1, 0, 0, 0, 0],
                  [-1, 9, 9, 9, 9,-1, 0, 0, 0, 0],
                  [-1,-1,-1,-1,-1,-1, 0, 0, 0, 0],
                  [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], 
                 q.Point(0, 5),
                 q.Point(7, 0))

game_collect_all = q.CollectAllGame([[ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-1],
                                     [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-1],
                                     [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-1],
                                     [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-1],
                                     [ 0, 0, 0,-9,-9,-9,-9, 0, 0, 0, 0],
                                     [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                     [ 0, 0, 0,-9,-9,-9,-9, 0, 0, 0, 0],
                                     [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], 
                                   q.Point(0, 6),
                                   q.Point(9, 6))
        
pc_man =    q.PcManGame([[-1,-1, 9, 9, 9, 9, 9, 9,-1,-1],
                  [-1, 9, 9, 9,-1, 9,-1, 9, 9, 9],
                  [-1, 9,-1, 9,-1,-1, 9, 9,-1, 9],
                  [ 9, 9, 9, 9, 9, 9, 9,-1, 9, 9],
                  [ 9,-1,-1, 9,-1,-1, 9,-1,-1, 9],
                  [ 9, 9, 9, 9, 9, 9, 9, 9, 9, 9]], 
                 q.Point(0, 5),
                 ghost_start = q.Point(7, 0)                 
                 )        


def on_policy_ex():
  game = game_collect_all
  q_algo1 = q.QLearningOnPolicyTDControlAlgo(['u', 'd', 'l', 'r'], game.get_state())

  q_algo1.gamma = 0.9

  teacher = q.Teacher(game, q_algo1)

  q_algo1.alpha = 1
  q_algo1.epsilon = 0.1
  teacher.teach(100, verbose = lambda x: False)

  q_algo1.alpha = 0.1
  q_algo1.epsilon = 0.1
  teacher.teach(2000, verbose = lambda x: False)

  q_algo1.alpha = 0
  q_algo1.epsilon = 0.1  
  teacher.teach(1)

def collect_all_ex():
  game = game_collect_all
  q_algo1 = q.QLearningOffPolicyTDControlAlgo(['u', 'd', 'l', 'r'], game.get_state())

  q_algo1.gamma = 0.9

  teacher = q.Teacher(game, q_algo1)

  q_algo1.alpha = 0.1
  q_algo1.epsilon = 0.1
  teacher.teach(2000, verbose = lambda x: False)

  #q_algo1.alpha = 0.1
  #q_algo1.epsilon = 0.1
  #teacher.teach(5000, verbose = lambda x: False)

  q_algo1.alpha = 0
  q_algo1.epsilon = 0  
  teacher.teach(1)  

game = game_collect_all
q_algo1 = q.QLearningOffPolicyWithRepeatAlgo(['u', 'd', 'l', 'r'], game.get_state())

q_algo1.gamma = 0.9

teacher = q.Teacher(game, q_algo1)

q_algo1.alpha = 0.1
q_algo1.epsilon = 0.1
teacher.teach(100, verbose = lambda x: False)


class Tester:
  def test(self, game, algo_factory, n = 100):
    return np.array([q.Teacher(game, algo_factory(game)).teach(100, verbose = lambda x: False) 
      for i in range(0, n)]).mean(axis = 0)
        

def teach_off_repeat():
  def factory(game):
    q_algo1 = q.QLearningOffPolicyWithRepeatAlgo(['u', 'd', 'l', 'r'], game.get_state(), sample_size = 1, history_length = 20)
    q_algo1.gamma = 0.9


    q_algo1.alpha = 0.1
    q_algo1.epsilon = 0.1
    return q_algo1
  return Tester().test(game_collect_all, factory)
  
def teach_off():
  def factory(game):
    q_algo1 = q.QLearningOffPolicyTDControlAlgo(['u', 'd', 'l', 'r'], game.get_state())
    q_algo1.gamma = 0.9

    q_algo1.alpha = 0.1
    q_algo1.epsilon = 0.1
    return q_algo1
  return Tester().test(game_collect_all, factory)


  # compare variation between repeat and not-repeat: plt.plot(map(lambda x: np.concatenate([x[0], x[1]], axis = 0), zip(u, t)))

