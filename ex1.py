
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



def random_on_mountain_car_game():
  game = q.MountainCarGame()
  q_algo = q.RandomAlgo(game.get_actions())
  visualizer = q.MountainCarGameVisualizer()

  teacher = q.Teacher(game, q_algo, visualizer)

  teacher.teach(1)

def on_policy_is_more_about_safety():
  game = game_collect_all
  q_algo1 = q.SARSA(game.get_actions(), game.get_state(), 20)

  q_algo1.gamma = 0.5

  teacher = q.Teacher(game, q_algo1, q.GameNoVisualizer())

  q_algo1.alpha = 0.1
  q_algo1.epsilon = 0.1
  teacher.teach(1500)

  teacher = q.Teacher(game, q_algo1, q.CollectAllGameVisualizer())
  
  q_algo1.epsilon = 0
  teacher.teach(1)  

initial_theta = [0] * 9 * 9 * 5

def sarsa_lambda_on_mountain_car_game():
  game = q.MountainCarGame()
  my_tilings, til = q.Tilings((-1.2, 0.5), (-0.07, 0.07), tile_in_row, n_tilings).calc()
  final_tilings = lambda (a, b): [x + y * tile_in_row + i * tile_in_row * tile_in_row for (x, y), i in zip(my_tilings((a,b)), range(n_tilings * n_tilings * tile_in_row))]
  q_algo1 = q.SARSALambda(game.get_actions(), game.get_state(), 0, memory_size = 40, state_transformation = final_tilings)
  q_algo1.lmbda = 0.9

  q_algo1.gamma = 0.5

  visualizer = q.MountainCarGameVisualizer(q_algo1)
  teacher = q.Teacher(game, q_algo1, visualizer)

  teacher.teach(1)

def sarsa_lambda_gradient_descent():
  game = q.MountainCarGame()

  tile_in_row = 9
  n_tilings = 5
  my_tilings, til = q.Tilings((-1.2, 0.5), (-0.07, 0.07), tile_in_row, n_tilings).calc()
  final_tilings = lambda (a, b): [x + y * tile_in_row + i * tile_in_row * tile_in_row for (x, y), i in zip(my_tilings((a,b)), range(n_tilings * n_tilings * tile_in_row))]

  initial_theta = [1] * n_tilings * tile_in_row * tile_in_row

  #dot = sum(initial_theta[phi((2,2))])

  q_algo1 = q.SARSALambdaGradientDescent(game.get_actions(), game.get_state(), 
    initial_q = 0, memory_size = 20, tilings = final_tilings, initial_theta = initial_theta)

  q_algo1.epsilon = 0
  q_algo1.lmbda = 0.9
  q_algo1.gamma = 0.9
  q_algo1.alpha = 0.1
  
  teacher = q.Teacher(game, q_algo1, q.MountainCarGameVisualizer(q_algo1))
  teacher.teach(1)

  teacher = q.Teacher(game, q_algo1, q.GameNoVisualizer())
  teacher.teach(30)
  

def sarsa_lambda_example():
  game = game_collect_all
  q_algo1 = q.SARSALambda(game.get_actions(), game.get_state(), 20, 4)
  q_algo1.lmbda = 0.8

  q_algo1.gamma = 0.5

  teacher = q.Teacher(game, q_algo1, q.GameNoVisualizer())

  q_algo1.alpha = 0.1
  q_algo1.epsilon = 0.1
  teacher.teach(80)

  #q_algo1.alpha = 0.1
  #q_algo1.epsilon = 0.1
  #teacher.teach(5000, verbose = lambda x: False)

  teacher = q.Teacher(game, q_algo1, q.CollectAllGameVisualizer())
  #q_algo1.alpha = 0
  q_algo1.epsilon = 0  
  teacher.teach(1)  

def sarsa_lambda_example2():
  game = game_big2
  q_algo1 = q.SARSALambda(game.get_actions(), game.get_state(), 20, 4)
  q_algo1.lmbda = 0.9999

  q_algo1.gamma = 0.5

  teacher = q.Teacher(game, q_algo1, q.GameNoVisualizer())

  q_algo1.alpha = 0.1
  q_algo1.epsilon = 0.1
  teacher.teach(1500)

  #q_algo1.alpha = 0.1
  #q_algo1.epsilon = 0.1
  #teacher.teach(5000, verbose = lambda x: False)

  teacher = q.Teacher(game, q_algo1, q.CollectAllGameVisualizer())
  #q_algo1.alpha = 0
  q_algo1.epsilon = 0  
  teacher.teach(1)  

def off_policy_example():
  game = game_collect_all
  q_algo1 = q.SARSRepeat(game.get_actions(), game.get_state())

  q_algo1.gamma = 0.5

  teacher = q.Teacher(game, q_algo1, q.GameNoVisualizer())

  q_algo1.alpha = 0.1
  q_algo1.epsilon = 0.1
  teacher.teach(1500)

  #q_algo1.alpha = 0.1
  #q_algo1.epsilon = 0.1
  #teacher.teach(5000, verbose = lambda x: False)

  teacher = q.Teacher(game, q_algo1, q.CollectAllGameVisualizer())
  #q_algo1.alpha = 0
  q_algo1.epsilon = 0  
  teacher.teach(1)  



class Tester:
  def test(self, game, algo_factory, n = 100):
    return np.array([q.Teacher(game, algo_factory(game)).teach(100, verbose = lambda x: False) 
      for i in range(0, n)]).mean(axis = 0)
        

def teach_off_repeat():
  def factory(game):
    q_algo1 = q.SARSRepeat(game.get_actions(), game.get_state(), sample_size = 1, history_length = 20)
    q_algo1.gamma = 0.9


    q_algo1.alpha = 0.1
    q_algo1.epsilon = 0.1
    return q_algo1
  return Tester().test(game_collect_all, factory)
  
def teach_off():
  def factory(game):
    q_algo1 = q.SARSRepeat(game.get_actions(), game.get_state())
    q_algo1.gamma = 0.9

    q_algo1.alpha = 0.1
    q_algo1.epsilon = 0.1
    return q_algo1
  return Tester().test(game_collect_all, factory)


  # compare variation between repeat and not-repeat: plt.plot(map(lambda x: np.concatenate([x[0], x[1]], axis = 0), zip(u, t)))

