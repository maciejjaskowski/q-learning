
import q_learning as q

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
        
pc_man =    q.PcManGame([[-1,-1, 9, 9, 9, 9, 9, 9,-1,-1],
                  [-1, 9, 9, 9,-1, 9,-1, 9, 9, 9],
                  [-1, 9,-1, 9,-1,-1, 9, 9,-1, 9],
                  [ 9, 9, 9, 9, 9, 9, 9,-1, 9, 9],
                  [ 9,-1,-1, 9,-1,-1, 9,-1,-1, 9],
                  [ 9, 9, 9, 9, 9, 9, 9, 9, 9, 9]], 
                 q.Point(0, 5),
                 ghost_start = q.Point(7, 0)                 
                 )        


#q_algo1 = QLearningOffPolicyTDControlAlgo(['u', 'd', 'l', 'r'], game_big2.get_state())
q_algo1 = q.QLearningOnPolicyTDControlAlgo(['u', 'd', 'l', 'r'], game_big2.get_state())

q_algo1.gamma = 0.995

teacher = q.Teacher(game_big2, q_algo1)

q_algo1.alpha = 1
q_algo1.epsilon = 0.1
teacher.teach(5000, verbose = lambda x: False)

q_algo1.alpha = 0.1
q_algo1.epsilon = 0.1
teacher.teach(5000, verbose = lambda x: False)

q_algo1.epsilon = 0
q_algo1.alpha = 0
teacher.teach(1)

