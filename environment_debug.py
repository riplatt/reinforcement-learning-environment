import Environment
import numpy as np
import tensorflow as tf

from tf_agents.environments import tf_py_environment, ActionDiscretizeWrapper
from tf_agents.environments import utils

def load_pieces(file_path):
    pieces = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                pieces.append(np.array([int(x) for x in line.split(' ')]))
    return pieces

tile_set = [
    [ 1, 17,  0,  0],
    [ 1,  5,  0,  0],
    [ 9, 17,  0,  0],
    [17,  9,  0,  0],
    [12, 13,  0,  9],
    [21,  1,  0,  9],
    [ 8,  9,  0, 17],
    [16, 13,  0, 17],
    [15,  9,  0,  5],
    [20,  5,  0,  5],
    [10,  1,  0, 13],
    [ 8, 17,  0, 13],
    [ 6, 20, 10, 21],
    [ 8, 15,  6,  8],
    [21,  8, 14, 16],
    [12,  8, 14, 21]
]

env = Environment.puzzleEnv(tile_set=tile_set)

board_img = env.render(mode='human')
board_img.save("debug_board_000.png")

# utils.validate_py_environment(env, episodes=5)

print('Action Spec: ', env.action_spec())
print('Observation Spec: ', env.time_step_spec().observation)
print('Reward Spec: ', env.time_step_spec().reward)

board_img = env.render(mode='human')
board_img.save("debug/images/debug_board_000.png")

def pack_action(action: np.int32, pos1: np.int32, pos2: np.int32):
    return action + (pos1 << 2) + (pos2 << 6)

moves = [
    pack_action(1, 0, 0),
    pack_action(2, 1, 5),
    pack_action(2, 2, 6),
    pack_action(2, 3, 6),
    pack_action(1, 4, 7),
    pack_action(0, 5, 14),
    pack_action(0, 6, 13),
    pack_action(3, 7, 8),
    pack_action(1, 8, 11),
    pack_action(2, 9, 15),
    pack_action(0, 10, 12),
    pack_action(3, 11, 15),
    pack_action(0, 12, 13),
    pack_action(0, 13, 15),
    pack_action(0, 14, 15),
    pack_action(3, 15, 15),
]

for i, move in enumerate(moves):
    time_step=env._step(move)
    print('Reward: {}, Solved Edges: {}'.format(time_step.reward, env._solved_edges))
    board_img = env.render(mode='human')
    board_img.save("debug/images/debug_board_{}.png".format(str(i+1).rjust(3, '0')))



# board_edges, board_pieces = env.render()
# print("https://e2.bucas.name/#board_w=16&board_h=16&board_edges={}&board_pieces={}&motifs_order=jblackwood".format(board_edges, board_pieces))
