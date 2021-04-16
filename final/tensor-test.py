"""
Make it more robust.
Stop episode once the finger stop at the final position for 50 steps.
Feature & reward engineering.
"""
from env import ArmEnv
from rl import DDPG
import mujoco_py
import numpy as np
import gym
import tensorflow as tf
import tensorflow.contrib as tc
from collections import deque
import sys
print(sys.path)
MAX_EPISODES = 900
MAX_EP_STEPS = 200
ON_TRAIN = True

# set env
env = gym.make('FetchReach-v1')
s_dim = 13
a_dim = 4
a_bound = env.action_space.high


def stg(s):
    #print(len(s))
    ob_1 = np.reshape(s['observation'],(1,10))
    de_1 = np.reshape(s['desired_goal'],(1,3))
    return np.concatenate([ob_1,de_1],axis=1)

# set RL method (continuous)
rl = DDPG(a_dim, s_dim, a_bound)

steps = []
def train():
    # start training
    for i in range(MAX_EPISODES):
        s = env.reset()
        ep_r = 0.
        for j in range(MAX_EP_STEPS):
            # env.render()
            ss = stg(s)
            a = rl.choose_action(ss)

            s_, r, done, info = env.step(a)

            rl.store_transition(s, a, r, s_)

            ep_r += r
            if rl.memory_full:
                # start to learn once has fulfilled the memory
                rl.learn()

            s = s_
            if done or j == MAX_EP_STEPS-1:
                print('Ep: %i | %s | ep_r: %.1f | step: %i' % (i, '---' if not done else 'done', ep_r, j))
                break
    rl.save()


def eval():
    rl.restore()
    env.render()
    env.viewer.set_vsync(True)
    s = env.reset()
    while True:
        env.render()
        a = rl.choose_action(s)
        s, r, done = env.step(a)


if ON_TRAIN:
    train()
else:
    eval()



