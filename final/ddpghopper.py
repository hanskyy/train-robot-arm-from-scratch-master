from ddpgrlhopper import DDPG
import numpy as np
import gym
import subprocess
import matplotlib.pyplot as plt
from PIL import Image

ON_TRAIN = False


# set env as Fetch and set dim
env = gym.make('Swimmer-v2')

# print(env.action_space.high[0])
# print(env.action_space)		#3
# print(env.observation_space)	#11


a_dim = env.action_space.shape[0]                               # 3
s_dim = env.observation_space.shape[0]  # 11
#s_dim = 13
#a_dim = 4
a_bound = env.action_space.high[0]

# max_action = env.action_space.high[0]
# min_action = env.action_space.low[0]

# state = env.reset().values()
# obs, curr_pos, goal_pos = state
# exp = np.concatenate((obs, curr_pos, goal_pos))
# exp.reshape((s_dim,))
# obs_log = []
# curr_pos_log = []
# next_obs_log = []
# next_curr_pos_log = []
rl = DDPG(input_dims=s_dim, env=env, n_actions=a_dim)

MAX_EPISODES = 1000
MAX_EP_STEPS = 500

def train():
    # start training
    for i in range(MAX_EPISODES):

        state = env.reset()
        # state = env.reset().values()
        ep_reward = 0

        for j in range(MAX_EP_STEPS):
            # env.render()
            action = rl.choose_action(state)

            next_state, reward, done, _ = env.step(action)
            # next_obs, next_curr_pos, goal_pos = next_state.values()

            #send new experience to replay buffer
            rl.store_transition(state, action, reward, next_state, done)

            # start to learn once has fulfilled the memory
            if rl.memory_full:
                rl.learn()

            # update the states
            state = next_state
            ep_reward += reward

            if done or j == MAX_EP_STEPS-1:
                print('Ep: %i | %s | ep_r: %.1f | step: %i' % (i, '---' if not done else 'done', ep_reward, j))
                break
    rl.save()


def eval():
    rl.restore()
    state = env.reset()
    # state = env.reset().values()
    subprocess.call(['rm', '-rf', 'frames'])
    subprocess.call(['mkdir', '-p', 'frames'])
    time_step_counter = 0
    while True:
        #env.render()
        action = rl.choose_action(state)
        next_state, reward, done, _ = env.step(action)
        # next_obs, next_curr_pos, goal_pos = next_state.values()

        #img = env.render(height=480, width=480)
        image_data = env.render(mode="rgb_array")
        # plt.imshow(img)
        # plt.show()
        # image_data = env.render(height=480, width=480,camera_id=-1)
        img = Image.fromarray(image_data, 'RGB')
        img.save("frames/frame-%.10d.png" % time_step_counter)
        time_step_counter += 1
        state = next_state
        if done or time_step_counter > 100:
            break
    subprocess.call([
        'ffmpeg', '-framerate', '50', '-y', '-i', 'frames/frame-%010d.png', '-r', '30', '-pix_fmt', 'yuv420p',
        'video_name.mp4'
    ])

if ON_TRAIN:
    train()
else:
    eval()