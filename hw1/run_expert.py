#!/usr/bin/env python

"""
Code to load an expert policy and generate roll-out data for behavioral cloning.
Example usage:
    python run_expert.py experts/Humanoid-v1.pkl Humanoid-v1 --render \
            --num_rollouts 20

Author of this script and included expert policies: Jonathan Ho (hoj@openai.com)
"""

import pickle
import tensorflow as tf
import numpy as np
import tf_util
import gym
import load_policy
from sklearn.metrics import mean_squared_error
from keras.models import load_model


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('expert_policy_file', type=str)
    parser.add_argument('envname', type=str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument("--max_timesteps", type=int)
    parser.add_argument('--num_rollouts', type=int, default=20,
                        help='Number of expert roll outs')
    args = parser.parse_args()

    print('loading and building expert policy')
    policy_fn = load_policy.load_policy(args.expert_policy_file)
    print('loaded and built')

    with tf.Session():
        tf_util.initialize()

        env = gym.make(args.envname)
        max_steps = args.max_timesteps or env.spec.timestep_limit
        print(env.spec.timestep_limit)
        returns = []
        observations = []
        actions = []
        for i in range(args.num_rollouts):
            print('iter', i)
            obs = env.reset()
            done = False
            totalr = 0.
            steps = 0
            while not done:
                action = policy_fn(obs[None,:])
                observations.append(obs)
                actions.append(action)
                obs, r, done, _ = env.step(action)
                totalr += r
                steps += 1
                if args.render:
                    env.render()
                if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
                if steps >= max_steps:
                    break
            returns.append(totalr)

        print('returns', returns)
        print('mean return', np.mean(returns))
        print('std of return', np.std(returns))

        expert_data = {'observations': np.array(observations),
                       'actions': np.array(actions)}
        print(expert_data['observations'].shape)


def get_data(policy_fn, envname, render=True, max_timesteps=1000, num_rollouts=20):
    with tf.Session():
        tf_util.initialize()
        return helper(policy_fn, envname, render, max_timesteps, num_rollouts)

def helper(policy_fn, envname, render=True, max_timesteps=1000, num_rollouts=20, get_expert_data=False):
    env = gym.make(envname)
    max_steps = max_timesteps or env.spec.timestep_limit
    returns = []
    observations = []
    actions = []

    if get_expert_data:
        expert_policy_fn = load_policy.load_policy('experts/' + envname + '.pkl')
    
    for i in range(num_rollouts):
        obs = env.reset()
        done = False
        totalr = 0.
        steps = 0
        while not done:
            action = policy_fn(obs[None,:])
            observations.append(obs)
            
            if get_expert_data:
                expert_action = expert_policy_fn(obs[None, :])
                actions.append(expert_action)
            else:
                actions.append(action)
            
            obs, r, done, _ = env.step(action)
            totalr += r
            steps += 1
            if render:
                env.render()
                
            if steps >= max_steps:
                break
        returns.append(totalr)

    print('returns', returns)
    print('mean return', np.mean(returns))
    print('std of return', np.std(returns))

    expert_data = {'observations': np.array(observations),
                   'actions': np.array(actions)}
    return expert_data


def test_model(behavior_clone, envname, render=True, max_timesteps=1000, num_rollouts=20, get_expert_data=False):
    with tf.Session():
        tf_util.initialize()
        behavior_clone.model = load_model(behavior_clone.get_file_name())
        return helper(behavior_clone.predict, envname, render, max_timesteps, num_rollouts, get_expert_data)


if __name__ == '__main__':
    main()
