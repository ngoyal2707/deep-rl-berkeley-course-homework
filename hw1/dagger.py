import tensorflow
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import sys
import argparse
import pickle as pkl
from keras.models import load_model

from run_expert import get_data
from run_expert import test_model

from sklearn.metrics import mean_squared_error
# fix random seed for reproducibility
import load_policy

ENV_NAME_TO_REAL_ENV_NAME = {
    'ant': 'Ant-v1',
    'halfcheetah': 'HalfCheetah-v1',
    'hopper': 'Hopper-v1',
    'humanoid': 'Humanoid-v1',
    'reacher': 'Reacher-v1',
    'walker': 'Walker2d-v1'
}


"""
python3 dagger.py --train --env_name humanoid --render --init_rollouts 5 --iterations 40 --per_iteration_rollout 4
python3 dagger.py --train --env_name ant --render --init_rollouts 5 --iterations 20 --per_iteration_rollout 4
python3 dagger.py --train --env_name halfcheetah --render --init_rollouts 5 --iterations 20 --per_iteration_rollout 4
python3 dagger.py --train --env_name reacher --render --init_rollouts 10 --iterations 30 --per_iteration_rollout 4
python3 dagger.py --train --env_name walker --render --init_rollouts 5 --iterations 20 --per_iteration_rollout 4
python3 dagger.py --train --env_name hopper --render --init_rollouts 2 --iterations 20 --per_iteration_rollout 4
"""
class Dagger:
    EPOCH_COUNT = 30
    BATCH_SIZE = 1024

    MODEL_SIZE = [256, 128, 128]

    def __init__(self, args):
        self.args = args

    def create_model(self, input_dim, output_dim):
        self.model = Sequential()
        self.model.add(Dense(self.MODEL_SIZE[0], input_dim=input_dim, activation='relu'))
        
        for layer_size in self.MODEL_SIZE[1:]:
            self.model.add(Dense(layer_size, activation='relu'))
        
        self.model.add(Dense(output_dim, activation='linear'))
        self.model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])
        

    def fit(self):
        initial_data = get_data(load_policy.load_policy('experts/' + ENV_NAME_TO_REAL_ENV_NAME[args.env_name] + '.pkl'), 
            ENV_NAME_TO_REAL_ENV_NAME[args.env_name], False, args.max_timesteps, args.init_rollouts)

        train_X = initial_data['observations']
        train_Y = initial_data['actions']
        input_dim = train_X.shape[1]
        sample_size,_, output_dim = train_Y.shape
        train_Y = np.reshape(train_Y, (sample_size, output_dim))

        self.create_model(input_dim, output_dim)
        
        self.model.fit(train_X, train_Y, epochs=self.EPOCH_COUNT, batch_size=self.BATCH_SIZE)
        self.model.save(self.get_file_name())

        for i in range(args.iterations):
            new_data = test_model(behavior_clone, ENV_NAME_TO_REAL_ENV_NAME[args.env_name], args.render, args.max_timesteps, args.per_iteration_rollout, True)
            new_train_X = new_data['observations']
            new_train_Y = new_data['actions']
            sample_size,_, output_dim = new_train_Y.shape
            new_train_Y = np.reshape(new_train_Y, (sample_size, output_dim))

            train_X = np.vstack((train_X, new_train_X))
            train_Y = np.vstack((train_Y, new_train_Y))
            
            self.model=load_model(self.get_file_name())
            self.model.fit(train_X, train_Y, epochs=self.EPOCH_COUNT, batch_size=self.BATCH_SIZE, verbose=0)
            print("===============================================================")
            print(i)
            print(train_X.shape)
            print("===============================================================")
            self.model.save(self.get_file_name())


        # print(self.model.get_weights())
        return self.predict

    def predict(self, X):
        predictions = self.model.predict(X)
        a,b = predictions.shape
        return np.reshape(predictions, (a,1,b))

    def get_file_name(self):
        return 'dagger_models/' + self.args.env_name + "_" + "_".join(list(map(str, self.MODEL_SIZE))) + '.h5'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_file')
    parser.add_argument('--env_name')

    parser.add_argument('--train', action='store_true')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--create_dataset', action='store_true')
    parser.add_argument("--max_timesteps", type=int)
    parser.add_argument('--init_rollouts', type=int, default=5,
                        help='Number of initial roll outs')
    parser.add_argument('--iterations', type=int, default=20,
                        help='iterations')
    parser.add_argument('--per_iteration_rollout', type=int, default=2,
                        help='per iteration rollouts')


    args = parser.parse_args()

    behavior_clone = Dagger(args)


    if args.train:
        policy_fn = behavior_clone.fit()
    else:
        test_model(behavior_clone, ENV_NAME_TO_REAL_ENV_NAME[args.env_name], args.render, args.max_timesteps, args.num_rollouts)

    import gc
    gc.collect()
