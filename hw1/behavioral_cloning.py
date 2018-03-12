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
python3 behavioral_cloning.py --create_dataset --env_name 'humanoid' --num_rollouts 50

python3 behavioral_cloning.py --train --data_file data/Humanoid-v1_50.pkl --env_name humanoid

python3 behavioral_cloning.py --env_name 'humanoid' --render
"""
class BehavioralCloning:
    EPOCH_COUNT = 500
    BATCH_SIZE = 1024

    MODEL_SIZE = [256, 128, 128]

    def __init__(self, args):
        self.args = args

    def fit(self, train_X, train_Y):
        self.input_dim = train_X.shape[1]
        self.output_dim = train_Y.shape[1]

        self.model = Sequential()
        self.model.add(Dense(self.MODEL_SIZE[0], input_dim=self.input_dim, activation='relu'))
        
        for layer_size in self.MODEL_SIZE[1:]:
            self.model.add(Dense(layer_size, activation='relu'))
        
        self.model.add(Dense(self.output_dim, activation='linear'))
        self.model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])
        
        self.model.fit(train_X[:80000], train_Y[:80000], validation_data=(train_X[80001:], train_Y[80001:]), epochs=self.EPOCH_COUNT, batch_size=self.BATCH_SIZE)

        model_file_name = self.get_file_name()
        self.model.save(model_file_name)

        # print(self.model.get_weights())
        return self.predict

    def predict(self, X):
        predictions = self.model.predict(X)
        a,b = predictions.shape
        return np.reshape(predictions, (a,1,b))

    def get_file_name(self):
        return 'clone_models/' + self.args.env_name + "_" + "_".join(list(map(str, self.MODEL_SIZE))) + '.h5'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_file')
    parser.add_argument('--env_name')

    parser.add_argument('--train', action='store_true')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--create_dataset', action='store_true')
    parser.add_argument("--max_timesteps", type=int)
    parser.add_argument('--num_rollouts', type=int, default=20,
                        help='Number of expert roll outs')

    args = parser.parse_args()

    behavior_clone = BehavioralCloning(args)


    if args.create_dataset:
    	data = get_data(load_policy.load_policy('experts/' + ENV_NAME_TO_REAL_ENV_NAME[args.env_name] + '.pkl') , ENV_NAME_TO_REAL_ENV_NAME[args.env_name], args.render, args.max_timesteps, args.num_rollouts)
    	pkl.dump(data, open('data/'+ ENV_NAME_TO_REAL_ENV_NAME[args.env_name] + '_' + str(args.num_rollouts) + '.pkl', 'wb'))
    elif args.train:
        env_data = pkl.load(open(args.data_file, 'rb'))
        sample_size,_,output_dimension = env_data['actions'].shape
        env_data['actions'] = np.reshape(env_data['actions'], (sample_size, output_dimension))
        print(env_data['actions'].shape)

        policy_fn = behavior_clone.fit(env_data['observations'], env_data['actions'])
	else:
        test_model(behavior_clone, ENV_NAME_TO_REAL_ENV_NAME[args.env_name], args.render, args.max_timesteps, args.num_rollouts)

    import gc
    gc.collect()
