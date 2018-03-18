# CS294-112 HW 1: Imitation Learning

## Behavioral Cloning

```
#Create Dataset
python3 behavioral_cloning.py --create_dataset --env_name 'humanoid' --num_rollouts 50

#Train
python3 behavioral_cloning.py --train --data_file data/Humanoid-v1_50.pkl --env_name humanoid

#Test
python3 behavioral_cloning.py --env_name 'humanoid' --render
```


## Dagger

```
python3 dagger.py --train --env_name humanoid --render --init_rollouts 5 --iterations 40 --per_iteration_rollout 4
python3 dagger.py --train --env_name ant --render --init_rollouts 5 --iterations 20 --per_iteration_rollout 4
python3 dagger.py --train --env_name halfcheetah --render --init_rollouts 5 --iterations 20 --per_iteration_rollout 4
python3 dagger.py --train --env_name reacher --render --init_rollouts 10 --iterations 30 --per_iteration_rollout 4
python3 dagger.py --train --env_name walker --render --init_rollouts 5 --iterations 20 --per_iteration_rollout 4
python3 dagger.py --train --env_name hopper --render --init_rollouts 2 --iterations 20 --per_iteration_rollout 4
```
