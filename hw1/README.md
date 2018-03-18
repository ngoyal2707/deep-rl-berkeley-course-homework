# CS294-112 HW 1: Imitation Learning

*Behavioral Cloning*


python3 behavioral_cloning.py --create_dataset --env_name 'humanoid' --num_rollouts 50
python3 behavioral_cloning.py --train --data_file data/Humanoid-v1_50.pkl --env_name humanoid
python3 behavioral_cloning.py --env_name 'humanoid' --render
