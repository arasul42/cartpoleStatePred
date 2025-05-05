#run in a terminal
docker compose up 

#in a seperate terminal

./run_gui_container.sh

#after the contaner starts

cd cartpole_dae

#to train state predictor model

python3 train.py

#to train RL with full state

python3 train_rl.py

#to train RL with state prediction

python3 train_dqn.py

#to evaluate full state RL performance

python3 rl_eva.py

#to evaluate state prediction RL performance 

python3 mpc_eva.py
