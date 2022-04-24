# Transition-Matching Permutation
Codebase modified from [GATA](https://github.com/xingdi-eric-yuan/GATA-public)

```
# Dependencies
pip install -r requirements.txt

# Download FastText Word Embeddings
curl -L -o crawl-300d-2M.vec.h5 "https://bit.ly/2U3Mde2"

# Download data for observation generation
cd obs_gen.0.1 ; wget https://aka.ms/twkg/obs_gen.0.1.zip ; unzip obs_gen.0.1.zip ; cd ..

# Download games
cd rl.0.2 ; wget https://aka.ms/twkg/rl.0.2.zip ; unzip rl.0.2.zip ; cd ..
```

## Initialization
```
python bake_vocab.py
python trajectory_matching.py
```

## Training Tr-DQN, Tr-DRQN, Tr-DQN-cat
```
python train_rl_with_continuous_belief configs/train_dqn_difficulty_1.yaml
python train_rl_with_continuous_belief configs/train_drqn_difficulty_1.yaml
python train_rl_with_continuous_belief configs/train_dqn_cat_difficulty_1.yaml
```

## Training CREST, CREST-cat
```
for i in 1 2 3 4
do
    python rollout.py configs/train_drqn_difficulty_${i}.yaml
done
python crest.py

python train_rl_with_continuous_belief configs/train_crest_difficulty_1.yaml
python train_rl_with_continuous_belief configs/train_crest_cat_difficulty_1.yaml
```

## Training with Transition-Matching Permutation
```
python train_rl_with_continuous_belief configs/train_dqn_traj_difficulty_1.yaml
```