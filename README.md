# Transition-Matching Permutation (GATA version)
Codebase modified from [GATA](https://github.com/xingdi-eric-yuan/GATA-public)

## Initialization
```
# Dependencies
pip install -r requirements.txt

# Download FastText Word Embeddings
curl -L -o crawl-300d-2M.vec.h5 "https://bit.ly/2U3Mde2"

# Download data for observation generation
cd obs_gen.0.1 ; wget https://aka.ms/twkg/obs_gen.0.1.zip ; unzip obs_gen.0.1.zip ; cd ..

# Download games
cd rl.0.2 ; wget https://aka.ms/twkg/rl.0.2.zip ; unzip rl.0.2.zip ; cd ..

# Vocabulary-related files
python bake_vocab.py

# Prepare Transition-Matching Permutation
python transition_matching.py
```

## Pre-training Graph Updater
```
python train_obs_generation.py configs/pretrain_observation_generation.yaml
python train_obs_generation.py configs/pretrain_contrastive_observation_classification.yaml
```

### Pre-Training Graph Updater with Transition-Matching Permutation
```
python train_obs_generation.py configs/pretrain_observation_generation_permute.yaml
python train_obs_generation.py configs/pretrain_contrastive_observation_classification_permute.yaml
```

## Training GATA
```
python train_rl_with_continuous_belief configs/train_og_difficulty_1.yaml
python train_rl_with_continuous_belief configs/train_coc_difficulty_1.yaml
```

### Training GATA with Transition-Matching Permutation
```
python train_rl_with_continuous_belief configs/train_og_permute_difficulty_1.yaml
python train_rl_with_continuous_belief configs/train_coc_permute_difficulty_1.yaml
```
