import json
from pathlib import Path
from typing import List

import gym
import textworld.gym

import generic
from agent import Agent
from reinforcement_learning_dataset import _wrap_env


def get_env(data_dir, difficulty_level, training_size, requested_infos,
            max_episode_steps, batch_size, config):
    # training games
    game_file_names: List[Path] = []
    game_path = (Path(data_dir) / f"train_{training_size}" /
                 f"difficulty_level_{difficulty_level}")
    if game_path.is_dir():
        game_file_names = list(game_path.glob('*.z8'))
    else:
        game_file_names.append(game_path)
    game_file_names.sort()

    env_config = config['rl'].get('eval_env') or config['rl']['env']
    env_id = textworld.gym.register_games(list(map(str, game_file_names)),
                                            request_infos=requested_infos,
                                            max_episode_steps=None,
                                            batch_size=batch_size,
                                            name="training",
                                            asynchronous=False,
                                            auto_reset=False)
    env = gym.make(env_id)

    env = _wrap_env(env, env_config)
    num_games = len(game_file_names)
    return env, num_games


def load_env(config, agent):
    # make game environments
    requested_infos = agent.select_additional_infos_lite()
    games_dir = "./"

    # training game env
    env, num_games = get_env(games_dir + config['rl']['data_path'],
                             config['rl']['difficulty_level'],
                             config['rl']['training_size'], requested_infos,
                             agent.eval_max_nb_steps_per_episode, agent.batch_size,
                             config)
    return env, num_games


def collect():
    config = generic.load_config()
    agent = Agent(config)
    output_dir = f"checkpoints/{agent.experiment_tag}"
    env, num_games = load_env(config, agent)

    agent.load_pretrained_model(agent.load_from_tag + ".pt",
                                load_partial_graph=False)
    agent.eval()
    agent.init()

    results = {}
    game_id = 0
    while game_id < num_games:
        obs, infos = env.reset()
        batch_size = len(obs)

        game_name_list = [
            game.metadata["uuid"].split("-")[-1] for game in infos["game"]
        ]
        actions = [[] for _ in range(batch_size)]

        was_done = [False] * batch_size
        prev_h, prev_c = None, None
        for _ in range(agent.eval_max_nb_steps_per_episode):
            _, chosen_indices, prev_h, prev_c = agent.act_greedy(
                obs,
                infos['admissible_commands'],
                previous_h=prev_h,
                previous_c=prev_c)
            chosen_actions_before_parsing = [
                item[idx] for item, idx in zip(infos["admissible_commands"],
                                               chosen_indices)
            ]
            obs, _, dones, infos = env.step(chosen_actions_before_parsing)
            for i, a in enumerate(chosen_actions_before_parsing):
                if not was_done[i]:
                    actions[i].append(' '.join(a))

            if all(was_done):
                break
            was_done = dones
        game_id += batch_size
        for game, _actions in zip(game_name_list, actions):
            results[game] = _actions

    with open(f'{output_dir}/trajectory-train.json', 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == '__main__':
    collect()
