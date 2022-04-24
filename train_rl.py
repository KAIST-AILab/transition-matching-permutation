import datetime
import json
import os

import numpy as np
from tensorboardX import SummaryWriter
from tqdm.auto import tqdm

import evaluate
import generic
import reinforcement_learning_dataset
from agent import Agent
from generic import HistoryScoreCache


def load_agent(agent, save_path, data_dir):
    if agent.load_pretrained:
        if os.path.exists(save_path):
            # this experiment itself (in case the experiment crashes for unknown reasons on server)
            agent.load_pretrained_model(save_path, load_partial_graph=False)
            agent.update_target_net()
        elif os.path.exists(data_dir + "/" + agent.load_from_tag + ".pt"):
            # load from pre-trained graph encoder
            agent.load_pretrained_model(data_dir + "/" + agent.load_from_tag +
                                        ".pt",
                                        load_partial_graph=False)
            agent.update_target_net()
        else:
            raise ValueError('No checkpoint to load!')


def load_env(config, agent):
    # make game environments
    if "ground-truth" in config["rl"]["env"]["summary_model"]:
        requested_infos = agent.select_additional_infos()
    else:
        requested_infos = agent.select_additional_infos_lite()
    requested_infos_eval = agent.select_additional_infos()
    games_dir = "./"

    # training game env
    env, _ = reinforcement_learning_dataset.get_training_game_env(
        games_dir + config['rl']['data_path'],
        config['rl']['difficulty_level'], config['rl']['training_size'],
        requested_infos, agent.max_nb_steps_per_episode, agent.batch_size,
        config)

    if agent.run_eval:
        # training game env
        eval_env, num_eval_game = reinforcement_learning_dataset.get_evaluation_game_env(
            games_dir + config['rl']['data_path'],
            config['rl']['difficulty_level'],
            requested_infos_eval,
            agent.eval_max_nb_steps_per_episode,
            agent.eval_batch_size,
            config,
            valid_or_test="valid")
    else:
        eval_env, num_eval_game = None, None
    return env, eval_env, num_eval_game


def run_step(agent, obs, infos, hidden_states, act_randomly):
    action_candidates = infos['admissible_commands']
    prev_h, prev_c = None, None
    if hidden_states:
        prev_h, prev_c = hidden_states
    chosen_actions, chosen_indices, prev_h, prev_c = agent.act(
        obs,
        action_candidates,
        previous_h=prev_h,
        previous_c=prev_c,
        random=act_randomly)
    hidden_states = prev_h, prev_c

    transition = {
        'observation': obs,
        'action_candidates': action_candidates,
        'chosen_action_index': chosen_indices,
        'chosen_action': chosen_actions,
    }

    raw_chosen_actions = [
        item[idx] for item, idx in zip(action_candidates, chosen_indices)
    ]
    return raw_chosen_actions, transition, hidden_states


class TransitionCache:
    def __init__(self):
        self.data = []

    def reset(self, data=None):
        self.data = data

    def append(self, batch_record):
        # [B, T, {K: []}], {K: [B]} --> [B, T+1, {K: []}]
        if not self.data:
            some_batch = list(batch_record.values())[0]
            self.data = [[] for _ in range(len(some_batch))]

        for i, episode_box in enumerate(self.data):
            strand = {k: v[i] for k, v in batch_record.items()}
            episode_box.append(strand)

    @property
    def shape(self):
        return (len(self.data), len(self.data[0]))

    def __getitem__(self, index):
        return self.data[index]


def push_into_replay_buffer(agent, transition_cache):
    _, episode_len = transition_cache.shape
    full_episode = episode_len == agent.max_nb_steps_per_episode

    avg_rewards_in_buffer = agent.dqn_memory.avg_rewards()
    reward_threshold = avg_rewards_in_buffer * agent.buffer_reward_threshold

    for episode in transition_cache:
        # need to pad one transition
        is_final = episode[-1]['finished']
        _need_pad = full_episode and not is_final

        rewards = [record['reward'] for record in episode]
        if _need_pad:
            rewards += [0.0]
        assert len(rewards) > 0, 'Zero length episode!'
        if np.mean(rewards) < reward_threshold:
            continue

        for record in episode:
            agent.dqn_memory.add(record['observation'],
                                 record['chosen_action'],
                                 record['action_candidates'],
                                 record['chosen_action_index'],
                                 record['reward'], record['finished'])
            if record['finished']:
                break
        if _need_pad:
            record = episode[-1]
            agent.dqn_memory.add(record['observation'],
                                 record['chosen_action'],
                                 record['action_candidates'],
                                 record['chosen_action_index'],
                                 record['reward'] * 0.0, True)


def should_save(curr_eval_performance, best_eval_performance_so_far,
                curr_train_performance, best_train_performance_so_far):
    if curr_eval_performance > best_eval_performance_so_far:
        return True
    if curr_eval_performance < best_eval_performance_so_far:
        return False
    if curr_eval_performance > 0.0:
        return True
    return curr_train_performance >= best_train_performance_so_far


class Patience:
    def __init__(self, agent, save_path):
        self.agent = agent
        self.save_path = save_path
        self.limit = agent.patience
        self.gauge = 0
        self.prev_performance = 0.0

    def reset(self):
        self.gauge = 0

    def reload(self):
        self.gauge += 1
        if self.gauge < self.limit:
            return

        if os.path.exists(self.save_path):
            print('reload from a good checkpoint...')
            self.agent.load_pretrained_model(self.save_path,
                                             load_partial_graph=False)
            self.agent.update_target_net()
            self.reset()

    def step(self, curr_performance):
        if not self.limit:
            return

        if self.prev_performance <= curr_performance:
            self.reset()
        else:
            self.reload()
        self.prev_performance = curr_performance


def train():
    time_1 = datetime.datetime.now()
    config = generic.load_config()
    agent = Agent(config)

    output_dir = "."
    data_dir = "."
    output_dir = os.path.join("checkpoints", agent.experiment_tag)
    save_path = os.path.join(output_dir, "model.pt")
    log_path = os.path.join(output_dir, 'log.json')
    os.makedirs(output_dir, exist_ok=True)

    # load model from checkpoint
    load_agent(agent, save_path, data_dir)
    env, eval_env, num_eval_game = load_env(config, agent)

    if config["general"]["tensorboard"]:
        writer = SummaryWriter(output_dir)

    step_in_total = 0
    episode_no = config['general'].get('starting_episode', 0)
    running_avg = {
        'returns': HistoryScoreCache(capacity=500),
        'game_points': HistoryScoreCache(capacity=500),
        'game_points_normalized': HistoryScoreCache(capacity=500),
        'game_steps': HistoryScoreCache(capacity=500),
        'loss': HistoryScoreCache(capacity=500),
        'Q': HistoryScoreCache(capacity=500),
        'seen_recipe': HistoryScoreCache(capacity=500)
    }

    step_penalty = config['rl']['training'].get('step_penalty', 0.0)
    step_penalty /= agent.max_nb_steps_per_episode

    best_train_performance_so_far = 0.0
    best_eval_performance_so_far = 0.0

    patience = Patience(agent, save_path)
    perfect_training = 0
    pbar = tqdm(total=agent.max_episode - episode_no)
    while episode_no <= agent.max_episode:
        np.random.seed(episode_no)
        env.seed(episode_no)
        obs, infos = env.reset()
        batch_size = len(obs)

        agent.train()
        agent.init()

        act_randomly = (not agent.noisy_net
                        and episode_no < agent.learn_start_from_this_episode)

        game_max_scores = [game.max_score for game in infos["game"]]
        game_names = [
            game.metadata["uuid"].split("-")[-1] for game in infos["game"]
        ]

        seen_recipe = [False] * batch_size
        was_done = [False] * batch_size
        prev_scores = [0.0] * batch_size
        returns = [0.0] * batch_size
        ep_lengths = [0] * batch_size
        print_actions = []

        transition_cache = TransitionCache()

        hidden_states = None
        for step_no in range(agent.max_nb_steps_per_episode):
            if agent.noisy_net:
                agent.reset_noise()  # Draw a new set of noisy weights

            (chosen_actions_before_parsing, transition,
             hidden_states) = run_step(agent, obs, infos, hidden_states,
                                       act_randomly)
            obs, scores, dones, infos = env.step(chosen_actions_before_parsing)

            # terminate the game because DQN requires one extra step
            if step_no == agent.max_nb_steps_per_episode - 1:
                dones = [True] * batch_size
            seen_recipe = infos['extra.seen_recipe']

            step_in_total += 1
            penalty = [-step_penalty] * batch_size
            if agent.use_negative_reward:
                penalty = [r - l for r, l in zip(penalty, infos["lost"])]
            rewards = [
                a - b + p for a, b, p in zip(scores, prev_scores, penalty)
            ]
            rewards = [0 if d else r for r, d in zip(rewards, was_done)]
            for i, r in enumerate(rewards):
                returns[i] += r
            prev_scores = scores

            transition['reward'] = rewards
            transition['finished'] = was_done
            transition_cache.append(transition)

            for i, done in enumerate(was_done):
                ep_lengths[i] += not done

            print_actions.append("--" if was_done[0] else " ".
                                 join(chosen_actions_before_parsing[0]))

            # DQN update
            if (episode_no >= agent.learn_start_from_this_episode
                    and step_in_total % agent.update_per_k_game_steps == 0):
                if agent.noisy_net:
                    agent.reset_noise()
                for _ in range(
                        config['rl']['training']['update_n_iterations']):
                    loss, Q = agent.update_dqn(episode_no)
                if loss is not None:
                    running_avg['loss'].push(loss)
                    running_avg['Q'].push(Q)

            if all(was_done):
                break
            was_done = dones

        for b in range(batch_size):
            running_avg['returns'].push(returns[b])
            running_avg['game_points'].push(prev_scores[b])
            running_avg['game_points_normalized'].push(prev_scores[b] /
                                                       game_max_scores[b])
            running_avg['game_steps'].push(ep_lengths[b])
            running_avg['seen_recipe'].push(seen_recipe[b])
        current_avg = {k: v.get_avg() for k, v in running_avg.items()}

        # Push into replay buffer
        push_into_replay_buffer(agent, transition_cache)

        # finish game
        agent.finish_of_episode(episode_no, batch_size)
        episode_no += batch_size
        pbar.update(batch_size)

        if episode_no < agent.learn_start_from_this_episode:
            continue
        if agent.report_frequency == 0 or (
                episode_no % agent.report_frequency >
            (episode_no - batch_size) % agent.report_frequency):
            continue
        time_2 = datetime.datetime.now()
        time_spent = str(time_2 - time_1).rsplit(".")[0]
        print(f"Episode: {episode_no:3d} | time spent: {time_spent} | " +
              ' | '.join(f"{k}: {v:2.3f}" for k, v in current_avg.items()))
        print(game_names[0] + ":    " + " | ".join(print_actions))

        # evaluate
        curr_train_performance = current_avg['game_points_normalized']
        if agent.run_eval:
            eval_stats, detailed_scores = evaluate.evaluate(
                eval_env, agent, num_eval_game)
            curr_eval_performance = eval_stats['game_points_normalized']
            curr_performance = curr_eval_performance
        else:
            eval_stats = {}
            detailed_scores = ""
            curr_eval_performance = 0.0
            curr_performance = curr_train_performance

        _should_save = should_save(curr_eval_performance,
                                   best_eval_performance_so_far,
                                   curr_train_performance,
                                   best_train_performance_so_far)
        if _should_save:
            agent.save_model_to_path(save_path)

        if curr_train_performance >= best_train_performance_so_far:
            best_train_performance_so_far = curr_train_performance
        if curr_eval_performance >= best_eval_performance_so_far:
            best_eval_performance_so_far = curr_eval_performance

        patience.step(curr_performance)

        if current_avg['game_points_normalized'] >= 0.95:
            perfect_training += 1
        else:
            perfect_training = 0

        # plot using tensorboard
        if config["general"]["tensorboard"]:
            for k, v in current_avg.items():
                writer.add_scalar('train/' + k, v, episode_no)
            lr = agent.scheduler.get_last_lr()[1]
            writer.add_scalar('train/learning_rate', lr, episode_no)
            writer.add_scalar('train/epsilon', agent.epsilon, episode_no)
            for k, v in eval_stats.items():
                writer.add_scalar('eval/' + k, v, episode_no)

        # write accuracies down into file
        report = {k: str(v) for k, v in current_avg.items()}
        report["time spent"] = time_spent
        for k, v in eval_stats.items():
            report['eval_' + k] = str(v)
        report["detailed scores"] = detailed_scores
        with open(log_path, 'a+') as f:
            f.write(json.dumps(report) + '\n')
            f.flush()

        if curr_performance == 1.0 and curr_train_performance >= 0.95:
            break
        if perfect_training >= 3:
            break
    pbar.close()
    if config['general']['tensorboard']:
        writer.close()


if __name__ == '__main__':
    train()
