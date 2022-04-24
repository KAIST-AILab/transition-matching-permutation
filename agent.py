from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from textworld import EnvInfos

import dqn_memory_priortized_replay_buffer
from generic import (LinearSchedule, _words_to_ids, ez_gather_dim_1, max_len,
                     pad_sequences, to_np, to_pt)
from model import BaseModel


def get_optimizer(step_rule, config, param_frozen_list, param_active_list):
    lr = config['general']['training']['optimizer']['learning_rate']
    wd = config['general']['training']['optimizer'].get('weight_decay', 0.0)
    params = [{
        'params': param_frozen_list,
        'lr': 0.0,
        'weight_decay': 0.0,
    }, {
        'params': param_active_list
    }]
    if step_rule == 'adam':
        return torch.optim.Adam(params, lr=lr, weight_decay=wd)
    elif step_rule == 'adamw':
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    elif step_rule == 'radam':
        from radam import RAdam
        return RAdam(params, lr=lr, weight_decay=wd)
    else:
        raise NotImplementedError


def get_scheduler(optimizer, warmup_steps=5000, total_steps=0, decay='none'):
    def lr(steps):
        ratio1 = steps / warmup_steps if warmup_steps > 0 else 1.0
        if decay == 'linear' and total_steps > warmup_steps:
            ratio2 = (steps - total_steps) / (warmup_steps - total_steps)
        else:
            ratio2 = 1.0
        return max(1e-6, min(ratio1, ratio2))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr)


class BaseAgent:
    def __init__(self, config):
        self.mode = "train"
        self.config = config
        self.load_config()

    def load_config(self):
        self.task = self.config['general']['task']

        self.step_rule = self.config['general']['training']['optimizer'][
            'step_rule']
        self.init_learning_rate = self.config['general']['training'][
            'optimizer']['learning_rate']
        self.clip_grad_norm = self.config['general']['training']['optimizer'][
            'clip_grad_norm']
        self.learning_rate_warmup_until = self.config['general']['training'][
            'optimizer']['learning_rate_warmup_until']
        self.learning_rate_decay = self.config['general']['training'][
            'optimizer'].get('learning_rate_decay', 'none')
        self.fix_parameters_keywords = list(
            set(self.config['general']['training']['fix_parameters_keywords']))
        self.batch_size = self.config['general']['training']['batch_size']
        self.max_episode = self.config['general']['training']['max_episode']
        self.smoothing_eps = self.config['general']['training'][
            'smoothing_eps']
        self.patience = self.config['general']['training']['patience']

        self.run_eval = self.config['general']['evaluate']['run_eval']
        self.eval_batch_size = self.config['general']['evaluate']['batch_size']

        # Set the random seed manually for reproducibility.
        self.random_seed = self.config['general']['random_seed']
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        if torch.cuda.is_available():
            if not self.config['general']['use_cuda']:
                print(
                    "WARNING: CUDA device detected but 'use_cuda: false' found in config.yaml"
                )
                self.use_cuda = False
            else:
                torch.backends.cudnn.deterministic = True
                torch.cuda.manual_seed(self.random_seed)
                self.use_cuda = True
        else:
            self.use_cuda = False

        self.use_fp16 = self.config['general'].get('use_fp16', False)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_fp16)

        self.experiment_tag = self.config['general']['checkpoint'][
            'experiment_tag']
        self.save_frequency = self.config['general']['checkpoint'][
            'save_frequency']
        self.report_frequency = self.config['general']['checkpoint'][
            'report_frequency']
        self.load_pretrained = self.config['general']['checkpoint'][
            'load_pretrained']
        self.load_from_tag = self.config['general']['checkpoint'][
            'load_from_tag']
        self.load_parameter_keywords = list(
            set(self.config['general']['checkpoint']
                ['load_parameter_keywords']))

    def train(self):
        """
        Tell the agent that it's training phase.
        """
        self.mode = "train"
        self.online_net.train()

    def eval(self):
        """
        Tell the agent that it's evaluation phase.
        """
        self.mode = "eval"
        self.online_net.eval()

    def save_model_to_path(self, save_to):
        torch.save(self.online_net.state_dict(), save_to)
        print("Saved checkpoint to %s..." % (save_to))


class Agent(BaseAgent):
    def __init__(self, config, given_net=None):
        self.given_net = given_net
        super().__init__(config)
        self.load_online_net()
        self.load_target_net()
        param_frozen_list, param_active_list = self._exclude_parameters()
        self.optimizer = get_optimizer(self.step_rule, self.config,
                                       param_frozen_list, param_active_list)
        self.scheduler = self.get_scheduler()

    def get_scheduler(self):
        warmup = self.learning_rate_warmup_until / self.batch_size
        total = self.max_episode / self.batch_size
        offset = self.learn_start_from_this_episode / self.batch_size
        return get_scheduler(self.optimizer,
                             warmup - offset,
                             total - offset,
                             decay=self.learning_rate_decay)

    def load_online_net(self):
        if self.given_net is not None:
            self.online_net = self.given_net
            return

        self.online_net = BaseModel(config=self.config,
                                    word_vocab=self.word_vocab)
        self.online_net.train()
        if self.use_cuda:
            self.online_net.cuda()

    def load_target_net(self):
        if self.task == "rl":
            self.target_net = BaseModel(config=self.config,
                                        word_vocab=self.word_vocab,
                                        use_pretrained=False)
            self.target_net.train()
            self.update_target_net()
            for param in self.target_net.parameters():
                param.requires_grad = False
            if self.use_cuda:
                self.target_net.cuda()
        else:
            self.target_net = None

    def load_pretrained_model(self, load_from, load_partial_graph=True):
        """
        Load pretrained checkpoint from file.

        Arguments:
            load_from: File name of the pretrained model checkpoint.
        """
        print("loading model from %s\n" % (load_from))
        try:
            if self.use_cuda:
                pretrained_dict = torch.load(load_from)
            else:
                pretrained_dict = torch.load(load_from, map_location='cpu')

            model_dict = self.online_net.state_dict()
            pretrained_dict = {
                k: v
                for k, v in pretrained_dict.items() if k in model_dict
            }
            if load_partial_graph and len(self.load_parameter_keywords) > 0:
                tmp_pretrained_dict = {}
                for k, v in pretrained_dict.items():
                    for keyword in self.load_parameter_keywords:
                        if keyword in k:
                            tmp_pretrained_dict[k] = v
                            break
                pretrained_dict = tmp_pretrained_dict
            model_dict.update(pretrained_dict)
            self.online_net.load_state_dict(model_dict)
            print("The loaded parameters are:")
            keys = [key for key in pretrained_dict]
            print(", ".join(keys))
            print("--------------------------")
        except:
            print("Failed to load checkpoint...")

    def _exclude_parameters(self):
        param_frozen_list = []
        param_active_list = []
        for k, v in self.online_net.named_parameters():
            keep_this = True
            for keyword in self.fix_parameters_keywords:
                if keyword in k:
                    param_frozen_list.append(v)
                    keep_this = False
                    break
            if keep_this:
                param_active_list.append(v)

        param_frozen_list = torch.nn.ParameterList(param_frozen_list)
        param_active_list = torch.nn.ParameterList(param_active_list)
        return param_frozen_list, param_active_list

    def load_config(self):
        super().load_config()
        # word vocab
        vocab_path = self.config['general']['model'].get(
            'vocab_path', "./vocabularies/word_vocab.txt")
        with open(vocab_path) as f:
            self.word_vocab = [line.strip() for line in f]
        self.word2id = {}
        for i, w in enumerate(self.word_vocab):
            self.word2id[w] = i

        self.max_target_length = self.config['general']['evaluate'][
            'max_target_length']

        self.backprop_frequency = self.config['obs_gen']['backprop_frequency']

        # RL specific
        # epsilon greedy
        self.epsilon_anneal_episodes = self.config['rl']['epsilon_greedy'][
            'epsilon_anneal_episodes']
        self.epsilon_anneal_from = self.config['rl']['epsilon_greedy'][
            'epsilon_anneal_from']
        self.epsilon_anneal_to = self.config['rl']['epsilon_greedy'][
            'epsilon_anneal_to']
        self.epsilon = self.epsilon_anneal_from
        self.epsilon_scheduler = LinearSchedule(
            schedule_timesteps=self.epsilon_anneal_episodes,
            initial_p=self.epsilon_anneal_from,
            final_p=self.epsilon_anneal_to)
        self.noisy_net = self.config['rl']['epsilon_greedy']['noisy_net']
        if self.noisy_net:
            # disable epsilon greedy
            self.epsilon_anneal_episodes = -1
            self.epsilon = 0.0
        # drqn
        self.replay_sample_history_length = self.config['rl']['replay'][
            'replay_sample_history_length']
        self.replay_sample_update_from = self.config['rl']['replay'][
            'replay_sample_update_from']
        # replay buffer and updates
        self.buffer_reward_threshold = self.config['rl']['replay'][
            'buffer_reward_threshold']
        self.prioritized_replay_beta = self.config['rl']['replay'][
            'prioritized_replay_beta']
        self.beta_scheduler = LinearSchedule(
            schedule_timesteps=self.max_episode,
            initial_p=self.prioritized_replay_beta,
            final_p=1.0)

        self.accumulate_reward_from_final = self.config['rl']['replay'][
            'accumulate_reward_from_final']
        self.prioritized_replay_eps = self.config['rl']['replay'][
            'prioritized_replay_eps']
        self.discount_gamma_game_reward = self.config['rl']['replay'][
            'discount_gamma_game_reward']
        self.replay_batch_size = self.config['rl']['replay'][
            'replay_batch_size']
        self.dqn_memory = dqn_memory_priortized_replay_buffer.PrioritizedReplayMemory(
            self.config['rl']['replay']['replay_memory_capacity'],
            priority_fraction=self.config['rl']['replay']
            ['replay_memory_priority_fraction'],
            discount_gamma_game_reward=self.discount_gamma_game_reward,
            accumulate_reward_from_final=self.accumulate_reward_from_final)
        self.update_per_k_game_steps = self.config['rl']['replay'][
            'update_per_k_game_steps']
        self.multi_step = self.config['rl']['replay']['multi_step']
        # input in rl training
        self.enable_recurrent_memory = self.config['rl']['model'][
            'enable_recurrent_memory']
        # rl train and eval
        self.max_nb_steps_per_episode = self.config['rl']['training'][
            'max_nb_steps_per_episode']
        self.learn_start_from_this_episode = self.config['rl']['training'][
            'learn_start_from_this_episode']
        self.target_net_update_frequency = self.config['rl']['training'][
            'target_net_update_frequency']
        self.use_negative_reward = self.config['rl']['training'][
            'use_negative_reward']
        self.eval_max_nb_steps_per_episode = self.config['rl']['evaluate'][
            'max_nb_steps_per_episode']

    def update_target_net(self):
        if self.target_net is not None:
            self.target_net.load_state_dict(self.online_net.state_dict())

    def select_additional_infos(self):
        """
        Returns what additional information should be made available at each game step.

        Requested information will be included within the `infos` dictionary
        passed to `CustomAgent.act()`. To request specific information, create a
        :py:class:`textworld.EnvInfos <textworld.envs.wrappers.filter.EnvInfos>`
        and set the appropriate attributes to `True`. The possible choices are:

        * `description`: text description of the current room, i.e. output of the `look` command;
        * `inventory`: text listing of the player's inventory, i.e. output of the `inventory` command;
        * `max_score`: maximum reachable score of the game;
        * `objective`: objective of the game described in text;
        * `entities`: names of all entities in the game;
        * `verbs`: verbs understood by the the game;
        * `command_templates`: templates for commands understood by the the game;
        * `admissible_commands`: all commands relevant to the current state;

        In addition to the standard information, game specific information
        can be requested by appending corresponding strings to the `extras`
        attribute. For this competition, the possible extras are:

        * `'recipe'`: description of the cookbook;
        * `'walkthrough'`: one possible solution to the game (not guaranteed to be optimal);

        Example:
            Here is an example of how to request information and retrieve it.

            >>> from textworld import EnvInfos
            >>> request_infos = EnvInfos(description=True, inventory=True, extras=["recipe"])
            ...
            >>> env = gym.make(env_id)
            >>> ob, infos = env.reset()
            >>> print(infos["description"])
            >>> print(infos["inventory"])
            >>> print(infos["extra.recipe"])

        Notes:
            The following information *won't* be available at test time:

            * 'walkthrough'
        """
        request_infos = EnvInfos()
        request_infos.admissible_commands = True
        request_infos.description = True
        request_infos.location = True
        request_infos.facts = True
        request_infos.last_action = True
        request_infos.game = True
        if self.use_negative_reward:
            request_infos.lost = True
            request_infos.won = True
        return request_infos

    def select_additional_infos_lite(self):
        """
        Returns what additional information should be made available at each game step.

        Requested information will be included within the `infos` dictionary
        passed to `CustomAgent.act()`. To request specific information, create a
        :py:class:`textworld.EnvInfos <textworld.envs.wrappers.filter.EnvInfos>`
        and set the appropriate attributes to `True`. The possible choices are:

        * `description`: text description of the current room, i.e. output of the `look` command;
        * `inventory`: text listing of the player's inventory, i.e. output of the `inventory` command;
        * `max_score`: maximum reachable score of the game;
        * `objective`: objective of the game described in text;
        * `entities`: names of all entities in the game;
        * `verbs`: verbs understood by the the game;
        * `command_templates`: templates for commands understood by the the game;
        * `admissible_commands`: all commands relevant to the current state;

        In addition to the standard information, game specific information
        can be requested by appending corresponding strings to the `extras`
        attribute. For this competition, the possible extras are:

        * `'recipe'`: description of the cookbook;
        * `'walkthrough'`: one possible solution to the game (not guaranteed to be optimal);

        Example:
            Here is an example of how to request information and retrieve it.

            >>> from textworld import EnvInfos
            >>> request_infos = EnvInfos(description=True, inventory=True, extras=["recipe"])
            ...
            >>> env = gym.make(env_id)
            >>> ob, infos = env.reset()
            >>> print(infos["description"])
            >>> print(infos["inventory"])
            >>> print(infos["extra.recipe"])

        Notes:
            The following information *won't* be available at test time:

            * 'walkthrough'
        """
        request_infos = EnvInfos()
        request_infos.admissible_commands = True
        request_infos.description = False
        request_infos.location = False
        request_infos.facts = False
        request_infos.last_action = False
        request_infos.game = True
        if self.use_negative_reward:
            request_infos.lost = True
            request_infos.won = True
        return request_infos

    def init(self):
        pass

    def reset_noise(self):
        if self.noisy_net:
            # Resets noisy weights in all linear layers (of online net only)
            self.online_net.reset_noise()

    def zero_noise(self):
        if self.noisy_net:
            self.online_net.zero_noise()
            if self.target_net is not None:
                self.target_net.zero_noise()

    def get_word_input(self, input_strings: List[List[str]]):
        word_id_list = [
            _words_to_ids(tokens, self.word2id) for tokens in input_strings
        ]
        input_word = pad_sequences(
            word_id_list, maxlen=max_len(word_id_list)).astype('int32')
        input_word = to_pt(input_word, self.use_cuda)
        return input_word

    def get_action_candidate_list_input(
            self, action_candidate_list: List[List[List[str]]]):
        # action_candidate_list (list): batch x num_candidate of strings
        batch_size = len(action_candidate_list)
        max_num_candidate = max_len(action_candidate_list)
        input_action_candidate_list = []
        for i in range(batch_size):
            word_level = self.get_word_input(action_candidate_list[i])
            input_action_candidate_list.append(word_level)
        max_word_num = max(
            [item.size(1) for item in input_action_candidate_list])

        input_action_candidate = np.zeros(
            (batch_size, max_num_candidate, max_word_num))
        input_action_candidate = to_pt(input_action_candidate,
                                       self.use_cuda,
                                       dtype="long")
        for i in range(batch_size):
            input_action_candidate[i, :input_action_candidate_list[i].
                                   size(0), :input_action_candidate_list[i].
                                   size(1)] = input_action_candidate_list[i]

        return input_action_candidate

    def choose_model(self, use_model="online"):
        if self.task != "rl":
            return self.online_net
        if use_model == "online":
            model = self.online_net
        elif use_model == "target":
            model = self.target_net
        else:
            raise NotImplementedError
        return model

    def encode_text(self, observation_strings: List[List[str]], use_model):
        model = self.choose_model(use_model)
        input_obs = self.get_word_input(observation_strings)
        # encode
        obs_encoding_sequence, obs_mask = model.encode_text(input_obs)
        return obs_encoding_sequence, obs_mask

    ##################################
    # RL specific
    ##################################

    def finish_of_episode(self, episode_no, batch_size):
        # Update target network
        if (
                episode_no + batch_size
        ) % self.target_net_update_frequency <= episode_no % self.target_net_update_frequency:
            self.update_target_net()
        # decay lambdas
        if episode_no < self.learn_start_from_this_episode:
            return
        if episode_no < self.epsilon_anneal_episodes + self.learn_start_from_this_episode:
            self.epsilon = self.epsilon_scheduler.value(
                episode_no - self.learn_start_from_this_episode)
            self.epsilon = min(max(self.epsilon, 0.0), 1.0)
        if self.scheduler:
            self.scheduler.step()

    def forward(self,
                observation_strings: List[List[str]],
                action_candidate_list: List[List[List[str]]],
                previous_h: torch.Tensor,
                previous_c: torch.Tensor,
                use_model="online"):
        assert self.task == "rl"
        obs_encoding_sequence, obs_mask = self.encode_text(observation_strings,
                                                           use_model=use_model)

        model = self.choose_model(use_model)
        input_action_candidate = self.get_action_candidate_list_input(
            action_candidate_list)
        action_scores, action_masks, new_h, new_c = model.score_actions(
            input_action_candidate, obs_encoding_sequence, obs_mask, None,
            None, previous_h, previous_c)
        # batch x num_actions
        return action_scores, action_masks, new_h, new_c

    # action scoring stuff (Deep Q-Learning)
    def choose_random_action(self, action_rank, action_unpadded=None):
        """
        Select an action randomly.
        """
        batch_size = action_rank.size(0)
        action_space_size = action_rank.size(1)
        if action_unpadded is None:
            indices = np.random.choice(action_space_size, batch_size)
        else:
            indices = []
            for j in range(batch_size):
                indices.append(np.random.choice(len(action_unpadded[j])))
            indices = np.array(indices)
        return indices

    def choose_maxQ_action(self, action_rank, action_mask=None):
        """
        Generate an action by maximum q values.
        """
        action_rank = action_rank - torch.min(
            action_rank, -1, keepdim=True
        )[0] + 1e-2  # minus the min value, so that all values are non-negative
        if action_mask is not None:
            assert action_mask.size() == action_rank.size(), (
                action_mask.size().shape, action_rank.size())
            action_rank = action_rank * action_mask
        action_indices = torch.argmax(action_rank, -1)  # batch
        return to_np(action_indices)

    def act_greedy(self,
                   observation_strings: List[List[str]],
                   action_candidate_list: List[List[List[str]]],
                   previous_h=None,
                   previous_c=None):
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=self.use_fp16):
            action_scores, action_masks, new_h, new_c = self.forward(
                observation_strings,
                action_candidate_list,
                previous_h,
                previous_c,
                use_model="online")
            action_indices_maxq = self.choose_maxQ_action(
                action_scores, action_masks)
            chosen_indices = action_indices_maxq
            chosen_indices = chosen_indices.astype(int)
            chosen_actions = [
                item[idx]
                for item, idx in zip(action_candidate_list, chosen_indices)
            ]

            return chosen_actions, chosen_indices, new_h, new_c

    def act_random(self,
                   observation_strings: List[List[str]],
                   action_candidate_list: List[List[List[str]]],
                   previous_h=None,
                   previous_c=None):
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=self.use_fp16):
            action_scores, _, new_h, new_c = self.forward(
                observation_strings,
                action_candidate_list,
                previous_h,
                previous_c,
                use_model="online")
            action_indices_random = self.choose_random_action(
                action_scores, action_candidate_list)

            chosen_indices = action_indices_random
            chosen_indices = chosen_indices.astype(int)
            chosen_actions = [
                item[idx]
                for item, idx in zip(action_candidate_list, chosen_indices)
            ]
            return chosen_actions, chosen_indices, new_h, new_c

    def act(self,
            observation_strings: List[List[str]],
            action_candidate_list: List[List[List[str]]],
            previous_h=None,
            previous_c=None,
            random=False):
        if self.mode == "eval":
            return self.act_greedy(observation_strings, action_candidate_list,
                                   previous_h, previous_c)
        if random:
            return self.act_random(observation_strings, action_candidate_list,
                                   previous_h, previous_c)

        with torch.no_grad(), torch.cuda.amp.autocast(enabled=self.use_fp16):
            batch_size = len(observation_strings)

            action_scores, action_masks, new_h, new_c = self.forward(
                observation_strings,
                action_candidate_list,
                previous_h,
                previous_c,
                use_model="online")

            action_indices_maxq = self.choose_maxQ_action(
                action_scores, action_masks)
            action_indices_random = self.choose_random_action(
                action_scores, action_candidate_list)

            # random number for epsilon greedy
            rand_num = np.random.uniform(low=0.0,
                                         high=1.0,
                                         size=(batch_size, ))
            less_than_epsilon = (rand_num < self.epsilon).astype(
                "float32")  # batch
            greater_than_epsilon = 1.0 - less_than_epsilon

            chosen_indices = less_than_epsilon * action_indices_random + greater_than_epsilon * action_indices_maxq
            chosen_indices = chosen_indices.astype(int)
            chosen_actions = [
                item[idx]
                for item, idx in zip(action_candidate_list, chosen_indices)
            ]

            return chosen_actions, chosen_indices, new_h, new_c

    def get_current_q_value(self,
                            obs_list: List[List[str]],
                            candidate_list: List[List[List[str]]],
                            action_indices,
                            prev_h=None,
                            prev_c=None,
                            use_model="online"):
        action_scores, _, new_h, new_c = self.forward(obs_list,
                                                      candidate_list,
                                                      prev_h,
                                                      prev_c,
                                                      use_model=use_model)
        # ps_a
        action_indices = to_pt(action_indices,
                               enable_cuda=self.use_cuda,
                               dtype='long').unsqueeze(-1)
        q_value = ez_gather_dim_1(action_scores,
                                  action_indices).squeeze(1)  # batch
        return q_value, new_h, new_c

    def get_next_q_value(self,
                         next_obs_list: List[List[str]],
                         next_candidate_list: List[List[List[str]]],
                         prev_h=None,
                         prev_c=None):
        with torch.no_grad():
            if self.noisy_net:
                for net in self.target_net:
                    net.reset_noise()  # Sample new target net noise
            # pns Probabilities p(s_t+n, ·; θonline)
            next_action_scores, next_action_masks, _, _ = self.forward(
                next_obs_list,
                next_candidate_list,
                prev_h,
                prev_c,
                use_model="online")

            # Perform argmax action selection using online network: argmax_a[(z, p(s_t+n, a; θonline))]
            next_action_indices = self.choose_maxQ_action(
                next_action_scores, next_action_masks)  # batch
            next_action_indices = to_pt(next_action_indices,
                                        enable_cuda=self.use_cuda,
                                        dtype='long').unsqueeze(-1)
            # pns # Probabilities p(s_t+n, ·; θtarget)
            next_action_scores, _, _, _ = self.forward(next_obs_list,
                                                       next_candidate_list,
                                                       prev_h,
                                                       prev_c,
                                                       use_model="target")

            # pns_a # Double-Q probabilities p(s_t+n, argmax_a[(z, p(s_t+n, a; θonline))]; θtarget)
            next_q_value = ez_gather_dim_1(
                next_action_scores, next_action_indices).squeeze(1)  # batch
            return next_q_value

    def sample(self, episode_no, size):
        if len(self.dqn_memory) < self.replay_batch_size:
            return None
        if self.enable_recurrent_memory:
            return self.dqn_memory.sample_sequence(
                size,
                beta=self.beta_scheduler.value(episode_no),
                sample_history_length=self.replay_sample_history_length)
        else:
            return self.dqn_memory.sample(
                size,
                beta=self.beta_scheduler.value(episode_no),
                multi_step=self.multi_step)

    def update_priorities(self, actual_indices: List[int],
                          new_priorities: List[float]):
        return self.dqn_memory.update_priorities(actual_indices,
                                                 new_priorities)

    def get_dqn_loss(self, data):
        """
        Update neural model in agent. In this example we follow algorithm
        of updating model in dqn with replay memory.
        """
        (obs_list, _, candidate_list, action_indices, _, rewards,
         next_obs_list, _, next_candidate_list, _, actual_indices, actual_ns,
         prior_weights) = data

        rewards = to_pt(rewards, enable_cuda=self.use_cuda, dtype='float')

        q_value, _, _ = self.get_current_q_value(obs_list,
                                                 candidate_list,
                                                 action_indices,
                                                 use_model="online")
        next_q_value = self.get_next_q_value(next_obs_list,
                                             next_candidate_list)
        discount = to_pt((np.ones_like(actual_ns) *
                          self.discount_gamma_game_reward)**actual_ns,
                         self.use_cuda,
                         dtype="float")

        rewards = rewards + next_q_value * discount  # batch
        loss = F.smooth_l1_loss(q_value, rewards, reduction='none')  # batch

        prior_weights = to_pt(prior_weights,
                              enable_cuda=self.use_cuda,
                              dtype="float")
        loss = loss * prior_weights
        loss = torch.mean(loss)

        abs_td_error = np.abs(to_np(q_value - rewards))
        new_priorities = abs_td_error + self.prioritized_replay_eps
        self.update_priorities(actual_indices, new_priorities)

        return loss, q_value

    def get_drqn_loss(self, data):
        """
        Update neural model in agent. In this example we follow algorithm
        of updating model in dqn with replay memory.
        """
        # obs_list, candidate_list, action_indices, _, rewards, next_obs_list, next_candidate_list, _
        actual_indices, _initial, prior_weights = data[-3:]
        initial = torch.tensor(_initial)
        if self.use_cuda:
            initial = initial.cuda()
        loss_list, td_error_list, q_value_list = [], [], []
        prev_h, prev_c = None, None

        for step_no in range(self.replay_sample_history_length):
            obs_list = data[step_no][0]
            candidate_list = data[step_no][1]
            action_indices = data[step_no][2]
            rewards = data[step_no][4]
            next_obs_list = data[step_no][5]
            next_candidate_list = data[step_no][6]

            rewards = to_pt(rewards, enable_cuda=self.use_cuda, dtype='float')

            q_value, prev_h, prev_c = self.get_current_q_value(
                obs_list,
                candidate_list,
                action_indices,
                prev_h,
                prev_c,
                use_model="online")

            mask = torch.ones_like(initial)
            if step_no < self.replay_sample_update_from:
                mask = initial
                _mask = mask[:, None]
                prev_h = _mask * prev_h + ~_mask * prev_h.detach()
                prev_c = _mask * prev_c + ~_mask * prev_c.detach()
                del _mask

            next_q_value = self.get_next_q_value(next_obs_list,
                                                 next_candidate_list, prev_h,
                                                 prev_c)

            rewards = rewards + next_q_value * self.discount_gamma_game_reward
            loss = F.smooth_l1_loss(q_value, rewards, reduction='none')
            loss = loss * mask

            p_weights = to_pt(prior_weights[step_no],
                              enable_cuda=self.use_cuda,
                              dtype="float")
            loss = loss * p_weights
            loss_list.append(loss)

            abs_td_error = np.abs(to_np(q_value - rewards))
            td_error_list.append(abs_td_error)
            q_value_list.append(q_value)

        for i, td_error in enumerate(td_error_list):
            indices = [item + i for item in actual_indices]
            priorities = td_error + self.prioritized_replay_eps
            if i < self.replay_sample_update_from:
                indices = [x for x, b in zip(indices, _initial) if b]
                priorities = [x for x, b in zip(priorities, _initial) if b]
            self.update_priorities(indices, priorities)

        loss = torch.stack(loss_list).mean()
        q_value = torch.stack(q_value_list).mean()

        return loss, q_value

    def update_dqn(self, episode_no):
        data = self.sample(episode_no, self.replay_batch_size)
        if data is None:
            return None, None

        with torch.cuda.amp.autocast(enabled=self.use_fp16):
            if self.enable_recurrent_memory:
                dqn_loss, q_value = self.get_drqn_loss(data)
            else:
                dqn_loss, q_value = self.get_dqn_loss(data)

        # Backpropagate
        self.optimizer.zero_grad()
        self.scaler.scale(dqn_loss).backward()
        self.scaler.unscale_(self.optimizer)
        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(),
                                       self.clip_grad_norm)
        self.scaler.step(self.optimizer)  # apply gradients
        self.scaler.update()
        return to_np(torch.mean(dqn_loss)), to_np(torch.mean(q_value))
