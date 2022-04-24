import json
import random
import re
from pathlib import Path
from typing import Iterable, List, Optional

import gym
import textworld.gym

from generic import Stopwatch, load_tokenizer, preproc, shuffled

SUMMARY_CACHE_SIZE = 2**13


def get_training_game_env(data_dir, difficulty_level, training_size,
                          requested_infos, max_episode_steps, batch_size,
                          config):
    # training games
    game_file_names: List[Path] = []
    game_path = (Path(data_dir) / f"train_{training_size}" /
                 f"difficulty_level_{difficulty_level}")
    if game_path.is_dir():
        game_file_names = list(game_path.glob('*.z8'))
    else:
        game_file_names.append(game_path)
    game_file_names.sort()

    env_config = config['rl']['env']
    env_id = textworld.gym.register_games(list(map(str, game_file_names)),
                                          request_infos=requested_infos,
                                          max_episode_steps=None,
                                          batch_size=batch_size,
                                          name="training",
                                          asynchronous=False,
                                          auto_reset=False)
    env = gym.make(env_id)

    env = _wrap_env(env, env_config, game_file_names)
    num_games = len(game_file_names)
    return env, num_games


def get_evaluation_game_env(data_dir,
                            difficulty_level,
                            requested_infos,
                            max_episode_steps,
                            batch_size,
                            config,
                            valid_or_test="valid"):
    assert valid_or_test in ["valid", "test"]
    # eval games
    game_file_names: List[Path] = []
    game_path = (Path(data_dir) / valid_or_test /
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
                                          name="eval",
                                          asynchronous=False,
                                          auto_reset=False)
    env = gym.make(env_id)

    env = _wrap_env(env, env_config)
    num_games = len(game_file_names)
    return env, num_games


def _wrap_env(env, config, game_file_names=None):
    split = 'eval' if game_file_names is None else 'train'

    env = EnhancedEnv(env, config, game_file_names)
    if split == 'train' and config.get('permute_entity'):
        env = EntityPermuteEnv(env, config)

    env = FormatEnv(env, config)
    env = TokenizationEnv(env, config)

    if config.get('lemma'):
        env = LemmatizeEnv(env, config)
    if config.get('stopwords'):
        env = StopwordEnv(env, config)

    if config["provide_history"]:
        env = HistoryEnv(env, config)
    env = TruncateEnv(env, config)

    env = LowerEnv(env, config)
    if not config.get('tokenize', True):
        env = StringEnv(env, config)
    return env


class BaseEnv:
    def __init__(self, env, config):
        self.env = env
        self.config = config
        self.sw = Stopwatch(self.__class__.__name__)

    def seed(self, seed):
        self.env.seed(seed)

    @property
    def batch_size(self):
        return self.env.batch_size

    def _proc_obs(self, obs):
        return obs

    def _proc_act(self, action):
        return action

    def _proc_cands(self, cmds):
        return cmds

    def _proc_infos(self, infos):
        cmds = infos['admissible_commands']
        cmds = [list(self._proc_cands(_cmds)) for _cmds in cmds]
        return {**infos, 'admissible_commands': cmds}

    def reset(self):
        obs, infos = self.env.reset()
        self.sw.start()
        obs = [self._proc_obs(o) for o in obs]
        infos = self._proc_infos(infos)
        self.sw.end()
        return obs, infos

    def step(self, action):
        action = [self._proc_act(a) for a in action]
        obs, scores, dones, infos = self.env.step(action)
        self.sw.start()
        obs = [self._proc_obs(o) for o in obs]
        infos = self._proc_infos(infos)
        self.sw.end()
        return obs, scores, dones, infos


class EnhancedEnv(BaseEnv):
    def __init__(self,
                 env,
                 config,
                 game_file_names: Optional[List[Path]] = None):
        super().__init__(env, config)
        self.recipe = None
        self.seen_recipe = None
        self.reached_kitchen = None
        self.game_indices = None
        self.game_file_names = None
        if game_file_names is not None:
            self.game_file_names = [game.stem for game in game_file_names]

    def _filter_admissible_commands(self, infos):
        forbid = ['look', 'inventory', 'examine']
        for cmds in zip(infos["admissible_commands"]):
            filtered = [
                c for c in cmds
                if c != "examine cookbook" and c.split()[0] in forbid
            ]
            for cmd in filtered:
                cmds.remove(cmd)

    def _get_game_indices(self, infos):
        if self.game_file_names is None:
            return None
        names = [game.metadata['uuid'] for game in infos['game']]
        indices = [self.game_file_names.index(name) for name in names]
        return indices

    def reset(self):
        obs, infos = super().reset()
        obs = tuple(map(preproc, obs))
        self.seen_recipe = [False] * len(obs)
        self.reached_kitchen = ['kitchen' in o for o in obs]
        self.recipe = [''] * len(obs)
        self.game_indices = self._get_game_indices(infos)
        self._filter_admissible_commands(infos)
        infos['extra.recipe'] = self.recipe
        infos['extra.seen_recipe'] = self.seen_recipe
        infos['extra.reached_kitchen'] = self.reached_kitchen
        infos['extra.game_indices'] = self.game_indices
        return obs, infos

    def _get_recipe(self, obs):
        bar = '-' * 9 + ' '
        return bar + obs.split(bar)[-1]

    def step(self, action):
        obs, scores, dones, infos = super().step(action)
        obs = tuple(map(preproc, obs))
        for i, (o, d) in enumerate(zip(obs, dones)):
            if 'kitchen' in o and not d:
                self.reached_kitchen[i] = True
        for i, (a, d) in enumerate(zip(action, dones)):
            if a == 'examine cookbook' and not d:
                self.seen_recipe[i] = True
                self.recipe[i] = self.recipe[i] or self._get_recipe(obs[i])

        self._filter_admissible_commands(infos)
        infos['extra.recipe'] = [
            '' if r in o else r for o, r in zip(obs, self.recipe)
        ]
        infos['extra.seen_recipe'] = self.seen_recipe
        infos['extra.reached_kitchen'] = self.reached_kitchen
        infos['extra.game_indices'] = self.game_indices
        return obs, scores, dones, infos


Token = str
Tokens = List[Token]


def simple_tokenize(string: str) -> Tokens:
    string = string.replace('.', ' . ').replace(':', ' : ').strip()
    return string.split()


def regex_tokenize(string: str) -> Tokens:
    pattern = r"<\w+>|'\w+|-=|\.\.\.|n't|turn\(s|[\w\d-]+(?!'t)|[^\w\d\s]"
    tokens = re.findall(pattern, string, flags=re.IGNORECASE)
    return tokens


class TokenizationEnv(BaseEnv):
    def __init__(self, env, config):
        super().__init__(env, config)
        self.type = config.get('tokenizer', 'regex')

        if self.type == 'spacy':
            self.tokenizer = load_tokenizer()
        elif self.type == 'regex':
            self.tokenizer = regex_tokenize
        else:
            self.tokenizer = simple_tokenize

    def tokenize(self, string: str) -> Tokens:
        tokens = self.tokenizer(string)
        if self.type == 'spacy':
            tokens = [w.text for w in tokens if not w.is_space]
        return tokens

    def _proc_obs(self, obs: str) -> Tokens:
        return self.tokenize(obs)

    def _proc_cands(self, cmds: Iterable[str]) -> Iterable[Tokens]:
        return map(self.tokenize, cmds)

    def _proc_act(self, action: Tokens) -> str:
        return ' '.join(action)


class LemmatizeEnv(BaseEnv):
    def __init__(self, env, config):
        super().__init__(env, config)
        with open(config['lemma']) as f:
            self.lemma = dict(line.strip().split('\t') for line in f)

    def _proc_obs(self, obs: Tokens) -> Tokens:
        return [self.lemma.get(token, token) for token in obs]


class StopwordEnv(BaseEnv):
    def __init__(self, env, config):
        super().__init__(env, config)
        with open(config['stopwords']) as f:
            self.stopwords = set(line.strip() for line in f)
        self.deduplicate = config.get('deduplicate', False)

    def _proc_obs(self, obs: Tokens) -> Tokens:
        result = []
        last = None
        for token in obs:
            if token == last:
                continue
            if token not in self.stopwords or token.isupper():
                result.append(token)
                if self.deduplicate:
                    last = token
        return result


class TruncateEnv(BaseEnv):
    def __init__(self, env, config):
        super().__init__(env, config)
        self.max_length = config['max_token']

    def _proc_obs(self, obs):
        return obs[-self.max_length:]


class LowerEnv(BaseEnv):
    def _proc_obs(self, obs: Tokens) -> Tokens:
        return [w.lower() for w in obs]

    def _proc_cands(self, cands: Iterable[Tokens]) -> Iterable[Tokens]:
        return map(self._proc_obs, cands)


class StringEnv(BaseEnv):
    def _proc_obs(self, obs: Tokens) -> str:
        return ' '.join(obs)

    def _proc_act(self, act: str) -> Tokens:
        return act.split(' ')

    def _proc_cands(self, cands: Iterable[Tokens]) -> Iterable[str]:
        return map(self._proc_obs, cands)


class FormatEnv(BaseEnv):
    def __init__(self, env, config):
        super().__init__(env, config)
        self.observation_format = config['observation_format']
        self.candidates = None

    def format_observations(self, **kwargs):
        for k, v in kwargs.copy().items():
            kwargs[k.upper()] = [x.upper() for x in v]
        batch_size = len(kwargs[list(kwargs.keys())[0]])
        per_batch = [{k: v[i]
                      for k, v in kwargs.items()} for i in range(batch_size)]
        return tuple(
            self.observation_format.format(**_kwargs) for _kwargs in per_batch)

    def reset(self):
        raw_obs, infos = self.env.reset()
        batch_size = len(raw_obs)
        prev_act = ['restart'] * batch_size
        recipe = infos['extra.recipe']
        obs = self.format_observations(
            prev_act=prev_act,
            obs=raw_obs,
            recipe=recipe,
        )
        return obs, infos

    def step(self, action):
        raw_obs, scores, dones, infos = self.env.step(action)

        recipe = infos['extra.recipe']
        obs = self.format_observations(
            prev_act=action,
            obs=raw_obs,
            recipe=recipe,
        )
        return obs, scores, dones, infos


class HistoryEnv(BaseEnv):
    def __init__(self, env, config):
        super().__init__(env, config)
        self.history = []

    def reset(self):
        obs, infos = self.env.reset()
        self.history = obs
        return obs, infos

    def step(self, action):
        obs, scores, dones, infos = self.env.step(action)
        obs = tuple(h + o for h, o in zip(self.history, obs))
        self.history = obs
        return obs, scores, dones, infos


def permute_observation(obs, food_map):
    permuted = obs.lower()
    for key in ['patio table', 'sliding patio door']:
        if key in food_map:
            permuted = permuted.replace(key, food_map[key].upper())

    for src, dest in food_map.items():
        permuted = permuted.replace(src.lower(), dest.upper())
    return permuted


class EntityPermuteEnv:
    def __init__(self, env, config):
        self.env = env
        self.config = config
        self.entity_map = None
        self.inverse_map = None
        with open(config['permute_entity']) as f:
            self.entity_class = json.load(f)

    def seed(self, _seed):
        random.seed(_seed)
        self.env.seed(_seed)

    def _generate_food_permutation(self):
        self.entity_map = {}
        for original in self.entity_class:
            permuted = shuffled(original)
            self.entity_map.update({x: y for x, y in zip(original, permuted)})
        self.inverse_map = {y: x for x, y in self.entity_map.items()}

    def permute_action(self, infos):
        new_cmds = [[permute_observation(x, self.entity_map) for x in cmds]
                    for cmds in infos["admissible_commands"]]
        return {**infos, 'admissible_commands': new_cmds}

    def parse_action(self, action):
        return permute_observation(action, self.inverse_map)

    def permute_recipe(self, infos):
        recipes = infos['extra.recipe']
        recipes = [permute_observation(x, self.entity_map) for x in recipes]
        return {**infos, 'extra.recipe': recipes}

    def reset(self):
        self._generate_food_permutation()
        obs, infos = self.env.reset()
        obs = [permute_observation(o, self.entity_map) for o in obs]
        infos = self.permute_action(infos)
        infos = self.permute_recipe(infos)
        return obs, infos

    def step(self, action):
        action = [self.parse_action(a) for a in action]
        obs, scores, dones, infos = self.env.step(action)
        obs = [permute_observation(o, self.entity_map) for o in obs]
        infos = self.permute_action(infos)
        infos = self.permute_recipe(infos)
        return obs, scores, dones, infos
