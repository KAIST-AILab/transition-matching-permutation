import argparse
import os
import random
import re
import time
from collections import Counter
from functools import lru_cache
from typing import Dict, List

import numpy as np
import spacy
import torch
import yaml
from textworld.logic import Proposition, Rule, State, Variable
from torch.nn import functional as F

missing_words = set()


def shuffled(items):
    items = sorted(items)
    random.shuffle(items)
    return items


def amalgamate(x):
    if not isinstance(x, (list, tuple)):
        return x
    if len(x) == 0:
        return x
    x = [amalgamate(item) for item in x]
    if isinstance(x[0], (np.ndarray, np.generic)):
        return np.stack(x, axis=0)
    elif isinstance(x[0], torch.Tensor):
        return torch.stack(x, dim=0)
    return x


def to_np(x) -> np.ndarray:
    x = amalgamate(x)
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, (list, tuple)):
        return np.array(x)
    return x.detach().cpu().numpy()


def to_pt(x, enable_cuda=False, dtype='long') -> torch.Tensor:
    x = amalgamate(x)
    if isinstance(x, (list, tuple)):
        x = torch.tensor(x)
    elif isinstance(x, np.ndarray):
        x = torch.from_numpy(x)

    x = x.type(torch.long if dtype == 'long' else torch.float)
    if enable_cuda:
        x = x.cuda()
    return x


def _words_to_ids(words: List[str], word2id: Dict[str, int]) -> List[int]:
    ids = []
    for word in words:
        ids.append(_word_to_id(word, word2id))
    return ids


def _word_to_id(word: str, word2id: Dict[str, int]) -> int:
    try:
        return word2id[word]
    except KeyError:
        key = word + "_" + str(len(word2id))
        if key not in missing_words:
            print("Warning... %s is not in vocab, vocab size is %d..." %
                  (word, len(word2id)))
            missing_words.add(key)
            with open("missing_words.txt", 'a+') as outfile:
                outfile.write(key + '\n')
                outfile.flush()
        return 1


def max_len(list_of_list):
    if len(list_of_list) == 0:
        return 0
    return max(map(len, list_of_list))


def max_tensor_len(list_of_tensor, dim):
    tmp = []
    for t in list_of_tensor:
        tmp.append(t.size(dim))
    return max(tmp)


def pad_sequences(sequences, maxlen=None, dtype='int32', value=0.):
    '''
    Partially borrowed from Keras
    # Arguments
        sequences: list of lists where each element is a sequence
        maxlen: int, maximum length
        dtype: type to cast the resulting sequence.
        value: float, value to pad the sequences to the desired value.
    # Returns
        x: numpy array with dimensions (number_of_sequences, maxlen)
    '''
    if isinstance(sequences, np.ndarray):
        return sequences
    lengths = [len(s) for s in sequences]
    nb_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)
    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.
    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break
    x = (np.ones((nb_samples, maxlen) + sample_shape) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if len(s) == 0:
            continue  # empty list was found
        # pre truncating
        trunc = s[-maxlen:]
        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError(
                'Shape of sample %s of sequence at position %s is different from expected shape %s'
                % (trunc.shape[1:], idx, sample_shape))
        # post padding
        x[idx, :len(trunc)] = trunc
    return x


def normalize_string(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_special_tokens(text):
        return re.sub(r'(<bos>|<eos>|<sep>|<pad>|<unk>)', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def lower(text):
        return text.lower()

    return white_space_fix(remove_special_tokens(lower(s)))


def f1_score(prediction, ground_truth):
    if prediction == ground_truth:
        return 1.0
    prediction_tokens = normalize_string(prediction).split()
    ground_truth_tokens = normalize_string(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def precision_recall_f1_score(prediction, ground_truth):
    if prediction == ground_truth:
        return 1.0, 1.0, 1.0
    prediction_tokens = normalize_string(prediction).split()
    ground_truth_tokens = normalize_string(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0., 0., 0.
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return precision, recall, f1


def f1_score_over_ground_truths(prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = f1_score(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def ez_gather_dim_1(input, index):
    if len(input.size()) == len(index.size()):
        return input.gather(1, index)
    res = []
    for i in range(input.size(0)):
        res.append(input[i][index[i][0]])
    return torch.stack(res, 0)


def get_match_result_obs_gen(prediction_string, groundtruth_string):
    pred_string = prediction_string.split("<eos>")[0].rstrip()
    gt_string = groundtruth_string.split("<eos>")[0].rstrip()
    return precision_recall_f1_score(pred_string, gt_string)


def get_match_result(prediction_string, groundtruth_string, type='exact'):
    predict_cmds = prediction_string.split("<sep>")
    if predict_cmds[-1].endswith("<eos>"):
        predict_cmds[-1] = predict_cmds[-1][:-5].strip()
    else:
        predict_cmds = predict_cmds[:-1]

    groundtruth_cmds = groundtruth_string.split("<sep>")

    predict_cmds = [" ".join(item.split()) for item in predict_cmds]
    groundtruth_cmds = [" ".join(item.split()) for item in groundtruth_cmds]
    predict_cmds = [item for item in predict_cmds if len(item) > 0]
    groundtruth_cmds = [item for item in groundtruth_cmds if len(item) > 0]

    if len(predict_cmds) == 0 and len(groundtruth_cmds) == 0:
        return 1.0, 1.0, 1.0
    elif len(predict_cmds) == 0 or len(groundtruth_cmds) == 0:
        return 0.0, 0.0, 0.0

    predict_cmds = list(set(predict_cmds))
    groundtruth_cmds = list(set(groundtruth_cmds))

    match_score = np.asarray([0.0] * len(predict_cmds), dtype='float32')

    for pred_id, pred in enumerate(predict_cmds):
        if type == 'exact':
            if pred in groundtruth_cmds:
                match_score[pred_id] = 1
        elif type == 'soft':
            match_score[pred_id] = f1_score_over_ground_truths(
                pred, groundtruth_cmds)

    precision = float(sum(match_score)) / float(
        len(predict_cmds)) if len(predict_cmds) > 0 else 0.0
    recall = float(sum(match_score)) / float(
        len(groundtruth_cmds)) if len(groundtruth_cmds) > 0 else 0.0

    if precision + recall > 0:
        f1 = float(2 * (precision * recall)) / (precision + recall)
    else:
        f1 = 0.0

    return precision, recall, f1


def power(a, b):
    if a == 0 or b == 0.0:
        return 1.0
    elif b > 0.0:
        return a**b
    else:
        return 1.0 / (a**-b)


def preproc(s, tokenizer=None):
    if s is None:
        return "nothing"
    if "$$$$$$$" in s:
        s = s.split("$$$$$$$")[-1]
    s = re.sub(r'\s+', ' ', s)
    s = s.strip()
    if len(s) == 0:
        return "nothing"
    if tokenizer:
        s = " ".join([t.text for t in tokenizer(s)])
    s = s.lower()
    return s


def load_tokenizer():
    nlp = spacy.load('en_core_web_sm',
                     exclude=['ner', 'parser', 'tagger', 'lemmatizer'])
    tokenizer = nlp.tokenizer
    special_tokens = [
        '<pad>', '<unk>', '<bos>', '<eos>', '<sep>', 'frosted-glass'
    ]
    for tok in special_tokens:
        tokenizer.add_special_case(tok, [{spacy.symbols.ORTH: tok}])
    return tokenizer


##############################
# KG stuff
##############################
# relations
two_args_relations = [
    "in", "on", "at", "west_of", "east_of", "north_of", "south_of", "part_of",
    "needs"
]
one_arg_state_relations = [
    "chopped", "roasted", "diced", "burned", "open", "fried", "grilled",
    "consumed", "closed", "sliced", "uncut", "raw"
]
ignore_relations = [
    "cuttable", "edible", "drinkable", "sharp", "inedible", "cut", "cooked",
    "cookable", "needs_cooking"
]
opposite_relations = {
    "west_of": "east_of",
    "east_of": "west_of",
    "south_of": "north_of",
    "north_of": "south_of"
}
equivalent_entities = {"inventory": "player", "recipe": "cookbook"}
FOOD_FACTS = [
    "sliced", "diced", "chopped", "cut", "uncut", "cooked", "burned",
    "grilled", "fried", "roasted", "raw", "edible", "inedible"
]


def process_exits_in_triplet(triplet):
    # ["exit", "kitchen", "backyard", "south_of"]
    if triplet[0] == "exit":
        return [triplet[0], triplet[1], triplet[3]]
    else:
        return triplet


def process_burning_triplets(list_of_triplets):
    burned_stuff = []
    for t in list_of_triplets:
        if "burned" in t:
            burned_stuff.append(t[0])
    res = []
    for t in list_of_triplets:
        if t[0] in burned_stuff and t[1] in ["grilled", "fried", "roasted"]:
            continue
        res.append(t)
    return res


def sort_target_commands(list_of_cmds):
    list_of_cmds = [item.split(" , ") for item in list_of_cmds]

    list_of_cmds.sort(key=lambda i: (
        i[0] == "add",  # add always before delete
        i[1] == "player",  # relations with player always first
        i[2] == "player",  # relations with player always first
        i[3] in ["west_of", "east_of", "north_of", "south_of"
                 ],  # room connections always first
        i[3] in ["part_of"],  # recipe
        i[3] in two_args_relations,  # two args relations first
        i[3] in ["is"],  # one arg state relations first
        i[3] in ["needs"],  # one arg requirement relations first
        i[2],
        i[1]))
    list_of_cmds = [" ".join(item) for item in list_of_cmds]
    list_of_cmds = list_of_cmds[::-1]
    res = []
    for cmd in list_of_cmds:
        if cmd not in res:
            res.append(cmd)
    return res


@lru_cache()
def _rules_predicates_scope():
    rules = [
        Rule.parse("query :: at(P, r) -> at(P, r)"),
        Rule.parse("query :: at(P, r) & at(o, r) -> at(o, r)"),
        Rule.parse("query :: at(P, r) & at(d, r) -> at(d, r)"),
        Rule.parse("query :: at(P, r) & at(s, r) -> at(s, r)"),
        Rule.parse("query :: at(P, r) & at(c, r) -> at(c, r)"),
        Rule.parse("query :: at(P, r) & at(s, r) & on(o, s) -> on(o, s)"),
        Rule.parse("query :: at(P, r) & at(c, r) & open(c) -> open(c)"),
        Rule.parse("query :: at(P, r) & at(c, r) & closed(c) -> closed(c)"),
        Rule.parse(
            "query :: at(P, r) & at(c, r) & open(c) & in(o, c) -> in(o, c)"),
        Rule.parse("query :: at(P, r) & link(r, d, r') & open(d) -> open(d)"),
        Rule.parse(
            "query :: at(P, r) & link(r, d, r') & closed(d) -> closed(d)"),
        Rule.parse(
            "query :: at(P, r) & link(r, d, r') & north_of(r', r) -> north_of(d, r)"
        ),
        Rule.parse(
            "query :: at(P, r) & link(r, d, r') & south_of(r', r) -> south_of(d, r)"
        ),
        Rule.parse(
            "query :: at(P, r) & link(r, d, r') & west_of(r', r) -> west_of(d, r)"
        ),
        Rule.parse(
            "query :: at(P, r) & link(r, d, r') & east_of(r', r) -> east_of(d, r)"
        ),
    ]
    rules += [
        Rule.parse(
            "query :: at(P, r) & at(f, r) & {fact}(f) -> {fact}(f)".format(
                fact=fact)) for fact in FOOD_FACTS
    ]
    rules += [
        Rule.parse(
            "query :: at(P, r) & at(s, r) & on(f, s) & {fact}(f) -> {fact}(f)".
            format(fact=fact)) for fact in FOOD_FACTS
    ]
    rules += [
        Rule.parse(
            "query :: at(P, r) & at(c, r) & open(c) & in(f, c) & {fact}(f) -> {fact}(f)"
            .format(fact=fact)) for fact in FOOD_FACTS
    ]
    return rules


@lru_cache()
def _rules_predicates_recipe():
    rules = [
        Rule.parse(
            "query :: in(ingredient, RECIPE) & base(f, ingredient) -> part_of(f, RECIPE)"
        ),
        Rule.parse(
            "query :: in(ingredient, RECIPE) & base(f, ingredient) & roasted(ingredient) -> needs_roasted(f)"
        ),
        Rule.parse(
            "query :: in(ingredient, RECIPE) & base(f, ingredient) & grilled(ingredient) -> needs_grilled(f)"
        ),
        Rule.parse(
            "query :: in(ingredient, RECIPE) & base(f, ingredient) & fried(ingredient) -> needs_fried(f)"
        ),
        Rule.parse(
            "query :: in(ingredient, RECIPE) & base(f, ingredient) & sliced(ingredient) -> needs_sliced(f)"
        ),
        Rule.parse(
            "query :: in(ingredient, RECIPE) & base(f, ingredient) & chopped(ingredient) -> needs_chopped(f)"
        ),
        Rule.parse(
            "query :: in(ingredient, RECIPE) & base(f, ingredient) & diced(ingredient) -> needs_diced(f)"
        ),
    ]
    return rules


@lru_cache()
def _rules_exits():
    rules = [
        Rule.parse("query :: at(P, r) & north_of(r', r) -> north_of(r', r)"),
        Rule.parse("query :: at(P, r) & west_of(r', r) -> west_of(r', r)"),
        Rule.parse("query :: at(P, r) & south_of(r', r) -> south_of(r', r)"),
        Rule.parse("query :: at(P, r) & east_of(r', r) -> east_of(r', r)"),
    ]
    return rules


@lru_cache()
def _rules_predicates_inv():
    rules = [
        Rule.parse("query :: in(o, I) -> in(o, I)"),
    ]
    rules += [
        Rule.parse(
            "query :: in(f, I) & {fact}(f) -> {fact}(f)".format(fact=fact))
        for fact in FOOD_FACTS
    ]
    return rules


@lru_cache()
def _rules_to_convert_link_predicates():
    rules = [
        Rule.parse(
            "query :: link(r, d, r') & north_of(r', r) -> north_of(d, r)"),
        Rule.parse(
            "query :: link(r, d, r') & south_of(r', r) -> south_of(d, r)"),
        Rule.parse(
            "query :: link(r, d, r') & west_of(r', r) -> west_of(d, r)"),
        Rule.parse(
            "query :: link(r, d, r') & east_of(r', r) -> east_of(d, r)"),
    ]
    return rules


def find_predicates_in_scope(state):
    actions = state.all_applicable_actions(_rules_predicates_scope())
    return [action.postconditions[0] for action in actions]


def find_exits_in_scope(state):
    actions = state.all_applicable_actions(_rules_exits())

    def _convert_to_exit_fact(proposition):
        return Proposition(proposition.name, [
            Variable("exit", "LOCATION"), proposition.arguments[1],
            proposition.arguments[0]
        ])

    return [
        _convert_to_exit_fact(action.postconditions[0]) for action in actions
    ]


def convert_link_predicates(state):
    actions = state.all_applicable_actions(_rules_to_convert_link_predicates())
    for action in list(actions):
        state.apply(action)
    return state


def find_predicates_in_inventory(state):
    actions = state.all_applicable_actions(_rules_predicates_inv())
    return [action.postconditions[0] for action in actions]


def find_predicates_in_recipe(state):
    actions = state.all_applicable_actions(_rules_predicates_recipe())

    def _convert_to_needs_relation(proposition):
        if not proposition.name.startswith("needs_"):
            return proposition

        return Proposition("needs", [
            proposition.arguments[0],
            Variable(proposition.name.split("needs_")[-1], "STATE")
        ])

    return [
        _convert_to_needs_relation(action.postconditions[0])
        for action in actions
    ]


def process_facts(prev_facts, info_game, info_facts, info_last_action, cmd):
    kb = info_game.kb
    if prev_facts is None or cmd == "restart":
        facts = set()
    else:
        if cmd == "inventory":  # Bypassing TextWorld's action detection.
            facts = set(
                find_predicates_in_inventory(State(kb.logic, info_facts)))
            return prev_facts | facts

        elif info_last_action is None:
            return prev_facts  # Invalid action, nothing has changed.

        elif info_last_action.name == "examine" and "cookbook" in [
                v.name for v in info_last_action.variables
        ]:
            facts = set(find_predicates_in_recipe(State(kb.logic, info_facts)))
            return prev_facts | facts

        state = State(kb.logic,
                      prev_facts | set(info_last_action.preconditions))
        success = state.apply(info_last_action)
        assert success
        facts = set(state.facts)

    # Always add facts in sight.
    facts |= set(find_predicates_in_scope(State(kb.logic, info_facts)))
    facts |= set(find_exits_in_scope(State(kb.logic, info_facts)))

    return facts


def process_fully_obs_facts(info_game, facts):
    state = State(info_game.kb.logic, facts)
    state = convert_link_predicates(state)
    inventory_facts = set(find_predicates_in_inventory(state))
    recipe_facts = set(find_predicates_in_recipe(state))
    return set(state.facts) | inventory_facts | recipe_facts


def process_local_obs_facts(info_game, info_facts, info_last_action, cmd):
    def _get_state():
        return State(info_game.kb.logic, info_facts)

    if cmd == "inventory":  # Bypassing TextWorld's action detection.
        return set(find_predicates_in_inventory(_get_state()))

    elif (info_last_action
          and info_last_action.name.startswith("go")) or cmd in [
              "restart", "look"
          ]:
        # Facts in sight.
        state = _get_state()
        facts = set(find_predicates_in_scope(state))
        facts |= set(find_exits_in_scope(state))
        return facts

    elif info_last_action is None:
        return set()  # Invalid action, no facts.

    elif info_last_action.name == "examine" and "cookbook" in [
            v.name for v in info_last_action.variables
    ]:
        return set(find_predicates_in_recipe(_get_state()))

    return info_last_action.postconditions


def serialize_facts(facts):
    PREDICATES_TO_DISCARD = {
        "ingredient_1", "ingredient_2", "ingredient_3", "ingredient_4",
        "ingredient_5", "out", "free", "used", "cooking_location", "link"
    }
    CONSTANT_NAMES = {
        "P": "player",
        "I": "player",
        "ingredient": None,
        "slot": None,
        "RECIPE": "cookbook"
    }
    # e.g. [("wooden door", "backyard", "in"), ...]
    serialized = [[
        arg.name if arg.name and arg.type not in CONSTANT_NAMES else
        CONSTANT_NAMES[arg.type] for arg in fact.arguments
    ] + [fact.name] for fact in sorted(facts)
                  if fact.name not in PREDICATES_TO_DISCARD]
    return filter_triplets([fact for fact in serialized if None not in fact])


def filter_triplets(triplets):
    tp = []
    for item in triplets:
        item = process_exits_in_triplet(item)
        if item[-1] in (two_args_relations + one_arg_state_relations):
            tp.append([it.lower() for it in item])
        else:
            if item[-1] not in ignore_relations:
                print("Warning..., %s not in known relations..." % (item[-1]))

    for i in range(len(tp)):
        if tp[i][-1] in one_arg_state_relations:
            tp[i].append("is")

    tp = process_burning_triplets(tp)
    return tp


class HistoryScoreCache:
    def __init__(self, capacity=1):
        self.capacity = capacity
        self.reset()

    def push(self, stuff):
        """stuff is float."""
        if len(self.memory) < self.capacity:
            self.memory.append(stuff)
        else:
            self.memory = self.memory[1:] + [stuff]

    def extend(self, iterable):
        self.memory.extend(iterable)
        if len(self.memory) < self.capacity:
            self.memory = self.memory[-self.capacity:]

    def get_avg(self):
        if not self.memory: return 0
        return np.mean(np.array(self.memory))

    def reset(self):
        self.memory = []

    def __len__(self):
        return len(self.memory)


class LinearSchedule(object):
    """
    Linear interpolation between initial_p and final_p over
    schedule_timesteps. After this many timesteps pass final_p is
    returned.
    :param schedule_timesteps: (int) Number of timesteps for which to linearly anneal initial_p to final_p
    :param initial_p: (float) initial output value
    :param final_p: (float) final output value
    """
    def __init__(self, schedule_timesteps, final_p, initial_p=1.0):
        self.schedule_timesteps = schedule_timesteps
        self.final_p = final_p
        self.initial_p = initial_p
        self.schedule = np.linspace(initial_p, final_p, schedule_timesteps)

    def value(self, step):
        if step < 0:
            return self.initial_p
        if step >= self.schedule_timesteps:
            return self.final_p
        else:
            return self.schedule[step]


def load_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", help="path to config file")
    parser.add_argument("-p",
                        "--params",
                        nargs="+",
                        metavar="my.setting=value",
                        default=[],
                        help="override params of the config file,"
                        " e.g. -p 'training.gamma=0.95'")
    args = parser.parse_args()
    assert os.path.exists(args.config_file), "Invalid config file"
    with open(args.config_file) as reader:
        config = yaml.safe_load(reader)
    # Parse overriden params.
    for param in args.params:
        fqn_key, value = param.split("=")
        entry_to_change = config
        keys = fqn_key.split(".")
        for k in keys[:-1]:
            entry_to_change = entry_to_change[k]
        entry_to_change[keys[-1]] = yaml.load(value)
    # print(config)
    return config


class Stopwatch:
    def __init__(self, tag, threshold=1):
        self.tag = tag
        self.threshold = threshold

    def start(self):
        self.time = time.time()

    def end(self):
        delta = time.time() - self.time
        if delta > self.threshold:
            print(f'{self.tag} waited {int(delta * 1000)} ms')


class Average:
    def __init__(self):
        self.value = 10000.0
        self.n = 0

    def reset(self):
        self.value = 10000.0
        self.n = 0

    def append(self, value):
        self.value = (self.n * self.value + value) / (self.n + 1)
        self.n += 1

    def mean(self):
        return self.value


class VisdomManager:
    def __init__(self, viz, experiment_tag):
        self.viz = viz
        self.experiment_tag = experiment_tag

        self.windows = {}
        self.plots = {}
        self.registry = {}

    def register(self, window, keys):
        self.registry[window] = keys

    def update(self, key, value):
        if key not in self.plots:
            self.plots[key] = []
        self.plots[key].append(value)

    def _plot_window(self, window, name):
        opts = {'title': self.experiment_tag + '_' + window}
        viz_y = self.plots[name]
        if window not in self.windows:
            viz_x = np.arange(len(viz_y)).tolist()
            self.windows[window] = self.viz.line(X=viz_x,
                                                 Y=viz_y,
                                                 opts=opts,
                                                 name=name)
        else:
            viz_x = [len(viz_y) - 1]
            viz_y = [viz_y[-1]]
            self.viz.line(X=viz_x,
                          Y=viz_y,
                          win=self.windows[window],
                          update='append',
                          name=name)

    def plot(self):
        for window, keys in self.registry.items():
            for key in keys:
                self._plot_window(window, key)


def recursive_map(mapper, obj):
    if isinstance(obj, list):
        return list(recursive_map(mapper, x) for x in obj)
    if isinstance(obj, tuple):
        return tuple(recursive_map(mapper, x) for x in obj)
    if isinstance(obj, dict):
        return {k: recursive_map(mapper, v) for k, v in obj.items()}
    return mapper(obj)


def to_shape(tensor):
    try:
        return tensor.shape
    except AttributeError:
        return tensor


#############################
# SEQ3
#############################


def straight_softmax(logits, tau=1, hard=False, target_mask=None):
    y_soft = F.softmax(logits / tau, dim=1)

    if target_mask is not None:
        y_soft = y_soft * target_mask.to(y_soft.dtype)
        y_soft.div(y_soft.sum(-1, keepdim=True))

    if hard:
        shape = logits.size()
        _, k = y_soft.max(-1)
        y_hard = logits.new_zeros(*shape).scatter_(-1, k.view(-1, 1), 1.0)
        y = y_hard - y_soft.detach() + y_soft
        return y
    else:
        return y_soft
