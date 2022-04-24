import json
from collections import defaultdict
from typing import Dict, List, Set, Tuple

import spacy

Transition = Tuple[str, str]


def templetize(doc, span):
    span = list(span)
    idxs = [w.i for w in span]
    s = []
    for i, w in enumerate(doc):
        if i not in idxs:
            s.append(w.text)
        elif not (s and s[-1] == '{}'):
            s.append('{}')
    return ' '.join(x.text for x in span), ' '.join(s)


def connected_components(is_edge: List[List[bool]]) -> List[Set[int]]:
    num_entities = len(is_edge)
    visited = [False] * num_entities
    clusters = list(range(num_entities))
    for i in range(num_entities):
        if visited[i]:
            continue
        stack = [i]
        while stack:
            x = stack.pop()
            clusters[x] = clusters[i]
            visited[x] = True
            stack.extend(j for j in range(num_entities)
                         if is_edge[x][j] and not visited[j])
    members = {
        c: set(i for i in range(num_entities) if clusters[i] == c)
        for c in set(clusters)
    }
    return [v for v in members.values() if len(v) > 1]


def delexicalize(transition: Transition,
                 action_to_pairs) -> Dict[str, Transition]:
    act, obs = transition
    pairs = action_to_pairs[act]
    delexed = {}
    for phrase, templates in pairs.items():
        for template in templates:
            delexed[phrase] = (template, obs.replace(phrase, '{}'))
    return delexed


if __name__ == '__main__':
    with open('obs_gen.0.1/train.json') as f:
        corpus = json.load(f)
    nlp = spacy.load('en_core_web_sm')
    nlp.tokenizer.add_special_case('frosted-glass', [{
        'ORTH': 'frosted-glass'
    }])

    actions: Set[str] = set()
    transitions: Set[Transition] = set()
    for trajectory in corpus:
        for step in trajectory:
            actions.add(step['previous_action'])
            # reward info is already contained in the observation
            transition = (step['previous_action'], step['observation'])
            transitions.add(transition)

    phrase_set: Set[str] = set()
    action_to_pairs = {}
    for act in actions:
        pairs = defaultdict(set)
        doc = nlp('You ' + act)  # due to POS tagger bug in SpaCy
        for token in doc:
            if token.pos_ != 'NOUN':
                continue
            phrase, template = templetize(doc, token.subtree)
            if template == '{}':
                continue
            phrase_set.add(phrase)
            pairs[phrase].add(template)
        action_to_pairs[act] = pairs
    phrases: List[str] = list(phrase_set)

    associated_transitions: List[Set[Transition]] = [set() for _ in phrases]
    for transition in transitions:
        delexicalized = delexicalize(transition, action_to_pairs)
        for k, v in delexicalized.items():
            i = phrases.index(k)
            associated_transitions[i].add(v)

    threshold = 0.6
    is_edge = [[False] * len(phrases) for _ in phrases]
    for i in range(len(phrases)):
        a = associated_transitions[i]
        for j in range(i):
            b = associated_transitions[j]
            score = len(a & b) / (len(a | b) + 1e-8)
            is_edge[i][j] = score >= threshold
            is_edge[j][i] = score >= threshold

    cc = connected_components(is_edge)
    targets = set(x for c in cc for x in c)
    entities = [
        x for x in targets
        if all(phrases[y] not in phrases[x] for y in targets if y != x)
    ]
    clusters = [[x for x in c if x in entities] for c in cc]
    clusters = [v for v in clusters if len(v) > 1]

    with open('vocabularies/interchangeable-0.6.json', 'w') as f:
        obj = [[phrases[i] for i in c] for c in clusters]
        json.dump(obj, f)
