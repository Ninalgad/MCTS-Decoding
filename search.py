from typing import Dict, List, Optional
import numpy as np
import math

from utils import *
from network import NetworkOutput, Network
from action import Action

MAXIMUM_FLOAT_VALUE = float('inf')


class MinMaxStats(object):
    """A class that holds the min-max values of the tree."""

    def __init__(self):
        self.maximum = -MAXIMUM_FLOAT_VALUE
        self.minimum = MAXIMUM_FLOAT_VALUE

    def update(self, value: float):
        self.maximum = max(self.maximum, value)
        self.minimum = min(self.minimum, value)

    def normalize(self, value: float) -> float:
        if self.maximum > self.minimum:
            # We normalize only when we have set the maximum and minimum values.
            return (value - self.minimum) / (self.maximum - self.minimum)
        return value


class Node(object):

    def __init__(self, prior, game):
        self.visit_count = 0
        self.prior = prior
        self.value_sum = 0
        self.children = {}
        self.reward = 0

        self.game = game

    def expanded(self) -> bool:
        return len(self.children) > 0

    def value(self) -> float:
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count


# Select the child with the highest UCB score.
def select_child(config: Dict, node: Node,
                 min_max_stats: MinMaxStats):
    best_score = -MAXIMUM_FLOAT_VALUE
    best_action, best_child = None, None
    for action, child in node.children.items():
        score = ucb_score(config, node, child, min_max_stats)
        if score > best_score:
            best_score = score
            best_action, best_child = action, child

    return best_action, best_child


def expand_node(config: Dict, node: Node, game, network_output: NetworkOutput, network: Network):
    node.reward = network_output.reward
    actions = [Action(t, network.get_token_id(t)) for t in game.legal_actions()]

    policy, policy_sum = {}, 0
    for a in actions:
        logit = network_output.policy_logits[a.index]
        t = config['policy_temperature']
        if a.token in game.seq:
            t = config['repetition_penalty']
        z = math.exp(logit / t)

        policy_sum += z
        policy[a] = z

    for action, p in policy.items():
        g = game.copy()
        g.step(action)
        node.children[action] = Node(p / policy_sum, g)


def select_action(config: Dict, node: Node):
    visit_counts, actions = [], []
    for action, child in node.children.items():
        visit_counts.append(child.visit_count)
        actions.append(action)

    t = config['visit_softmax_temperature']
    _, action = softmax_sample(actions, visit_counts, t)
    return action


# The score for a node is based on its value, plus an exploration bonus based on
# the prior.
def ucb_score(config: Dict, parent: Node, child: Node,
              min_max_stats: MinMaxStats) -> float:
    pb_c = math.log((parent.visit_count + config['pb_c_base'] + 1) /
                    config['pb_c_base']) + config['pb_c_init']
    pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)

    prior_score = pb_c * child.prior
    value_score = min_max_stats.normalize(child.value())
    return prior_score + value_score


# At the end of a simulation, we propagate the evaluation all the way up the
# tree to the root.
def backpropagate(search_path: List[Node], value: float,
                  discount: float, min_max_stats: MinMaxStats):
    for node in search_path:
        node.value_sum += value
        node.visit_count += 1
        min_max_stats.update(node.value())

        value = node.reward + discount * value


# At the start of each search, we add dirichlet noise to the prior of the root
# to encourage the search to explore new actions.
def add_exploration_noise(config: Dict, node: Node):
    actions = list(node.children.keys())
    noise = np.random.dirichlet([config['root_dirichlet_alpha']] * len(actions))
    frac = config['root_exploration_fraction']
    for a, n in zip(actions, noise):
        node.children[a].prior = node.children[a].prior * (1 - frac) + n * frac
