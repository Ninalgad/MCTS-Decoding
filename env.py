from copy import deepcopy
from action import Action


class Environment(object):
    """The environment the search is interacting with."""

    def __init__(self, token_list):
        self.unused = token_list
        self.seq = []
        self.hist = set()

    def __str__(self):
        return self.observation()

    def observation(self):
        return " ".join(self.seq)

    def legal_actions(self):
        return self.unused

    def step(self, action: Action):
        self.seq.append(action.token)
        self.hist.add(action.token)
        del self.unused[self.unused.index(action.token)]

    def copy(self):
        new = Environment(deepcopy(self.unused))
        new.seq = deepcopy(self.seq)
        return new
