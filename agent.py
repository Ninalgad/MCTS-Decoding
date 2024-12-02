from search import *
from mcts import run_mcts


class PermAgent:
    def __init__(self, config, network):
        self.config = config
        self.network = network

    def select_action(self, state):
        g = state.copy()
        root = Node(0, g)
        run_mcts(self.config, root, g, self.network)
        action = select_action(self.config, root)

        return action
