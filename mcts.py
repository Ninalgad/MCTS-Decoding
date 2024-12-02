from search import *


# Core Monte Carlo Tree Search algorithm.
# To decide on an action, we run N simulations, always starting at the root of
# the search tree and traversing the tree according to the UCB formula until we
# reach a leaf node.
def run_mcts(config: Dict, root: Node, game,
             network: Network):
    min_max_stats = MinMaxStats()

    for _ in range(config['num_simulations']):
        node = root
        search_path = [node]

        while node.expanded():
            action, node = select_child(config, node, min_max_stats)
            search_path.append(node)

        network_output = network.inference(game.observation())

        expand_node(node, game, network_output, network)

        backpropagate(search_path, network_output.value,
                      config['discount'], min_max_stats)
