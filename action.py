class Action(object):

    def __init__(self, token: str, index: int):
        self.token = token
        self.index = index

    def __hash__(self):
        return self.index

    def __eq__(self, other):
        return self.index == other.index
