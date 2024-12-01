class Action(object):

    def __init__(self, text: str, index: int):
        self.text = text
        self.index = index

    def __hash__(self):
        return self.index

    def __eq__(self, other):
        return self.index == other.index
