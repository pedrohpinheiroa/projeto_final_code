class History:

    def __init__(self):
        self.history = []

    def __len__(self):
        return len(self.history)

    def add(self, state):
        self.history.append(state)

    def get(self):
        return self.history

    def clear(self):
        self.history = []
