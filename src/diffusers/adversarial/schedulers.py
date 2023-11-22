class CoefficientScheduler:
    def __init__(self, changes_dict: dict, initial_value: float):
        self.position = 0
        self.coefficient = initial_value

        val_key = sorted([(index, value) for index, value in changes_dict.items()])

        self.indices, self.values = zip(*val_key)

    def step(self, iter_num):
        if self.position >= len(self.indices):
            return self.coefficient
        if self.indices[self.position] <= iter_num:
            self.coefficient = self.values[self.position]
            self.position += 1
        return self.coefficient
