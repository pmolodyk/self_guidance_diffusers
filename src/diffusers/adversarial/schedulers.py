import numpy as np


class CoefficientScheduler:
    def __init__(self, changes_dict: dict, initial_value: float):
        self.position = 1
        self.coefficient = initial_value
        changes_dict[0] = initial_value

        val_key = sorted(changes_dict.items())
        self.indices, self.values = zip(*val_key)

    def step(self, iter_num):
        if self.position >= len(self.indices):
            return self.coefficient
        if self.indices[self.position] <= iter_num:
            self.coefficient = self.values[self.position]
            self.position += 1
        return self.coefficient


class LinearScheduler(CoefficientScheduler):
    def __init__(self, n_steps: int, initial_value: float, final_value: float):
        super().__init__(dict(zip(range(n_steps), np.linspace(initial_value, final_value, n_steps))), initial_value)


def get_scheduler(type, changes_dict, initial_value, n_steps): 
    if type == 'basic':
        return CoefficientScheduler(changes_dict, initial_value)
    elif type == 'linear':
        return LinearScheduler(n_steps, initial_value, 0)
    else:
        raise ValueError(f'no type called {type}')
