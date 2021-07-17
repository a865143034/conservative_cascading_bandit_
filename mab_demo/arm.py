import random


class Arm():
    """
    For convenience, we simulate for arms with Bernoulli distribution
    """
    def __init__(self, p):
        self.p = p

    def pull(self):
        if random.random() > self.p:
            return 0.0
        else:
            return 1.0