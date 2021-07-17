import math
import random

def index_max(x):
  m = max(x)
  return x.index(m)


class UniformExploration():
    """
    Implementation of Uniform Exploration algorithm
    K: The number of arms
    T: Time Horizon
    N: The round for uniform exploration
    counts: Count the number of samples for each arm
    values: The empirical accumulative rewards for each arm
    """
    def __init__(self, K, N):
        self.K = K
        self.N = N

    def initialize(self):
        self.counts = [0 for col in range(self.K)]
        self.values = [0.0 for col in range(self.K)]

    def select_arm(self):
        for arm in range(self.K):
            if self.counts[arm] < self.N:
                return arm
        return index_max(self.values)

    def update(self, chosen_arm, reward):
        self.counts[chosen_arm] += 1
        n = self.counts[chosen_arm]
        value = self.values[chosen_arm]
        new_value = ((n - 1) / float(n)) * value + (1 / float(n)) * reward
        self.values[chosen_arm] = new_value


class UCB1():
    """
    Implementation of UCB algorithm
    K: The number of arms
    T: Time Horizon
    counts: Count the number of samples for each arm
    values: The empirical accumulative rewards for each arm
    The initialization can be non-zero to encourage exploration
    """
    def __init__(self, K):
        self.K = K
        self.t = 0

    def initialize(self):
        self.counts = [0 for col in range(self.K)]
        self.values = [0.0 for col in range(self.K)]

    def select_arm(self):
        """
        If an arm has never been selected, its UCB is viewed as infinity
        """
        for arm in range(self.K):
            if self.counts[arm] == 0:
                return arm
        ucb = [0.0 for arm in range(self.K)]
        for arm in range(self.K):
            radius = math.sqrt((2 * math.log(self.t)) / float(self.counts[arm]))
            ucb[arm] = self.values[arm] + radius
        return index_max(ucb)

    def update(self, chosen_arm, reward):
        self.counts[chosen_arm] += 1
        n = self.counts[chosen_arm]
        value = self.values[chosen_arm]
        new_value = ((n - 1) / float(n)) * value + (1 / float(n)) * reward
        self.values[chosen_arm] = new_value
        self.t += 1


class UCB2(object):
    def __init__(self, alpha, n_arms):
        self.alpha = alpha
        self.__current_arm = 0
        self.__next_update = 0
        self.n_arms = n_arms
        return

    def initialize(self):
        self.counts = [0 for col in range(self.n_arms)]
        self.values = [0.0 for col in range(self.n_arms)]
        self.r = [0 for col in range(self.n_arms)]
        self.__current_arm = 0
        self.__next_update = 0

    def __bonus(self, n, r):
        tau = self.__tau(r)
        bonus = math.sqrt((1. + self.alpha) * math.log(math.e * float(n) / tau) / (2 * tau))
        return bonus

    def __tau(self, r):
        return int(math.ceil((1 + self.alpha) ** r))

    def __set_arm(self, arm):
        """
        When choosing a new arm, make sure we play that arm for
        tau(r+1) - tau(r) episodes.
        """
        self.__current_arm = arm
        self.__next_update += max(1, self.__tau(self.r[arm] + 1) - self.__tau(self.r[arm]))
        self.r[arm] += 1

    def select_arm(self):
        n_arms = len(self.counts)

        # play each arm once
        for arm in range(n_arms):
            if self.counts[arm] == 0:
                self.__set_arm(arm)
                return arm

        # make sure we aren't still playing the previous arm.
        if self.__next_update > sum(self.counts):
            return self.__current_arm

        ucb_values = [0.0 for arm in range(n_arms)]
        total_counts = sum(self.counts)
        for arm in range(n_arms):
            bonus = self.__bonus(total_counts, self.r[arm])
            ucb_values[arm] = self.values[arm] + bonus

        chosen_arm = ind_max(ucb_values)
        self.__set_arm(chosen_arm)
        return chosen_arm

    def update(self, chosen_arm, reward):
        self.counts[chosen_arm] = self.counts[chosen_arm] + 1
        n = self.counts[chosen_arm]

        value = self.values[chosen_arm]
        new_value = ((n - 1) / float(n)) * value + (1 / float(n)) * reward
        self.values[chosen_arm] = new_value


class AnnealingEpsilonGreedy():
    def __init__(self, c, d, n_arms):
        self.c = c
        self.d = d
        self.n_arms = n_arms
        return

    def initialize(self):
        self.counts = [0 for col in range(self.n_arms)]
        self.values = [0.0 for col in range(self.n_arms)]
        return

    def select_arm(self):
        t = sum(self.counts) + 1
        epsilon = self.c * self.n_arms / (self.d**2 * t)
        if random.random() > epsilon:
            return ind_max(self.values)
        else:
            return random.randrange(len(self.values))

    def update(self, chosen_arm, reward):
        self.counts[chosen_arm] = self.counts[chosen_arm] + 1
        n = self.counts[chosen_arm]

        value = self.values[chosen_arm]
        new_value = ((n - 1) / float(n)) * value + (1 / float(n)) * reward
        self.values[chosen_arm] = new_value
        return