import random
K = 5
T = 5000
N = 600
random.seed(1)
means = [0.1, 0.3, 0.5, 0.7, 0.9]
random.shuffle(means)

ucb = UCB1(K)
uniExplo = UniformExploration(K, N)
ucb2 = UCB2(0.5, K) # alpha=0.5
epGreedy = AnnealingEpsilonGreedy(7, 0.2, K) # c=7, d=0.2

resultUCB1 = test_algorithm(ucb1, arms, 500, 5000)
resultepGreedy = test_algorithm(epGreedy, arms, 500, 5000)
resultUniExplo = test_algorithm(uniExplo, arms, 500, 5000)
resultUCB2 = test_algorithm(ucb2, arms, 500, 5000)