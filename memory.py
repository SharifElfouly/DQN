from collections import deque
import numpy as np

class Memory:

    def __init__(self, max_capacity):
        self.max_capacity = max_capacity
        self.memory = deque(maxlen = max_capacity)

    def add(self, experience):
        self.memory.append(experience)

    def size(self):
        return len(self.memory)

    def sample(self, n_samples):
        memory_size = len(self.memory)
        sample_indices = np.random.choice(memory_size, n_samples)
        samples = [self.memory[sample_indices[i]] for i in range(len(sample_indices))]
        return np.array(samples)

    def get_sars(self, samples):
        """
        get state action reward next state
        """
        states = []
        actions = []
        rewards = []
        next_states = []
        for sample in samples:
            states.append(sample[0])
            actions.append(sample[1])
            rewards.append(sample[2])
            next_states.append(sample[3])

        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states)
