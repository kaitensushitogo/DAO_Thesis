import random


class Reality:
    def __init__(self, m):
        self.m = m
        self.vector = [random.randint(0, 1) for _ in range(self.m)]
