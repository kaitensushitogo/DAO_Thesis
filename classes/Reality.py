import random
from functions.others import *


class Reality:
    def __init__(self, m, k):
        self.m = m
        self.vector = [generate_beliefs(0.5) for _ in range(m)]
        self.interdependence = []
        for i in range(m):
            a = random.sample(range(m), k)
            while i in a:
                a = random.sample(range(m), k)
            self.interdependence.append(a)
