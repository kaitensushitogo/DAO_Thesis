# Import libraries
import numpy as np
import random
import matplotlib.pyplot as plt
from itertools import product

# FUNCTIONS


"""
(1) Generate random vectors
- Generate vectors based on the uniform distribution.
- After having knowledge/performance value first, then match user or organization's attribute value according to the value assigned
- so that we prevent the situation where the most of user/organization have its knowledge/performance of very near to 0.5.
"""


def generate_user_vector(organization):
    m = len(organization.vector)
    nums = list(range(m))
    random_perf = random.uniform(0, 1)
    idxs = random.sample(nums, int(np.floor(random_perf*m)))
    user_vector = [None] * m

    # match values between user and organization as random performance value
    for idx in idxs:
        user_vector[idx] = organization.vector[idx]
    # then unmatch the rest values
    for i in range(m):
        if user_vector[i] == None:
            user_vector[i] = int(not(organization.vector[i]))
    return user_vector


def generate_org_vector(reality):
    m = len(reality.vector)
    nums = list(range(m))
    random_perf = random.uniform(0, 1)
    idxs = random.sample(nums, int(np.floor(random_perf*m)))
    org_vector = [None] * m

    # match values between user and reality as random knowledge value
    for idx in idxs:
        org_vector[idx] = reality.vector[idx]
    # then unmatch the rest values
    for i in range(m):
        if org_vector[i] == None:
            org_vector[i] = int(not(reality.vector[i]))
    return org_vector


"""
(2) Calculators
- Performance calculator: Sum up the matches of vectors between Reality and Organization
- Knowledge calculator: Sum up the matches of vectors between User and Organization
- Whale calculator: Return the number of whales given n and wr
"""


def get_performance(organization, reality):
    cnt = 0
    for i in range(organization.m):
        if reality.vector[i] == organization.vector[i]:
            cnt += 1
    performance = cnt/organization.m
    return performance


# def calculate_whales(n, wr):
#     if wr == 0:
#         whale_number = 0
#     else:
#         whale_number = int(n*wr)
#     return whale_number


"""
(3) Distributing tokens
- This function wraws a random number in an uniform distribution.
- If wr(distribution rate) does not equals to 1, two groups of users hold different amount of tokens.
- 'wr' decides the size of the first group and that group owns (1-wr) tokens.
- '1-wr' of the remaining group owns the rest tokens(wr).
- Example) if wr=0.2, 20% of users owns 80% of tokens and the remaining 80% of users owns 20% of tokens.
"""


def distribute_tokens(n, t, wr):
    if wr == 0:
        return np.random.dirichlet(np.ones(n)) * t
    elif wr > 0.5:
        raise Exception("ValueError: wr should be less than 0.5.")
    elif n*wr < 1:
        raise Exception(
            "ValueError: n*wr should be larger than 1. There must be at least one whale.")
    elif wr == 1:
        raise Exception(
            "ValueError: wr must be less than 1. There cannot be all whales. ")
    else:
        # whale에게 1- wr %만큼 토큰 부여
        # a = np.random.dirichlet(np.ones(int(n*wr))) * ((1-wr)*t)
        # b = np.random.dirichlet(np.ones(int(n*(1-wr)))) * (wr*t)

        # whale에게 50% 토큰 부여
        a = np.random.dirichlet(np.ones(int(n*wr))) * (0.5*t)
        b = np.random.dirichlet(np.ones(int(n*(1-wr)))) * (0.5*t)
        return np.concatenate((a, b))


"""
(4) Generate a vote list
- Sampling random voting targets from the number of m. NO repeatition.
- Sampling leader's voting targets from leaders randomly chosen.
  They decide what to vote only if more than half of the leaders share the same attribute values.
"""


def generate_random_vote_list(m, v):
    vote_list = random.sample(list(range(m)), v)
    return vote_list


"""
(5) Mean results
- mean result: mean results of votes, delegations, knowledges, performances and participations
- mean influencers: mean results of influencer counts
"""


def mean_result(var):
    var = np.array(var)
    n_o = len(var)
    if n_o == 1:
        var = var.ravel()
        return var
    else:
        x = var[0].ravel()
        for i in range(1, len(var)):
            y = var[i].ravel()
            x = [x+y for x, y in zip(x, y)]
    res = np.array(x)/n_o
    return res


def mean_influencers(var, n_o, rds, v, c_index):
    org_cnts = []
    for i in range(n_o):
        rd_cnts = []
        for j in range(rds):
            for k in range(v):
                cnts = 0
                infs = var[i][j][k]
                for inf in infs:
                    if inf >= c_index:
                        cnts += 1
                rd_cnts.append(cnts)
        org_cnts.append(rd_cnts)
    res = np.array(org_cnts)
    tmp = res[0]
    for i in range(1, n_o):
        tmp += res[i]
    avg = tmp/n_o
    return avg


"""
(6) Plotting
"""


def plot_vote_dele_result(vote_res, dele_res, n_u, wr, p, k, dele_size, dele_duration, dele_ratio):
    plt.figure(figsize=(12, 6))
    plt.plot(vote_res, label='Vote', color='black', ls="--")
    plt.plot(dele_res, label='Delegate', color='black')
    plt.xlabel('Rounds')
    plt.ylabel('Counts')
    plt.ylim(0, n_u+5)
    plt.grid(axis='x', alpha=0.5, ls=':')
    plt.legend(loc='upper left')
    plt.savefig("./images/wr({wr})_p({p})_k({k})_size({dele_size})_duration({dele_duration})_ratio({dele_ratio})__vote_dele.png".format(
        wr=wr, p=p, k=k, dele_size=dele_size, dele_duration=dele_duration, dele_ratio=dele_ratio))


def plot_operf_result(perf_res, wr, p, k, dele_size, dele_duration, dele_ratio):
    plt.figure(figsize=(12, 6))
    plt.plot(perf_res, label='Performance',
             color='black', ls='dotted')
    plt.xlabel('Rounds')
    plt.ylabel('Counts')
    plt.ylim(0, 1.0)
    plt.grid(axis='x', alpha=0.5, ls=':')
    plt.legend(loc='upper left')
    plt.savefig("./images/wr({wr})_p({p})_k({k})_size({dele_size})_duration({dele_duration})_ratio({dele_ratio})__org_perf.png".format(
        wr=wr, p=p, k=k, dele_size=dele_size, dele_duration=dele_duration, dele_ratio=dele_ratio))


def plot_uperf_result(perf_res, wr, p, k, dele_size, dele_duration, dele_ratio):
    plt.figure(figsize=(12, 6))
    plt.plot(perf_res, label='Performance',
             color='black', ls='dotted')
    plt.xlabel('Rounds')
    plt.ylabel('Counts')
    plt.ylim(0, 1)
    plt.grid(axis='x', alpha=0.5, ls=':')
    plt.legend(loc='upper left')
    plt.savefig("./images/wr({wr})_p({p})_k({k})_size({dele_size})_duration({dele_duration})_ratio({dele_ratio})__user_perf.png".format(
        wr=wr, p=p, k=k, dele_size=dele_size, dele_duration=dele_duration, dele_ratio=dele_ratio))


def plot_part_res(res, n_u, wr, p, k, dele_size, dele_duration, dele_ratio):
    plt.figure(figsize=(12, 6))
    plt.plot(res, label='Participation', color='black')
    plt.xlabel('Rounds')
    plt.ylabel('Rate')
    plt.ylim(0, n_u+5)
    plt.grid(axis='x', alpha=0.5, ls=':')
    plt.legend(loc='upper left')
    plt.savefig("./images/wr({wr})_p({p})_k({k})_size({dele_size})_duration({dele_duration})_ratio({dele_ratio})__part_res.png".format(
        wr=wr, p=p, k=k, dele_size=dele_size, dele_duration=dele_duration, dele_ratio=dele_ratio))


def plot_infl_res(res, wr, p, k, dele_size, dele_duration, dele_ratio):
    plt.figure(figsize=(12, 6))
    plt.plot(res, label='Influencers', color='black')
    plt.xlabel('Rounds')
    plt.ylabel('Counts')
    plt.ylim(0, 10)
    plt.grid(axis='x', alpha=0.5, ls=':')
    plt.legend(loc='upper left')
    plt.savefig("./images/wr({wr})_p({p})_k({k})_size({dele_size})_duration({dele_duration})_ratio({dele_ratio})__infl_res.png".format(
        wr=wr, p=p, k=k, dele_size=dele_size, dele_duration=dele_duration, dele_ratio=dele_ratio))


def plot_gini_res(res, wr,  p, k, dele_size, dele_duration, dele_ratio):
    plt.figure(figsize=(12, 6))
    plt.plot(res, label='Gini Coefficient', color='black')
    plt.xlabel('Rounds')
    plt.ylabel('Counts')
    plt.ylim(0, 1)
    plt.grid(axis='x', alpha=0.5, ls=':')
    plt.legend(loc='upper left')
    plt.savefig("./images/wr({wr})_p({p})_k({k})_size({dele_size})_duration({dele_duration})_ratio({dele_ratio})__gini_res.png".format(
        wr=wr, p=p, k=k, dele_size=dele_size, dele_duration=dele_duration, dele_ratio=dele_ratio))


"""
(7) Others
- Generate paramter grids
- Get vote method 
"""


def param_grid(params):
    for vcomb in product(*params.values()):
        yield dict(zip(params.keys(), vcomb))
