# %%
# Import Classes and Functions
from functions.others import *
from functions.runner import *
from functions.generators import *
from classes.User import *
from classes.Organization import *
from classes.Reality import *
# Import Libraries
import numpy as np
import pandas as pd
import random
np.set_printoptions(4)
np.seterr(invalid='ignore')
random.seed(3)


"""
Variables
- rds: rounds
- v: number of votes in each round
- m : number of attributes
- n_u : number of users
- n_o : number of organizations
- n_l: number of leaders
- k : degree of interdependence
- p : participation rate
- t : total number of tokens
- wr: whale ratio (1 = no whale, 0.x = x% of whales)
- dr: delegate ratio
"""
# Setting Variables
params = {
    'rounds': [100],
    'vote_agendas': [3],
    'attributes': [100],
    'users': [100],
    'organizations': [1],
    'tokens': [100000000],

    'whale_ratio': [0],
    'participation': [0.3],
    'interdependence': [0],
    'delegate_size': [100],
    'delegation_duration': [1],
    'delegator_ratio': [0]
}

vote_df = pd.DataFrame()
dele_df = pd.DataFrame()
perf_df = pd.DataFrame()
part_df = pd.DataFrame()
infl_df = pd.DataFrame()
gini_df = pd.DataFrame()

for config in param_grid(params):
    rds = config.get('rounds')
    v = config.get('vote_agendas')
    m = config.get('attributes')
    n_u = config.get('users')
    n_o = config.get('organizations')
    t = config.get('tokens')
    wr = config.get('whale_ratio')
    p = config.get('participation')
    k = config.get('interdependence')
    ds = config.get('delegate_size')
    dd = config.get('delegation_duration')
    dr = config.get('delegator_ratio')

    print()
    print()
    print("Parameters")
    print("Whale ratio:", wr)
    print("k:", k)
    print("p:", p)
    print("ds:", ds)
    print("dd:", dd)
    print("dr:", dr)
    print()
    print()

    # Initiate Reality
    reality = generate_reality(m)

    # Initiate Organizations
    organizations = generate_organizations(m, reality, n_o)

    # Initiate Organization
    users, deles = generate_users(
        reality, organizations, n_u, m, k, p, t, wr, dr)

    # Run Simulation
    votes, delegations, participations, o_performances, u_performances, influencers, ginis = run_model(
        reality, organizations, users, deles, rds, v, ds, dd)

    # Mean Results
    mean_votes = mean_result(votes)
    mean_deles = mean_result(delegations)
    mean_operfs = mean_result(o_performances)
    mean_uperfs = mean_result(u_performances)
    mean_parts = mean_result(participations)
    mean_ginis = mean_result(ginis)
    mean_infls = mean_influencers(influencers, n_o, rds, v, c_index=0.05)

    plot_vote_dele_result(mean_votes, mean_deles, n_u,wr, p, k, ds, dd, dr)
    plot_operf_result(mean_operfs, wr, p, k, ds, dd, dr)
    plot_uperf_result(mean_uperfs, wr, p, k, ds, dd, dr)
    # plot_part_res(mean_parts, n_u, wr, p, k, ds, dd, dr)
    plot_gini_res(mean_ginis, wr, p, k, ds, dd, dr)

    # dele_df['p={}, k={}, size={}, dur={}, ratio={}'.format(
    #     p, k, ds, dd, dr)] = pd.Series(mean_deles)
    # vote_df['p={}, k={}, size={}, dur={}, ratio={}'.format(
    # #     p, k, ds, dd, dr)] = pd.Series(mean_votes)
    # perf_df['p={}, k={}, size={}, dur={}, ratio={}'.format(
    #     p, k, ds, dd, dr)] = pd.Series(mean_operfs)
    # part_df['p={}, k={}, size={}, dur={}, ratio={}'.format(
    #     p, k, ds, dd, dr)] = pd.Series(mean_parts)
    # gini_df['p={}, k={}, size={}, dur={}, ratio={}'.format(
    #     p, k, ds, dd, dr)] = pd.Series(mean_ginis)
    # infl_df['p={}, k={}, size={}, dur={}, ratio={}'.format(
    #     p, k, ds, dd, dr)] = pd.Series(mean_infls)

# vote_df.to_csv('./result/vote.csv')
# dele_df.to_csv('./result/dele.csv')
# perf_df.to_csv('./result/perf.csv')
# part_df.to_csv('./result/part.csv')
# infl_df.to_csv('./result/infl.csv')
# gini_df.to_csv('./result/gini.csv')

# %%
