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
import warnings
np.set_printoptions(4)
np.seterr(invalid='ignore')
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
random.seed(3)


# Setting Variables
params = {
    'rounds': [500],
    'vote_agendas': [1],
    'attributes': [100],
    'users': [100],
    'organizations': [50],
    'tokens': [100000000],

    'whale_ratio': [0, 0.2],
    'interdependence': [0, 5, 10],
    'delegate_size': [100],
    'delegation_duration': [1, 10, 50],
    'delegator_ratio': [1],
    'search_ratio': [1, 0.5, 0.2],
    'gas_fee': [0]
}

vote_df = pd.DataFrame()
dele_df = pd.DataFrame()
perf_df = pd.DataFrame()
part_df = pd.DataFrame()
infl_df = pd.DataFrame()
gini_df = pd.DataFrame()
final_df = pd.DataFrame()

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
    sr = config.get('search_ratio')
    gf = config.get('gas_fee')

    # Initiate Reality
    reality = generate_reality(m, k)

    # Initiate Organizations
    organizations = generate_organizations(reality, m, k, n_o)

    # Initiate Organization
    users, deles = generate_users(
        reality, organizations, n_u, m, k, t, wr, dr)

    # Run Simulation
    votes, delegations, participations, o_performances, u_performances, influencers, ginis = run_model(
        reality, organizations, users, deles, rds, v, ds, dd, sr, gf)

    # Average Results
    mean_votes = mean_result(votes)
    mean_deles = mean_result(delegations)
    mean_operfs = mean_result(o_performances)

    # Draw Plots
    plot_vote_dele_result(mean_votes, mean_deles, n_u,
                          wr, k, ds, dd, dr, sr, gf)
    plot_operf_result(mean_operfs, wr, k, ds, dd, dr, sr, gf)

    # Save to CSV File
    data = {'whale': wr,
            'k': k,
            'pool_ratio': dr,
            'size': ds,
            'duration': dd,
            'search_ratio': sr,
            'gas_fee': gf,
            'performance': round(mean_operfs[-1], 4)}
    print(data)
    perf_df = pd.DataFrame([data])
    final_df = final_df.append(perf_df)

final_df.to_csv('./result/final_pf.csv')

# %%
