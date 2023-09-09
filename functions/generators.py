from classes.Reality import *
from classes.Organization import *
from classes.User import *
from functions.others import *


def generate_reality(m, k):
    reality = Reality(m, k)
    return reality


def generate_organizations(reality, m, k, n_o):
    organizations = []
    for _ in range(n_o):
        organizations.append(Organization(reality, m, k))
    return organizations


def generate_users(reality, organizations, n_u, m, k, t, wr, dr):
    # Initiate Users
    user_list = []
    delegate_list = []

    for organization in organizations:
        tokens = list(distribute_tokens(n_u, t, wr))
        ids = list(range(n_u))
        whale_number = int(n_u * wr)
        users = []

        for _ in range(n_u):
            users.append(User(reality, organization, m, k, ids, tokens))

        for j in range(whale_number):  # range(n_u-whale_number, n_u)
            users[j].whale = True

         # Initiate Delegates
        #dele_num = int(round(n_u * dr))
        # 고래는 위임 대상에서 제거해야 하는지????????????????
        # Selection by Random
        # delegates = random.sample(users, dele_num)
        
        dele_num = int(round(n_u * dr))
        # Selection by Reputation
        delegates = sorted(users, key=lambda user: user.performance, reverse=True)
        delegates = delegates[:dele_num]
        # for d in delegates:
        #     print(d.id, d.performance, d.p)

        user_list.append(users)
        delegate_list.append(delegates)
    return user_list, delegate_list
