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

        # User Performance Test
        # print()
        # print("User Performance List")
        # for i in range(len(users)):
        #     print(users[i].id, users[i].performance)

        # Initiate Delegates
        # 고래는 위임 대상에서 제거하지 않음

        # Selection by Random
        dele_num = int(round(n_u * dr))
        delegates = random.sample(users, dele_num)
        delegates = delegates[:dele_num]

        # Selection by Utility(만들어보기!)

        # Selection by Performance

        # Selection by Reputation
        # dele_num = int(round(n_u * dr))
        # delegates = sorted(
        #     users, key=lambda user: user.performance, reverse=True)
        # delegates = delegates[:dele_num]

        # Delegators Performance Test
        # print()
        # print("Chosen Delegates List")
        # for i in range(len(delegates)):
        #     print(delegates[i].id, delegates[i].performance)

        user_list.append(users)
        delegate_list.append(delegates)
    return user_list, delegate_list
