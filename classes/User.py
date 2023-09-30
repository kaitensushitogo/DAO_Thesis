import random
import math
from functions.others import *


class User:
    def __init__(self, reality, organization, m, k, ids, tokens):
        self.reality = reality
        self.organization = organization
        self.id = ids.pop(0)
        self.m = m
        self.k = k
        self.vector = generate_org_vector(reality)
        # [random.random() for _ in range(m)]
        self.utility = [generate_beliefs(0.5) for _ in range(m)]
        self.p = 0
        self.performance = update_user_performance(self, reality)
        self.token = tokens.pop(0)
        self.tokens_delegated = 0
        self.total_tokens_delegated = 0
        self.voted = False
        self.participated = False
        self.delegated = False
        self.changed = False
        self.whale = False
        self.vote_ctr = 0
        self.delegate_ctr = 0
        self.delegating = False
        self.delegator_id = None
        self.dele_size = 0
        self.dele_dur = 0

    def get_performance(self):
        cnt = 0
        for i in range(self.m):
            if self.vector[i] == self.reality.vector[i]:
                cnt += 1
        self.performance = cnt/self.m
        return self.performance

    # def check_interdependence(self, candidate, interdependence, vote_on):
    #     cnt = 0
    #    #print(interdependence[vote_on])
    #     for id in interdependence[vote_on]:    # [1,2,3,4,5] if k=5
    #        #print(self.vector[id], candidate.vector[id])
    #         if self.vector[id] == candidate.vector[id]:  # user1[1] == user7[1]
    #             cnt += 1

    #     if cnt == self.k:
    #        #print("manzoku")
    #         return True
    #     else:
    #         return False

        # cnt = 0
        # idxs = list(range(self.m))
        # idxs.pop(self.vote_on)
        # chosen_idxs = random.sample(idxs, self.k)

        # for idx in chosen_idxs:
        #     if self.vector[idx] == candidate.vector[idx]:
        #         cnt += 1
        # if self.k == cnt:
        #     return True
        # else:
        #     return False

    def search(self, users, delegators, vote_on, dele_size, vote_res, search_ratio, gas_fee):
        #print("USER PERFORMANCE: ", self.performance)
        # print("VOTE ON ATTR#:", vote_on)
        self.vote_on = vote_on
        attr_0 = vote_res[0]
        attr_1 = vote_res[1]

        if self.vector[vote_on] == 0:
            attr_0 += self.token
        else:
            attr_1 += self.token

        # Calculate the Probability of Decisive Vote
        if sum(vote_res) == 0:
            if vote_res[self.vector[vote_on]] + self.token > vote_res[not(self.vector[vote_on])]:
                #print("0타이. 무조건 p=1.")
                prob_of_decisive = 1
            else:
                prob_of_decisive = 0
        else:
            if np.argmax(vote_res) != np.argmax([attr_0, attr_1]):
                #print("값아ㅣ 바뀌어서 p=1")
                prob_of_decisive = 1
            else:
                prob_of_decisive = 0

        # Calculate the Value of Outcome
        value_of_outcome = self.utility[vote_on]

        # Calculate the Cost of Voting
        cost_of_voting = gas_fee

        # Calculate the Probability of Voting
        self.p = (prob_of_decisive * value_of_outcome) - cost_of_voting
        #print("유저의 p는??", self.p)

        if self.delegated:
            #print("I am delegator")
            return self.vote(vote_on)
        elif self.whale:
            #print("I am whale")
            return self.vote(vote_on)
        elif self.delegating:
            #print("I am delegating")
            #print("dele_id는 그대로인가?", self.delegator_id)
            delegator = users[self.delegator_id]
            return self.delegate(vote_on, delegator)
        else:
            # Case of Probability of Voting
            if self.p > random.random():
                # print("투-표")
                return self.vote(vote_on)
            else:
                # print("검-색")
                search = random.sample(delegators, int(
                    len(delegators) * search_ratio))

                #print("검색 결과 위임 후보")
                # for candidate in search:
                #print(candidate.id, end=' | ')

                # # Exclude self from the searched list
                # if self in search:
                #     search.remove(self)

                max_performance = 0
                for s in search:
                    # if self.check_interdependence(s, self.organization.interdependence, vote_on):
                    if s.dele_size < dele_size:
                        if s.performance > max_performance:
                            #print("--> 위임검색 아이디: ", s.id,"위임 Performance:", s.performance)
                            self.delegator_id = s.id
                            max_performance = s.performance
                            max_id = s.id

                if self.performance < max_performance:
                    delegator = users[max_id]

                    # if delgatee is a voter, delagatee can be delegated.
                    if delegator.delegator_id is None:
                        self.delegating = True
                        delegator.dele_size += 1
                        #print("DELEGATE ID?: ", delegator.id)
                        #print("DELEGATE의 DELE_SIZE?: ", delegator.dele_size)
                        # print("위-임")
                        return self.delegate(vote_on, delegator)
                    # if delegator's dele_size is full, deleagtee should find alternatives.
                    elif users[delegator.id].dele_size < dele_size:
                        #print("DELE_SIZE FULL, FINDING ALTERNATIVE DELEGATE!")
                        idx = delegators.index(users[delegator.id])
                        # print(users[delegator.id].id)
                        # print(idx)
                        tmp_delegators = delegators
                        tmp_delegators[idx].performance = 0
                        self.search(users, tmp_delegators, vote_on,
                                    dele_size, vote_res, search_ratio, gas_fee)
                else:
                    return

    def vote(self, vote_on):
        #print("<<<<<<<< VOTE >>>>>>>>>>")
        self.voted = True
        self.participated = True
        self.vote_ctr += 1
        #print("self.token", self.token)
        return self.vector[vote_on], self.token

    def delegate(self, vote_on, delegator):
        #print(">>>>> DELEGATE <<<<<")
        self.participated = True
        self.delegate_ctr += 1

        #print("자 이제 dele 증가시킨다")
        self.dele_dur += 1

        delegator.delegated = True
        # initiate tokens_delagated and add new delgated tokens.
        #print("self.toekn", self.token)
        #print("delegator tokens delegated b4 ", delegator.tokens_delegated)
        delegator.tokens_delegated = self.token
        delegator.total_tokens_delegated += self.token

        #print("delegator tokens delegated af: ", delegator.tokens_delegated)
        #print('vote_on>????', vote_on)
        #print("-- delegator.vector[vote_on]", delegator.vector[vote_on])
        #print("-- delegator.tokens_delegated", delegator.tokens_delegated)
        return delegator.vector[vote_on], delegator.tokens_delegated
