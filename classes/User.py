import random, math
from functions.others import *


class User:
    def __init__(self, reality, organization, m, k, ids, tokens):
        self.id = ids.pop(0)
        self.m = m
        self.k = k
        self.vector = [generate_beliefs(0.5) for _ in range(m)]
        self.utility = [random.random() for _ in range(m)]
        self.p = 0
        self.performance = get_performance(self, reality)
        self.token = tokens.pop(0)
        self.tokens_delegated = 0
        self.total_tokens_delegated = 0
        self.reality = reality
        self.organization = organization
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
    
    def check_interdependence(self, candidate):
        cnt = 0
        idxs = list(range(self.m))
        idxs.pop(self.vote_on)
        chosen_idxs = random.sample(idxs, self.k)

        for idx in chosen_idxs:
            if self.vector[idx] == candidate.vector[idx]:
                cnt += 1
        if self.k == cnt:
            return True
        else:
            return False
        
    def search(self, users, delegators, vote_on, dele_size, vote_res, search_ratio, gas_fee):
        #print("USER PERFORMANCE", self.performance)
        self.vote_on = vote_on
        
        # Calculate the Probability of Decisive Vote
        if np.argmax(vote_res) != np.argmax(vote_res[self.vector[vote_on]] + self.token):
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
            if self.p > random.random():
                self.vote(vote_on)
            else:
                search = random.sample(delegators, int(len(delegators)*search_ratio))
                    
                # Exclude self from the searched list
                if self in search:
                    search.remove(self)
                    
                max_reputation = 0
                for s in search:
                    if self.check_interdependence(s):
                        if s.dele_size < dele_size:
                            # Delegator's participation rate should be 1
                            if s.performance * 1 > max_reputation:
                                #print("--> 위임검색 아이디: ", s.id, "위임 명성:", s.performance * 1)
                                self.delegator_id = s.id
                                max_reputation = s.performance
                                max_id = s.id

                if self.performance < max_reputation:
                    delegator = users[max_id]

                    # if delgatee is a voter, delagatee can be delegated.
                    if delegator.delegator_id is None:
                        self.delegating = True
                        delegator.dele_size += 1
                        #print("DELEGATE ID?: ", delegator.id)
                        #print("DELEGATE의 DELE_SIZE?: ", delegator.dele_size)
                        return self.delegate(vote_on, delegator)
                    # if delegator's dele_size is full, deleagtee should find alternatives.
                    elif users[delegator.id].dele_size < dele_size:
                        #print("DELE_SIZE FULL, FINDING ALTERNATIVE DELEGATE!")
                        idx = delegators.index(users[delegator.id])
                        #print(users[delegator.id].id)
                        #print(idx)
                        tmp_delegators = delegators
                        tmp_delegators[idx].performance = 0
                        self.search(users, tmp_delegators, vote_on, dele_size, vote_res, search_ratio, gas_fee)
                else:
                    #print("=== p_yn is false 참여 안한다잉 -===")
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
