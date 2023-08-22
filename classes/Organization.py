# Import libraries
from functions.others import *


class Organization:
    def __init__(self, m, reality):
        self.m = m
        self.reality = reality
        self.vector = generate_org_vector(reality)
        self.performance = get_performance(self, reality)
        self.reality = reality
        self.changed = False

    def get_performance(self):
        cnt = 0
        for i in range(self.m):
            if self.vector[i] == self.reality.vector[i]:
                cnt += 1
        # renew performance here
        self.performance = cnt/self.m
        return self.performance

    def initiate_vote_on(self, vote_on, users, delegators, dele_size, dele_duration):
        self.users = users
        self.vote_on = vote_on
        self.changed = False
        vote_result = [0, 0]
        n_u = len(users)

        # generate random user order list
        user_in_order = []
        random_order = random.sample(range(n_u), n_u)
        for order in random_order:
            user_in_order.append(users[order])

        # re-initiate the values in every vote
        for user in users:
            user.p_yn = user.get_p_yn()
            user.voted = False
            user.participated = False
            user.delegated = False
            user.tokens_delegated = 0
            user.total_tokens_delegated = 0
            user.changed = False
            user.vote_ctr = 0
            user.delegate_ctr = 0

            # If dele_duration is full, start new search!
            if user.dele_dur == dele_duration:
                print("A!!!!!!!!! dele_duration full: ", user.dele_dur)
                user.delegating = False
                user.dele_dur = 0
                # delegation duration이 끝났으면 delegator의 size 하나를 줄인다.
                users[user.delegator_id].dele_size -= 1
                user.delegator_id = None

        for user in user_in_order:
            # Call vote function - here begins vote, search, and delegate
            print()
            print("====================================")
            print("USER ID: ", user.id)
            print("====================================")
            result = user.search(users, delegators, vote_on, dele_size)
            if result != None:
                vote_on_value, token = result
                vote_result[vote_on_value] += token
            print()
            print()
            print()
            print("VOTE RESULT: ", vote_result)

        if vote_result[0] > vote_result[1]:
            chosen_value = 0
        else:
            chosen_value = 1

        return vote_result, chosen_value

    def change_org_attr(self, chosen_value):
        current_value = self.vector[self.vote_on]
        if current_value != chosen_value:
            print("Organization Attr Changed!")
            self.changed = True
            self.vector[self.vote_on] = chosen_value
        perf_before = self.performance
        perf_after = self.get_performance()
        print("ORGANIZATION PERFORMANCE: ", perf_before, "->", perf_after)
        return perf_before, perf_after

    def change_usr_attr(self, org_perf_before, org_perf_after, chosen_value):
        # change user attributes only when organization's performance increased:
        if org_perf_after > org_perf_before:
            for user in self.users:
                if user.participated:
                    if user.vector[self.vote_on] != chosen_value:
                        user.vector[self.vote_on] = chosen_value
                # re-calculate user's performance
                user.get_performance()

        # #Change user attributes according to the result
        # for user in self.users:
        #     if user.participated:
        #         if user.vector[self.vote_on] != chosen_value:
        #             user.vector[self.vote_on] = chosen_value
        #     # re-calculate user's performance
        #     user.get_performance()

    def get_vote_ctrs(self):
        vote_ctr_sum = 0
        dele_ctr_sum = 0
        for user in self.users:
            vote_ctr_sum += user.vote_ctr
            dele_ctr_sum += user.delegate_ctr

        return vote_ctr_sum, dele_ctr_sum

    def get_participation_ctrs(self):
        p_cnt = 0
        for user in self.users:
            if user.participated:
                p_cnt += 1
        return p_cnt

    def get_usr_performance(self):
        perf_sum = 0
        n = len(self.users)
        for user in self.users:
            perf_sum += user.performance
        perf_avg = round(perf_sum/n, 4)
        return perf_avg

    def get_user_influence(self, vote_result, chosen_value):
        user_influences = []
        for user in self.users:
            if user.vector[self.vote_on] == chosen_value:
                if sum(vote_result) == 0:
                    user_influence = 0
                else:
                    user_influence = round(
                        user.total_tokens_delegated/sum(vote_result), 4)
            else:
                user_influence = 0
            user_influences.append(user_influence)
        return user_influences

    def get_gini_coefficient(self, users):
        tkns_list = []
        for user in users:
            if user.participated:
                if user.delegated:
                    user.total_tokens = user.token + user.total_tokens_delegated
                elif user.delegating:
                    user.total_tokens = 0
                else:
                    user.total_tokens = user.token
            else:
                user.total_tokens = 0 
                
            tkns_list.append(user.total_tokens)
        data = np.array(tkns_list)

        data = np.sort(data)
        n = len(data)

        lorenz_curve = np.cumsum(data) / np.sum(data)
        perfect_equality_curve = np.linspace(0, 1, n)
        area_between_curves = np.trapz(perfect_equality_curve - lorenz_curve, dx=1/n)
        gini_index = area_between_curves / 0.5

        # tkns_list = []
        # for user in users:
        #     tkns_list.append(user.total_tokens_delegated)
        # x = np.array(tkns_list)
        # x.sort()
        # total = 0
        # for i, xi in enumerate(x[:-1], 1):
        #     total += np.sum(np.abs(xi-x[i:]))
        # return total/(len(x)**2 * np.mean(x))

        return gini_index
        
        
       