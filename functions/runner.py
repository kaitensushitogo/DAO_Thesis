from functions.others import *

"""
(5) Vote handler
- Sampling random voting targets from the number of m. NO repeatition.
"""


def vote_handler(organization, users, delegators, vote_list, dele_size, dele_duration, search_ratio, gas_fee):
    vote_ctr_sum_list = []
    dele_ctr_sum_list = []
    part_list = []
    org_perf_list = []
    usr_perf_list = []
    infl_list = []
    gini_list = []

    for vote_target in vote_list:
        vote_result, chosen_value = organization.initiate_vote_on(
            vote_target, users, delegators, dele_size, dele_duration, search_ratio, gas_fee)

        infl = organization.get_user_influence(vote_result, chosen_value)
        infl_list.append(infl)

        perf_before, perf_after = organization.change_org_attr(chosen_value)
        organization.change_usr_attr(perf_before, perf_after, chosen_value)

        vot_ctr_sum, dele_ctr_sum = organization.get_vote_ctrs()
        vote_ctr_sum_list.append(vot_ctr_sum)
        dele_ctr_sum_list.append(dele_ctr_sum)

        part = organization.get_participation_ctrs()
        part_list.append(part)

        org_perf = organization.get_performance()
        org_perf_list.append(org_perf)

        usr_perf = organization.get_usr_performance()
        usr_perf_list.append(usr_perf)

        gini = organization.get_gini_coefficient(users)
        gini_list.append(gini)

    return vote_ctr_sum_list, dele_ctr_sum_list, part_list, org_perf_list, usr_perf_list, infl_list, gini_list


def run_model(reality, organizations, users_list, dele_list, rds, v, dele_size, dele_duration, search_ratio, gas_fee):

    m = reality.m
    n_o = len(organizations)
    # Create empty info list
    votes_fn = []
    deles_fn = []
    parts_fn = []
    o_perfs_fn = []
    u_perfs_fn = []
    infls_fn = []
    ginis_fn = []

    for i in range(n_o):
        votes = []
        deles = []
        parts = []
        o_perfs = []
        u_perfs = []
        infls = []
        ginis = []

        # Initiate Vote
        for rd in range(rds):
            vote_list = generate_random_vote_list(m, v)

            vs, ds, ps, opfs, upfs, infs, gs = vote_handler(
                organizations[i], users_list[i], dele_list[i], vote_list, dele_size, dele_duration, search_ratio, gas_fee)

            votes.append(vs)
            deles.append(ds)
            parts.append(ps)
            o_perfs.append(opfs)
            u_perfs.append(upfs)
            infls.append(infs)
            ginis.append(gs)

        votes_fn.append(votes)
        deles_fn.append(deles)
        parts_fn.append(parts)
        o_perfs_fn.append(o_perfs)
        u_perfs_fn.append(u_perfs)

        infls_fn.append(infls)
        ginis_fn.append(ginis)

    return votes_fn, deles_fn, parts_fn, o_perfs_fn, u_perfs_fn, infls_fn, ginis_fn
