import docplex.mp.model as cpx

# Given parameters

# Large number
large_number = 999999.0

# m - number of firms
m = 6

# k - number of orders
# k = 2
k = 4

# holding_cost[k] - unit holding cost of order k
# holding_cost = [1.0, 1.0]
holding_cost = [1.0, 1.0, 1.0, 1.0]

# delay_penalty[k] - unit penalty cost for delayed order delivery
# delay_penalty = [2.0, 2.0]
delay_penalty = [2.0, 2.0, 2.0, 2.0]

# process_time[k][m] - unit processing time of order k at firm m
# process_time = [[1.0, 1.5, 2.2, 2.4, 1.7, 1.8],
#                 [1.0, 1.5, 2.2, 2.4, 1.7, 1.8]]
process_time = [[1.0, 1.5, 2.2, 2.4, 1.7, 1.8],
                [1.0, 1.5, 2.2, 2.4, 1.7, 1.8],
                [1.0, 1.5, 2.2, 2.4, 1.7, 1.8],
                [1.0, 1.5, 2.2, 2.4, 1.7, 1.8]]

# quantity[k] - quantity of order k
# quantity = [10, 10]
quantity = [10, 10, 10, 10]

# variable_cost[m] - given variable cost of firm m
variable_cost = [0.3, 0.2, 0.4, 0.5, 0.4, 0.3]

# fixed_cost[m] -  given fixed cost of firm n
fixed_cost = [95.1, 94.6, 94.8, 94.5, 94.4, 93.9]

# trsptt_time[m1][m2] - transportation time from firm m1 to firm m2
trsptt_time = [[0, 3.7, 3.8, 4.4, 4.5, 3.5],
               [4.4, 0, 3.3, 3.7, 4.2, 4.5],
               [4.2, 3.8, 0, 3.3, 4.5, 3.6],
               [3.1, 4.9, 4.6, 0, 3.7, 3.8],
               [4, 5, 3.3, 4.0, 3.6, 0, 4.2],
               [3, 7, 3.5, 4.9, 3.9, 4.5, 0]]

# due_time[k] - due time of order k
# due_time = [55.0, 55.0]
due_time = [75.0, 95.0, 75.0, 85.0]

# firm_start - the set of firms where production can start
firm_start = [0, 1]

# firm_end - the set of firms where production can end
firm_end = [4, 5]

# order_process[][] - processes needed for each type of product. In this case: 3 processes and 3 order_types
order_process = [[1, 1, 1],
                 [1, 0, 1],
                 [1, 1, 1]]

# order_network[][][] - network of each type of product (network presented by two-dimensional array)
order_network = [
    [
        [0, 0, 1, 1, 0, 0],
        [0, 0, 1, 1, 0, 0],
        [0, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0]
    ],
    [
        [0, 1, 1, 1, 1, 1],
        [1, 0, 1, 1, 1, 1],
        [1, 1, 0, 1, 1, 1],
        [1, 1, 1, 0, 1, 1],
        [1, 1, 1, 1, 0, 1],
        [1, 1, 1, 1, 1, 0]
    ],
    [
        [0, 1, 0, 1, 0, 1],
        [0, 0, 1, 1, 1, 0],
        [1, 0, 0, 1, 0, 1],
        [1, 1, 1, 0, 1, 1],
        [1, 1, 1, 1, 0, 0],
        [0, 1, 1, 0, 1, 0]
    ]
]

# order_type[k] - type of product of each order
# order_type = [0, 0]
order_type = [0, 0, 0, 0]

# process_firm[][] - which firm can conduct the processes
process_firm = [[0, 1], [2, 3], [4, 5]]


# ----------------------------------------------------------------------------
# Build the model
# ----------------------------------------------------------------------------

def build_model():
    # Create model
    opt_model = cpx.Model(name="MIP Model")

    # Decision parameters

    # process_start[k][m] - processing start time of order k
    process_start = opt_model.continuous_var_matrix(k, m, lb=0, ub=999, name="process_start_of_k%s_at_m%s")

    # process_start_hat[k][m] = process_start[k][m]*is_processsed_by[k][m]
    process_start_hat = opt_model.continuous_var_matrix(k, m, lb=0, ub=999, name="process_start_hat_of_k%s_at_m%s")

    # process_end[k][m] - processing end time of order k
    process_end = opt_model.continuous_var_matrix(k, m, lb=0, ub=999, name="process_end_of_k%s_at_m%s")

    # process_end_hat[k][m] = process_end[k][m]*is_processsed_by[k][m]
    process_end_hat = opt_model.continuous_var_matrix(k, m, lb=0, ub=999, name="process_end_hat_of_k%s_at_m%s")

    # production_start[k] - production start time of order k
    production_start = opt_model.continuous_var_list(k, lb=0, ub=999, name="production_start")

    # production_end[k] - production end time of order k
    production_end = opt_model.continuous_var_list(k, lb=0, ub=999, name="production_end")

    # is_processed_by[k][m] - equals 1 if order k is processed at manufacturer m, else 0
    is_processed_by = opt_model.binary_var_matrix(k, m, name="k%s_is_processed_by_m%s")

    # is_processed_by_hat[k][m1][m2] = is_processed_by[k][m1] * is_processed_by[k][m2]
    is_processed_by_hat = opt_model.binary_var_cube(k, m, m, name="k%s_is_processed_by_(hat)_m%s_and_m%s")

    # is_delayed[k] - equals 1 if the production ends after the required delivery time, else 0
    is_delayed = opt_model.binary_var_list(k, name="k%s_is_delayed")

    # is_delayed_hat[k] = is_delayed[k] * production_end[k]
    is_delayed_hat = opt_model.continuous_var_list(k, name="k%s_is_delayed_hat")

    # is_passed_from[k][m1][m2] - equals 1 if order k is passed from firm m1 to firm m2
    is_passed_from = opt_model.binary_var_cube(keys1=k, keys2=m, keys3=m, name="k%s_is_passed_from_m%s_to_m%s")

    # is_processed_straight_before[k1][k2][m] - equals 1 if order k2 is the next order processed after order k1 at firm m, else 0
    is_processed_straight_before = opt_model.binary_var_cube(keys1=k, keys2=k, keys3=m,
                                                            name="k%s_is_processed_straight_before_k%s_at_m%s")

    # is_merged_with[k1][k2][m] - equals 1 if order k2 starts immediately after order k1 at firm m
    is_merged_with = opt_model.binary_var_cube(keys1=k, keys2=k, keys3=m, name="k%s_is_merged_with_k%s_at_m%s")

    # is_merged_with_hat[k1][k2][m] = is_merged_with[k1][k2][m] * is_processed_by[k1][m]
    is_merged_with_hat = opt_model.binary_var_cube(keys1=k, keys2=k, keys3=m,
                                                   name="k%s_is_merged_with_(hat)_k%s_at_m%s")

    # Objective function
    objective_function = opt_model.sum(
        quantity[i] * holding_cost[i] * (production_end[i] - production_start[i] -
                            opt_model.sum(is_processed_by[(i, j)] * process_time[i][j] * quantity[i]
                                          + opt_model.sum(
                                is_passed_from[(i, j, l)] * trsptt_time[j][l] for l in range(m))
                                          for j in range(m)))
        + quantity[i]*(holding_cost[i] + delay_penalty[i]) * (is_delayed_hat[i] - is_delayed[i] * due_time[i])
        + quantity[i]*holding_cost[i] * (due_time[i] - production_end[i])
        + opt_model.sum(variable_cost[j] * quantity[i] * is_processed_by[(i, j)] +
                        fixed_cost[j] * (
                                is_processed_by[(i, j)] - opt_model.sum(
                            is_merged_with_hat[(l, i, j)] for l in range(k)))
                        for j in range(m))
        for i in range(k))

    # opt_model.add_kpi(objective_function, 'Objective Function')
    opt_model.minimize(objective_function)

    # constraint #
    opt_model.add_constraints_(
        opt_model.sum(is_processed_by[(i, j)] for j in process_firm[p]) >= order_process[order_type[i]][p]
        for p in range(len(process_firm))
        for i in range(k)
    )

    # constraint #
    opt_model.add_constraints_(
        opt_model.sum(is_processed_by[(i, j)] for j in range(m)) == opt_model.sum(
            order_process[order_type[i]][p] for p in range(len(process_firm)))
        for i in range(k)
    )

    # constraint #12
    opt_model.add_constraints_(
        opt_model.sum(opt_model.sum(is_passed_from[(i, j, l)] for l in range(m)) for j in range(m)) ==
        opt_model.sum(is_processed_by[(i, j)] for j in range(m)) - 1
        for i in range(k))

    # constraint #13
    opt_model.add_constraints_(
        opt_model.sum(opt_model.sum(is_passed_from[(i, j, l)] for l in range(m)) for j in firm_start) == 1
        for i in range(k))

    # constraint #14
    opt_model.add_constraints_(
        opt_model.sum(opt_model.sum(is_passed_from[(i, l, j)] for l in range(m)) for j in firm_end) == 1
        for i in range(k))

    # constraint #15
    opt_model.add_constraints_(
        opt_model.sum(is_passed_from[(i, l, j)] for l in range(m)) == opt_model.sum(
            is_passed_from[(i, j, l)] for l in range(m)) for j in set(range(m)) - set(firm_start) - set(firm_end)
        for i in range(k)
    )

    # constraint #16
    opt_model.add_constraints_(
        process_end[(i, j)] + trsptt_time[j][l] + process_time[i][l] * quantity[i] <=
        process_end[(i, l)] + large_number * (1 - is_passed_from[(i, j, l)])
        for l in range(m) for j in range(m) for i in range(k))

    # constraint #17
    opt_model.add_constraints_(
        opt_model.sum(is_processed_by[(i, j)] for j in firm_start) == 1
        for i in range(k)
    )

    # constraint #18
    opt_model.add_constraints_(
        opt_model.sum(is_processed_by[(i, j)] for j in firm_end) == 1
        for i in range(k)
    )

    # constraint #21
    opt_model.add_constraints_(
        process_end[(i, j)] == process_start[(i, j)] + process_time[i][j] * quantity[i] for j in range(m) for i in
        range(k)
    )

    # constraint #22 part one
    f = lambda x, y: 0 if x == y else 1
    opt_model.add_constraints_(
        is_merged_with[(i, l, j)] <= f(i, l) for l in range(k) for i in range(k) for j in range(m)
    )

    # constraint #22 part two
    f = lambda x, y: 1 if x == y else 0
    opt_model.add_constraints_(
        is_merged_with[(i, l, j)] <= f(order_type[i], order_type[l]) for l in range(k) for i in range(k) for j in
        range(m)
    )

    # constraint #23
    opt_model.add_constraints_(
        opt_model.sum(is_merged_with[(i, l, j)] for i in range(k)) <= is_processed_by[(l, j)] for j in range(m) for l in
        range(k)
    )

    # constraint #24
    opt_model.add_constraints_(
        opt_model.sum(is_merged_with[(l, i, j)] for i in range(k)) <= is_processed_by[(l, j)] for j in range(m)
        for l in range(k)
    )

    # constraint #
    # opt_model.add_constraints_(
    #     opt_model.sum(opt_model.sum(is_merged_with[(i, l, j)] for l in range(k)) for i in range(k)) <=
    #     opt_model.sum(is_processed_by[(i, j)] for i in range(k)) - 1
    #     for j in range(m)
    # )

    # constraint #25
    opt_model.add_constraints_(
        process_start[(l, j)] >= process_start[(i, j)] + is_processed_by[(i, j)] * quantity[i] * process_time[i][j]
        - large_number * (1 - is_merged_with[(i, l, j)])
        for l in range(k) for j in range(m) for i in range(k)
    )

    # constraint #26
    opt_model.add_constraints_(
        process_start[(l, j)] <= process_start[(i, j)] + is_processed_by[(i, j)] * quantity[i] * process_time[i][j]
        + large_number * (1 - is_merged_with[(i, l, j)])
        for l in range(k) for j in range(m) for i in range(k)
    )

    # constraint #27
    f = lambda x, y: 0 if x == y else 1
    opt_model.add_constraints_(
        is_processed_straight_before[(i, l, j)] <= f(i, l) for l in range(k) for i in range(k) for j in range(m)
    )

    # constraint #28
    opt_model.add_constraints_(
        process_end[(i, j)] - large_number * (1 - is_processed_straight_before[(i, l, j)]) - process_start[(l, j)] <= 0
        for l in range(k) for j in range(m) for i in range(k)
    )

    # constraint #29
    opt_model.add_constraints_(
        opt_model.sum(is_processed_straight_before[(i, l, j)] for i in range(k)) <= is_processed_by[(l, j)] for j in
        range(m) for l in range(k)
    )

    # constraint #30
    opt_model.add_constraints_(
        opt_model.sum(is_processed_straight_before[(l, i, j)] for i in range(k)) <= is_processed_by[(l, j)] for j in
        range(m) for l in range(k)
    )

    # constraint #31
    opt_model.add_constraints_(
        opt_model.sum(opt_model.sum(is_processed_straight_before[(i, l, j)] for l in range(k)) for i in range(k)) ==
        opt_model.sum(is_processed_by[(i, j)] for i in range(k)) - 1
        for j in range(m)
    )

    # constraint #32
    opt_model.add_constraints_(
        is_processed_straight_before[(i, l, j)] >= is_merged_with[(i, l, j)]
        for l in range(k)
        for j in range(m)
        for i in range(k)
    )

    # constraint #38-1
    opt_model.add_constraints_(
        is_processed_by_hat[(i, j, l)] * order_network[order_type[i]][j][l] >= is_passed_from[(i, j, l)]
        for l in range(m)
        for j in range(m)
        for i in range(k)
    )

    # constraint #38-2
    opt_model.add_constraints_(
        is_processed_by_hat[(i, j, l)] - is_processed_by[(i, j)] <= 0
        for l in range(m)
        for j in range(m)
        for i in range(k)
    )

    # constraint #38-3
    opt_model.add_constraints_(
        is_processed_by_hat[(i, j, l)] - is_processed_by[(i, l)] <= 0
        for l in range(m)
        for j in range(m)
        for i in range(k)
    )

    # constraint #38-4
    opt_model.add_constraints_(
        is_processed_by[(i, j)] + is_processed_by[(i, l)] - is_processed_by_hat[(i, j, l)] <= 1
        for l in range(m)
        for j in range(m)
        for i in range(k)
    )

    # constraint #39
    opt_model.add_constraints_(
        production_start[i] == opt_model.sum(process_start_hat[(i, j)] for j in firm_start)
        for i in range(k)
    )

    # constraint #40
    opt_model.add_constraints_(
        production_end[i] == opt_model.sum(process_end_hat[(i, j)] for j in firm_end)
        for i in range(k)
    )

    # constraint #43-1
    opt_model.add_constraints_(
        process_start_hat[(i, j)] <= process_start[(i, j)] + large_number * (1 - is_processed_by[(i, j)])
        for j in range(m)
        for i in range(k)
    )

    # constraint #43-2
    opt_model.add_constraints_(
        process_start_hat[(i, j)] >= process_start[(i, j)] - large_number * (1 - is_processed_by[(i, j)])
        for j in range(m)
        for i in range(k)
    )

    # constraint #43-3
    opt_model.add_constraints_(
        process_end_hat[(i, j)] <= process_end[(i, j)] + large_number * (1 - is_processed_by[(i, j)])
        for j in range(m)
        for i in range(k)
    )

    # constraint #43-4
    opt_model.add_constraints_(
        process_end_hat[(i, j)] >= process_end[(i, j)] - large_number * (1 - is_processed_by[(i, j)])
        for j in range(m)
        for i in range(k)
    )

    # constraint #43-5
    opt_model.add_constraints_(
        process_start_hat[(i, j)] <= large_number * is_processed_by[(i, j)]
        for j in range(m)
        for i in range(k)
    )

    # constraint #43-6
    opt_model.add_constraints_(
        process_start_hat[(i, j)] >= -large_number * is_processed_by[(i, j)]
        for j in range(m)
        for i in range(k)
    )

    # constraint #43-7
    opt_model.add_constraints_(
        process_end_hat[(i, j)] <= large_number * is_processed_by[(i, j)]
        for j in range(m)
        for i in range(k)
    )

    # constraint #43-8
    opt_model.add_constraints_(
        process_end_hat[(i, j)] >= -large_number * is_processed_by[(i, j)]
        for j in range(m)
        for i in range(k)
    )

    # constraint #44-1
    opt_model.add_constraints_(
        is_merged_with_hat[(i, l, j)] - is_processed_by[(i, j)] <= 0
        for l in range(k)
        for j in range(m)
        for i in range(k)
    )

    # constraint #44-2
    opt_model.add_constraints_(
        is_merged_with_hat[(i, l, j)] - is_merged_with[(i, l, j)] <= 0
        for l in range(k)
        for j in range(m)
        for i in range(k)
    )

    # constraint #44-3
    opt_model.add_constraints_(
        is_processed_by[(i, j)] + is_merged_with[(i, l, j)] - is_merged_with_hat[(i, l, j)] <= 1
        for l in range(k)
        for j in range(m)
        for i in range(k)
    )

    # constraint #45-1
    opt_model.add_constraints_(
        is_delayed_hat[i] <= large_number * is_delayed[i]
        for i in range(k)
    )

    # constraint #45-2
    opt_model.add_constraints_(
        is_delayed_hat[i] >= -large_number * is_delayed[i]
        for i in range(k)
    )

    # constraint #45-3
    opt_model.add_constraints_(
        is_delayed_hat[i] <= production_end[i] + large_number * (1 - is_delayed[i])
        for i in range(k)
    )

    # constraint #45-4
    opt_model.add_constraints_(
        is_delayed_hat[i] >= production_end[i] - large_number * (1 - is_delayed[i])
        for i in range(k)
    )

    # constraint #46-1
    opt_model.add_constraints_(
        due_time[i] - production_end[i] <= large_number * (1 - is_delayed[i])
        for i in range(k)
    )

    # constraint #46-2
    opt_model.add_constraints_(
        due_time[i] - production_end[i] >= 0.001 - (0.001 + large_number) * is_delayed[i]
        for i in range(k)
    )



    return opt_model


def run_cplex():
    md = build_model()
    md.print_information()
    md.solve()
    md.report()
    print(md.solution)
    print(md.get_solve_status())
    print(md.get_statistics())


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    run_cplex()
