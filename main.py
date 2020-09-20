import docplex.mp.model as cpx

# Given parameters

# Large number
large_number = 999999

# m - number of firms
m = 6

# k - number of orders
k = 2

# product_info[][] - information(processes needed) of each type of product
product_info = [[0, 1, 2],
                [0, 2]]

# holding_cost[k] - unit holding cost of order k
holding_cost = [1.0, 1.0]

# delay_penalty[k] - unit penalty cost for delayed order delivery
delay_penalty = [2.0, 2.0]

# process_time[k][m] - unit processing time of order k at firm m
process_time = [[1.0, 1.5, 2.2, 2.4, 1.7, 1.8],
                [1.0, 1.5, 2.2, 2.4, 1.7, 1.8]]

# quantity[k] - quantity of order k
quantity = [10, 10]

# variable_cost[m] - given variable cost of firm m
variable_cost = [0.3, 0.2, 0.4, 0.5, 0.4, 0.3]

# fixed_cost[m] -  given fixed cost of firm n
fixed_cost = [5.1, 4.6, 4.8, 4.5, 4.4, 3.9]

# trsptt_time[k][m1][m2] - transportation time from firm m1 to firm m2 for order k
trsptt_time = [[[0, 3.7, 3.8, 4.4, 4.5, 3.5],
                [4.4, 0, 3.3, 3.7, 4.2, 4.5],
                [4.2, 3.8, 0, 3.3, 4.5, 3.6],
                [3.1, 4.9, 4.6, 0, 3.7, 3.8],
                [4, 5, 3.3, 4.0, 3.6, 0, 4.2],
                [3, 7, 3.5, 4.9, 3.9, 4.5, 0]],
               [[0, 3.7, 3.8, 4.4, 4.5, 3.5],
                [4.4, 0, 3.3, 3.7, 4.2, 4.5],
                [4.2, 3.8, 0, 3.3, 4.5, 3.6],
                [3.1, 4.9, 4.6, 0, 3.7, 3.8],
                [4, 5, 3.3, 4.0, 3.6, 0, 4.2],
                [3, 7, 3.5, 4.9, 3.9, 4.5, 0]]]

# due_time[k] - due time of order k
due_time = [18, 18]

# firm_start - the set of firms where production can start
firm_start = [0, 1]

# firm_end - the set of firms where production can end
firm_end = [4, 5]

# order_type[k] - type of product of each order
order_type = [0, 0]

# firm_process[m][] - processes that each firm can conduct
firm_process = [[0],
                [0],
                [1],
                [1],
                [2],
                [2]]

# Create model
opt_model = cpx.Model(name="MIP Model")

# Decision parameters

# process_start[k][m] - processing start time of order k
process_start = opt_model.continuous_var_matrix(k, m, lb=0, ub=None, name="process_start_of_k%s_at_m%")

# process_start_hat[k][m] - processing start time of order k
process_start_hat = opt_model.continuous_var_matrix(k, m, lb=0, ub=None, name="process_start_hat_of_k%s_at_m%")

# process_end[k][m] - processing end time of order k
process_end = opt_model.continuous_var_matrix(k, m, lb=0, ub=None, name="process_end_of_k%s_at_m%")

# process_end_hat[k][m] - processing start time of order k
process_end_hat = opt_model.continuous_var_matrix(k, m, lb=0, ub=None, name="process_end_hat_of_k%s_at_m%")

# production_start[k] - production start time of order k
production_start = opt_model.continuous_var_list(k, lb=0, ub=None, name="production_start")

# production_end[k] - production end time of order k
production_end = opt_model.continuous_var_list(k, lb=0, ub=None, name="production_end")

# is_processed_by[k][m] - equals 1 if order k is processed at manufacturer m, else 0
is_processed_by = opt_model.binary_var_matrix(k, m, name="k%s_is_processed_by_m%s")

# is_delayed[k] - equals 1 if the production ends after the required delivery time, else 0
is_delayed = opt_model.binary_var_list(k, name="k%s_is_delayed")

# is_delayed_hat[k] = is_delayed[k] * production_end[k]
is_delayed_hat = opt_model.binary_var_list(k, name="k%s_is_delayed_hat")

# is_passed_from[k][m1][m2] - equals 1 if order k is passed from firm m1 to firm m2
is_passed_from = opt_model.binary_var_cube(keys1=k, keys2=m, keys3=m, name="k%s_is_passed_from_m%s_to_m%s")

# is_merged_with[k1][k2][m] - equals 1 if order k2 starts immediately after order k1 at firm m
is_merged_with = opt_model.binary_var_cube(keys1=k, keys2=k, keys3=m, name="is_merged_with")

# is_merged_with_hat[k1][k2][m] = is_merged_with[k1][k2][m] * is_processed_by[k1][m]
is_merged_with_hat = opt_model.binary_var_cube(keys1=k, keys2=k, keys3=m, name="is_merged_with_hat")

# Objective function
objective_function = opt_model.sum(
    (holding_cost[i] * (process_end[i] - process_start[i] -
                        opt_model.sum((is_processed_by[i][j] * process_time[i][j] * quantity[i])
                                      + opt_model.sum(is_passed_from[i][j][l] * trsptt_time[i][j][l] for l in range(m))
                                      for j in range(m))))
    + (holding_cost[i] + delay_penalty[i]) * (is_delayed_hat[i] - is_delayed[i] * due_time[i])
    + holding_cost[i] * (due_time[i] - production_end[i])
    + opt_model.sum(variable_cost[j] * quantity[i] * is_processed_by[i][j] +
                    fixed_cost[j] * (
                            is_processed_by[i][j] - opt_model.sum(is_merged_with_hat[l][i][j] for l in range(k)))
                    for j in range(m))
    for i in range(k))

opt_model.add_kpi(objective_function, 'Objective Function')

# constraint #12
opt_model.add_constraints_(
    opt_model.sum(opt_model.sum(is_passed_from[i][j][l] for l in range(m)) for j in firm_start) == 1 for i in range(k))

# constraint #13
opt_model.add_constraints_(
    opt_model.sum(opt_model.sum(is_passed_from[i][l][j] for l in range(m)) for j in firm_end) == 1 for i in range(k))

# constraint #14
opt_model.add_constraints_(
    opt_model.add_constraints(opt_model.sum(is_passed_from[i][l][j] for l in range(m)) == opt_model.sum(
        is_passed_from[i][j][l] for l in range(m)) for j in set(range(m)) - set(firm_start) - set(firm_end)) for i
    in range(k)
)

# constraint #15
opt_model.add_constraints_(
    opt_model.add_constraints(
        opt_model.add_constraints(process_end[i][j] + trsptt_time[i][j][l] + process_time[i][l] * quantity[i] <=
                                  process_end[i][l] + large_number * (1 - is_passed_from[i][j][l])
                                  for l in range(m)) for j in range(m)) for i in range(k))

# constraint #16
opt_model.add_constraints_(
    opt_model.sum(is_processed_by[i][j] for j in firm_start) == 1 for i in range(k)
)

# constraint #17
opt_model.add_constraints_(
    opt_model.sum(is_processed_by[i][j] for j in firm_end) == 1 for i in range(k)
)

# constraint #20 part one
f = lambda x, y: 0 if x == y else 1
opt_model.add_constraints_(
    opt_model.add_constraints(
        opt_model.add_constraints(is_merged_with[i][l][j] <= f(i, l) for l in range(k)
                                  ) for i in range(k)
    ) for j in range(m)
)

# constraint #20 part two
f = lambda x, y: 1 if x == y else 0
opt_model.add_constraints_(
    opt_model.add_constraints(
        opt_model.add_constraints(is_merged_with[i][l][j] <= f(order_type[i], order_type[l]) for l in range(k)) for i in
        range(k)
    ) for j in range(m)
)

# constraint #21
opt_model.add_constraints_(
    opt_model.add_constraints(opt_model.sum(is_merged_with[i][l][j] for i in range(k)) <= is_processed_by[l][j] for j
                              in range(m))
    for l in range(k)
)

# constraint #22
opt_model.add_constraints_(
    opt_model.sum(opt_model.sum(is_merged_with[i][l][j] for l in range(k)) for i in range(k)) <=
    opt_model.sum(is_processed_by[i][j] for i in range(k)) - 1 for j in range(m)
)

# constraint #23
opt_model.add_constraints_(
    opt_model.add_constraints(
        opt_model.add_constraints(
            process_start[l][j] >= process_start[i][j] + is_processed_by[i][j] * quantity[i] * process_time[
                i] - large_number * (1 - is_merged_with[i][l][j])
            for l in range(k)) for j in range(m)) for
    i in range(k)
)

# constraint #24
opt_model.add_constraints_(
    opt_model.add_constraints(
        opt_model.add_constraints(
            process_start[l][j] <= process_start[i][j] + is_processed_by[i][j] * quantity[i] * process_time[
                i] + large_number * (1 - is_merged_with[i][l][j])
            for l in range(k)) for j in range(m)) for
    i in range(k)
)












# def print_hi(name):
#     # Use a breakpoint in the code line below to debug your script.
#     print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.
#
#
# # Press the green button in the gutter to run the script.
# if __name__ == '__main__':
#     print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
