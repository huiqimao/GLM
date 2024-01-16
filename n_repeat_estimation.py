from poisson import generate_feature
from glm_parameter_estimation import read_sample_data, coef_estimation_n_times


def n_repeat_est(n):
    file_path = '/Users/huiqiangmao/PycharmProjects/GLM/data/'
    k = 5
    param_list = []
    param_err_list = []
    for i in range(n):
        print("Start to run " + str(i) + "th simulation")
        file_name = file_path + 'poisson_samples_' + str(i) + '.csv'
        generate_feature(1000, 10, file_name)
        data = read_sample_data(file_name)
        param, param_err = coef_estimation_n_times(data, k, 5)
        param_list.append(param)
        param_err_list.append(param_err)
    return param_list, param_err_list


if __name__ == "__main__":
    n = 3
    param_list, param_err_list = n_repeat_est(n)
    print('PARAM LIST: ', param_list)
    print('PARAM ERROR: ', param_err_list)