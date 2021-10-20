import pickle as pk
import matplotlib.pyplot as plt
import random
import numpy as np
from scipy import stats

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"],
    "font.size":20})
for st in  ['_8']:
    file = 'results/lambda'+st+'.pkl'
    with open(file, 'rb') as f:
        result = pk.load(f)
    # fake to delete
    # for key, val in result.items():
    #     initial = val['all_p_val'] 
    #     val['all_p_val'] =  [random.uniform(0, 1) for i in initial]
    #     val['mean_p_val'] = np.mean(val['all_p_val'] )
    find_best = {'method': '', 'value':None}
    not_best_method = []
    def is_better_than(a,b):
        if b is None:
            return True
        else:
            return b<a
    height = []
    x = []
    label = []
    count = 0
    delta = 1

    def key_to_label(key):
        label = key.split(':')[1]
        label = label.split('diff')[0]
        if label == '-1':
            return '$\lambda_t$'
        elif label == '-2':
            return '$\lambda_{const}$'
        elif label == '-3':
            return '$\lambda_{scaled}$'
        elif len(label) >4:
            return '$\lambda_{tuned}$'
        return label

    for key, val in result.items():
        print(key)
        mean_result = val['mean_p_val']
        if is_better_than(mean_result,find_best['value']):
            find_best['method'] = key
            find_best['value'] = mean_result
        height.append(mean_result)
        x.append(count)
        label.append(key_to_label(key))
        count +=delta
        not_best_method.append(key)
    not_best_method.remove(find_best['method'])
    print(not_best_method)
    print('Winning method', find_best)
    print('Checking significance test')
    print(result[find_best['method']]['all_p_val'])
    for key in not_best_method:
        print(key, 'vs', find_best['method'])
        result_sign = stats.wilcoxon(result[key]['all_p_val'],result[find_best['method']]['all_p_val'])
        if result_sign.pvalue < 0.05:
            print('Significant')
        else:
            print('nope')

    print(height)
    plt.bar(x, height, width=0.4, tick_label=label)
    plt.xlabel('$\lambda$')
    plt.ylabel('\% valid samples')
    plt.tight_layout()
   # plt.show()
    plt.savefig('lambda_experiment'+st+'.pdf')
    plt.close()