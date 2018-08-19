import lgbm
import numpy as np

debug=True

def train_1st_stage(remove_list = None, postfix = '', param=None):
    print('param: {}'.format(param))
    if debug:
        n_est = 10
    else:
        n_est = 10000

    m = lgbm.LGBM(name='stack_1st_{}'.format(postfix), remove_prefix_list = remove_list, param=param, n_estimators=n_est)

    _, auc, train1, test1 = m.cv()

    print('{} : auc {}'.format(postfix, auc))

    np.save('train_{}'.format(postfix), train1)
    np.save('test_{}'.format(postfix), test1)


train_1st_stage(None, 'full')
train_1st_stage(['ins','p_','pos','b_','credit'], 'app')
train_1st_stage(['ins','pos','b_','credit'], 'app_prev')
train_1st_stage(['p_','pos','b_','credit'], 'app_ins')
train_1st_stage(['ins','p_','b_','credit'], 'app_pos')
train_1st_stage(['ins','p_','pos','credit'], 'app_bureau')
train_1st_stage(['ins','p_','pos','b_'], 'app_credit')
