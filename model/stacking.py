import lgbm
import numpy as np
import gc

debug=False

def train_1st_stage(remove_list = None, postfix = '', param=None, averaging=1):
    print('param: {}'.format(param))
    if debug:
        n_est = 10
    else:
        n_est = 10000

    if averaging == 1:
        m = lgbm.LGBM(name='stack_1st_{}'.format(postfix), remove_prefix_list = remove_list, param=param, n_estimators=n_est)
        _, auc, train1, test1 = m.cv()
        print('{} : auc {}'.format(postfix, auc))
        np.save('train_{}'.format(postfix), train1)
        np.save('test_{}'.format(postfix), test1)
    else:
        for i in range(averaging):
            m = lgbm.LGBM(name='stack_1st_{}_seed{}_dart'.format(postfix, i), remove_prefix_list=remove_list, param=param,
                          n_estimators=n_est, lgb_seed=i)
            _, auc, train1, test1 = m.cv()
            print('{} : auc {}'.format(postfix, auc))
            np.save('train_{}_seed{}_dart'.format(postfix,i), train1)
            np.save('test_{}_seed{}_dart'.format(postfix,i), test1)
            del m
            gc.collect()


train_1st_stage(None, 'full', averaging=20)
train_1st_stage(['ins','p_','pos','b_','credit'], 'app', averaging=20)

