import lgbm
import sys

if __name__ == "__main__":
    argc = len(sys.argv)

    n_seeds = 10 if argc == 1 else int(sys.argv[1])
    debug = 0 if argc < 3 else int(sys.argv[2])

    print('n_seeds:{}, debug:{}'.format(n_seeds, debug))

    for i in range(n_seeds):
        name = 'lgbm_seed{}'.format(i)

        m = lgbm.LGBM(name, lgb_seed=i, n_estimators=100 if debug else 10000)

        m.cv(save_oof='../stack/{}_'+name+'.npy')