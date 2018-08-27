import lgbm
import sys
import gc

if __name__ == "__main__":
    argc = len(sys.argv)

    n_seeds = 10 if argc == 1 else int(sys.argv[1])
    debug = 0 if argc < 3 else int(sys.argv[2])
    sample = 0 if argc < 4 else int(sys.argv[3])

    print('n_seeds:{}, debug:{}, sample:{}'.format(n_seeds, debug, sample))

    for i in range(n_seeds):
        if sample > 0:
            name = 'lgbm_sample{}_seed{}'.format(sample, i)
        else:
            name = 'lgbm_seed{}'.format(i)

        gc.collect()

        m = lgbm.LGBM(name, lgb_seed=i, n_estimators=100 if debug else 10000, undersample=sample)

        m.cv(save_oof='../stack/{}_'+name+'.npy')

        del m
