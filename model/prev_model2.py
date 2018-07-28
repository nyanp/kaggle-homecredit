import prev_model
import pandas as pd
import numpy as np

# previous_applicationからの予測値をapplicationに特徴として加える
class PrevModel2(prev_model.PrevModel):
    def __init__(self, name, param=None, seed=None):
        super().__init__(name, 14, -1, 0.0, param, seed)

    def custom_features(self, x_train, x_test):
        x_all = pd.concat([x_train,x_test])

        def make_groupby(df, by, on_, postfix):
            clm_base = '{}_BY_{}'.format(on_, postfix)

            if clm_base not in df:
                g = df.groupby(by)[on_].mean().reset_index().rename(columns={on_: clm_base})
                df = pd.merge(df, g, on=by, how='left')
            df[clm_base + '_RATIO'] = df[on_] / df[clm_base]
            df[clm_base + '_DIFF'] = df[on_] - df[clm_base]
            # df.drop(clm_base, axis=1, inplace=True)
            return df

        x_all = make_groupby(x_all, ['ORGANIZATION_TYPE', 'NAME_CONTRACT_TYPE'], 'AMT_ANNUITY', 'ORG_CONTRACT')
        x_all = make_groupby(x_all, ['OCCUPATION_TYPE', 'NAME_CONTRACT_TYPE'], 'AMT_ANNUITY', 'ORG_CONTRACT')
        x_all = make_groupby(x_all, ['ORGANIZATION_TYPE', 'NAME_CONTRACT_TYPE'], 'AMT_CREDIT', 'ORG_CONTRACT')
        x_all = make_groupby(x_all, ['OCCUPATION_TYPE', 'NAME_CONTRACT_TYPE'], 'AMT_CREDIT', 'ORG_CONTRACT')

        print(x_train.shape)
        print(x_test.shape)

        x_train = x_all[~x_all.delay.isnull()]
        x_test = x_all[x_all.delay.isnull()].drop('delay',axis=1)

        print(x_train.shape)
        print(x_test.shape)

        return x_train, x_test


if __name__ == "__main__":
    m = PrevModel2(name='prevmodel2')
    m.cv()
