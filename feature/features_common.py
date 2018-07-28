import pandas as pd


def make_agg_names(prefix, columns):
    return pd.Index([prefix + e[1] + "(" + e[0] + ")" for e in columns])


def group_by_1(df, by, target, agg_func, column_name, merge=True):
    if isinstance(by, str):
        by = [by]

    g = df.groupby(by)[target].agg([agg_func]).reset_index()
    g.columns = by + [column_name]

    if merge:
        df = pd.merge(df, g, on=by, how='left')
        return df
    else:
        return g


def aggregate(df, aggregations, target, prefix):
    agg = target.groupby('SK_ID_CURR').agg({**aggregations})
    agg.columns = make_agg_names(prefix, agg.columns.tolist())
    agg.reset_index(inplace=True)
    return pd.merge(df, agg, on='SK_ID_CURR', how='left')
