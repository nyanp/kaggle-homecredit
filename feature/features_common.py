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


def aggregate(df, aggregations, target, prefix, key='SK_ID_CURR', count_column=None):
    assert len(df) > 0 and len(target) > 0
    agg = target.groupby(key).agg({**aggregations})
    agg.columns = make_agg_names(prefix, agg.columns.tolist())
    agg.reset_index(inplace=True)
    if count_column is not None:
        agg[count_column] = target.groupby(key).size()
    return pd.merge(df, agg, on=key, how='left')
