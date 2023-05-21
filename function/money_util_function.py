def rd_data(path):
    
    """
    データを読み込む。

    Args:
        path(str):データ読み込みのpath
    Returns:
        df(DataFrame):読み込んだデータ

    """
    
    df = spark.table(path).toPandas()
    
    print(f'{path}を読み込みます。')
    print(df.shape)
    print(df.info())
    
    display(df)
    
    print('読み込み終了。')
    print('*'*80)
    print('*'*80)
    
    return df


def detail_data(df):
    
    """
    データの統計値を出力する
    
    Args:
        df(DataFrame):統計値を出力するデータ

    """
    
    
    print('統計値を表示します。')
    
    # 統計値出力
    print(df.describe())
    
    print('統計値表示終了。')
    print('*'*80)
    print('*'*80)
    
    
    print('ユニーク値の詳細を表示します。')
    
    # ユニーク値出力
    for col_nm in df.columns:
        
        print(col_nm)
        print(df[col_nm].unique())
        print('*'*80)
        print('*'*80)
        
    print('ユニーク値詳細表示終了。')
        
    print('*'*80)
    print('*'*80)
    
        
    print('ユニーク値のカウント値を表示します。')
 
    # ユニーク値カウント出力    
    for col_nm in df.columns:
        
        print(col_nm)
        print(df[col_nm].value_counts())
        print('*'*80)
        print('*'*80)
        
    print('ユニーク値のカウント値表示終了。')
        
    print('*'*80)
    print('*'*80)
    
    
def confirm_null(df):
    
    """
    データのnull数を出力する
    
    Args:
        df(DataFrame):null数を出力するデータ

    """
    
    print('null値の個数を表示します。')
    
    # null数の出力
    print(df.isnull().sum())
    
    print('null値個数表示終了。')
    
    print('*'*80)
    print('*'*80)
    
def merging_train_test(train_df, test_df):
    
    """
    trainとtestをマージする
    
    Args:
        train_df(DataFrame):trainデータ
        test_df(DataFrame):testデータ
    Returns:
        train_testdf(DataFrame):train_testをマージしたデータ
    """
    
    
    print('train_testを結合します。')
    
    # train_testを結合する
    train_df['data_kind'] = 'train'
    test_df['data_kind'] = 'test'
    train_test_df = pd.concat([train_df, test_df], axis = 0)
    
    print(train_test_df.shape)
    display(train_test_df.head())
    print('train_testの結合が終了しました。')
    
    return train_test_df
  
def filling(df, col_nm, stat_kind):
    
    """
    空欄を埋める
    
    Args:
        df(DataFrame):空欄を埋めるデータ
        col_nm(str):空欄を埋める列
        stat_kind(str):空欄の埋め方 minだと列ごとのminで埋める
    Returns:
        df(DataFrame):空欄を埋めた後のデータ
    """

    # stat_kind毎に空欄を埋める
    if stat_kind == 'min':
        df[col_nm] = df[col_nm].fillna(df[col_nm].min())
    
    if stat_kind == 'max':
        df[col_nm] = df[col_nm].fillna(df[col_nm].max())
        
    if stat_kind == 'zero':
        df[col_nm] = df[col_nm].fillna(0)
        
    return df


def grouping_agg(df, group_col_nm, agg_col_nm, stat_kind):
    
    """
    グループごとの統計値列を作成する。
    
    Args:
        df(DataFrame):統計値を作るデータ
        group_col_nm(str):統計値を作るグループ
        agg_col_nm(str):統計値を計算する列
        stat_kind(str):統計値の計算方法
    Returns:
        df(DataFrame):統計値を作った後のデータ
    """
    
    # stat_kind毎に統計値を計算
    if stat_kind == 'max':
        df[agg_col_nm + '_' + stat_kind]=df.groupby(group_col_nm)[agg_col_nm].transform('max')
        
    if stat_kind == 'min':
        df[agg_col_nm + '_' + stat_kind]=df.groupby(group_col_nm)[agg_col_nm].transform('min')
        
    if stat_kind == 'mean':
        df[agg_col_nm + '_' + stat_kind]=df.groupby(group_col_nm)[agg_col_nm].transform('mean')
        
    if stat_kind == 'sum':
        df[agg_col_nm + '_' + stat_kind]=df.groupby(group_col_nm)[agg_col_nm].transform('sum')
    
    return df


def saving_data(df, file_name):
  
    """
    データを保存する。
    
    Args:
        df(DataFrame):保存するデータ
        file_name(str):保存データ名
    """

    print(f'{file_name}を保存します。')
    print(df.shape)
    print(df.columns.tolist())
    display(df.head())
    spark.createDataFrame(df).write.mode("overwrite").option("mergeSchema", "true").saveAsTable(file_name)

def fit_lgbm(x_tra, x_val, y_tra, y_val, params):

    """
    optunaでチューニングする。

    Args:
        x_tra(DataFrame):学習データの説明変数
        x_val(DataFrame):検証データの説明変数
        y_tra(DataFrame):学習データの目的変数
        y_val(DataFrame):検証データの目的変数
        prams(dict):学習パラメータ
    Returns:
        score(float):学習スコア
        model(model):モデル
        model.params(dict):最適モデルのパラメータ

    """
    
    oof_pred = np.zeros(len(y_val), dtype=np.float32)

    trains = lgbm_opt.Dataset(x_tra, y_tra)
    valids = lgbm_opt.Dataset(x_val, y_val)

    model = lgbm_opt.train(
        params, 
        trains,
        valid_sets = valids,
        num_boost_round = 10000,
        verbose_eval = False,
        early_stopping_rounds = 100
    )


    oof_pred = model.predict(x_val)
    score = roc_auc_score(y_val, oof_pred)

    print('*'*50)
    print('*'*50)
    print('*'*50)
    
    print(model.params)
    print('auc:', score)

    print('*'*50)
    print('*'*50)
    print('*'*50)


    return score, model, model.params

def visualize_importance(models, feat_train_df):
    
    """
    lightGBM の model 配列の feature importance を plot する
    CVごとのブレを box plot として表現します.
    
    args:
        models:
            List of lightGBM models
        feat_train_df:
            学習時に使った DataFrame
    """
    
    feature_importance_df = pd.DataFrame()
    for i, model in enumerate(models):
        _df = pd.DataFrame()
        _df['feature_importance'] = model.feature_importance()
        _df['column'] = feat_train_df.columns
        _df['fold'] = i + 1
        feature_importance_df = pd.concat([feature_importance_df, _df], axis=0, ignore_index=True)

    order = feature_importance_df.groupby('column')\
        .sum()[['feature_importance']]\
        .sort_values('feature_importance', ascending=False).index[:50]

    fig, ax = plt.subplots(figsize=(max(6, len(order) * .4), 7))
    sns.boxenplot(data=feature_importance_df, x='column', y='feature_importance', order=order, ax=ax, palette='viridis')
    ax.tick_params(axis='x', rotation=90)
    ax.grid()
    fig.tight_layout()
    return fig, ax

