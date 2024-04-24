"""
source: https://www.kaggle.com/code/ye11725/tfidf-lgbm-baseline-with-code-comments?scriptVersionId=172203959
"""
import os.path
import re
import numpy as np
import pandas as pd
import polars as pl
import lightgbm as lgb

from tqdm.auto import tqdm
from lightgbm import log_evaluation, early_stopping
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import cohen_kappa_score, accuracy_score


PARAGRAPH_FEATURES = ['paragraph_len', 'paragraph_sentence_cnt', 'paragraph_word_cnt']
SENTENCE_FEATURES = ['sentence_len', 'sentence_word_cnt']
WORD_FEATURES = ['word_len']


def removeHTML(x_input):
    html = re.compile(r'<.*?>')

    return html.sub(r'', x_input)


def data_preprocess(x_input):
    x_input = x_input.lower()
    x_input = removeHTML(x_input)
    x_input = re.sub("@\w+", '', x_input)
    x_input = re.sub("'\d+", '', x_input)
    x_input = re.sub("\d+", '', x_input)
    x_input = re.sub("http\w+", '', x_input)
    x_input = re.sub(r"\s+", " ", x_input)
    x_input = re.sub(r"\.+", ".", x_input)
    x_input = re.sub(r",+", ",", x_input)
    x_input = x_input.strip()

    return x_input


def preprocess_paragraph(data_df):
    data_df = data_df.explode('paragraph')
    data_df = data_df.with_columns(pl.col('paragraph').map_elements(data_preprocess))
    data_df = data_df.with_columns(pl.col('paragraph').map_elements(lambda x_: len(x_)).alias("paragraph_len"))
    data_df = data_df.with_columns(
        pl.col('paragraph').map_elements(lambda x_: len(x_.split('.'))).alias("paragraph_sentence_cnt"),
        pl.col('paragraph').map_elements(lambda x_: len(x_.split(' '))).alias("paragraph_word_cnt"),
    )

    return data_df


def preprocess_sentence(data_df):
    data_df = data_df.with_columns(pl.col('full_text').map_elements(data_preprocess).str.split(by=".").alias("sentence"))
    data_df = data_df.explode('sentence')
    data_df = data_df.with_columns(pl.col('sentence').map_elements(lambda x_: len(x_)).alias("sentence_len"))
    data_df = data_df.filter(pl.col('sentence_len') >= 15)
    data_df = data_df.with_columns(pl.col('sentence').map_elements(lambda x_: len(x_.split(' '))).alias("sentence_word_cnt"))

    return data_df


def preprocess_word(data_df):
    data_df = data_df.with_columns(pl.col('full_text').map_elements(data_preprocess).str.split(by=" ").alias("word"))
    data_df = data_df.explode('word')
    data_df = data_df.with_columns(pl.col('word').map_elements(lambda x_: len(x_)).alias("word_len"))
    data_df = data_df.filter(pl.col('word_len') != 0)

    return data_df


def feature_engineer(data_df, target):
    """
    段落特征：

    （1）essay的段落数，如paragraph_len_50表示该essay中长度大于50的段落数

    （2）essay的段落统计量，如paragraph_sentence_cnt_max表示该essay中句子数最多段落所包含的句子数量

    句子特征：

    （1）essay的句子数，类似essay的段落数

    （2）essay的句子统计量，essay的段落统计量

    单词特征：

    （1）essay的单词数，类似essay的段落数

    （2）word_len_qk

    :param data_df:
    :param target:
    :return:
    """
    agg_functions = []
    if target == "paragraph":
        agg_functions += [
            *[pl.col('paragraph').filter(pl.col('paragraph_len') >= i).count().alias(f"paragraph_{i}_cnt") for i in
              [50, 75, 100, 125, 150, 175, 200, 250, 300, 350, 400, 500, 600, 700]],
            *[pl.col('paragraph').filter(pl.col('paragraph_len') <= i).count().alias(f"paragraph_{i}_cnt") for i in
              [25, 49]],
        ]
        features = PARAGRAPH_FEATURES
    elif target == "sentence":
        agg_functions += [
            *[pl.col('sentence').filter(pl.col('sentence_len') >= i).count().alias(f"sentence_{i}_cnt") for i in
              [15, 50, 100, 150, 200, 250, 300]],
        ]
        features = SENTENCE_FEATURES
    elif target == "word":
        agg_functions += [
            *[pl.col('word').filter(pl.col('word_len') >= i + 1).count().alias(f"word_{i + 1}_cnt") for i in range(15)],
            pl.col('word_len').quantile(0.25).alias(f"word_len_q1"),
            pl.col('word_len').quantile(0.50).alias(f"word_len_q2"),
            pl.col('word_len').quantile(0.75).alias(f"word_len_q3"),
        ]
        features = WORD_FEATURES
    else:
        raise NotImplementedError()

    if target == "word":
        agg_functions += [
            *[pl.col(fea).max().alias(f"{fea}_max") for fea in features],
            *[pl.col(fea).mean().alias(f"{fea}_mean") for fea in features],
            *[pl.col(fea).std().alias(f"{fea}_std") for fea in features],
            *[pl.col(fea).first().alias(f"{fea}_first") for fea in features],
            *[pl.col(fea).last().alias(f"{fea}_last") for fea in features]
        ]
    else:
        agg_functions += [
            *[pl.col(fea).max().alias(f"{fea}_max") for fea in features],
            *[pl.col(fea).mean().alias(f"{fea}_mean") for fea in features],
            *[pl.col(fea).min().alias(f"{fea}_min") for fea in features],
            *[pl.col(fea).first().alias(f"{fea}_first") for fea in features],
            *[pl.col(fea).last().alias(f"{fea}_last") for fea in features]
        ]
    feature_df = data_df.group_by(['essay_id'], maintain_order=True).agg(agg_functions).sort("essay_id").to_pandas()

    return feature_df


def quadratic_weighted_kappa(y_true, y_pred):
    y_true = y_true + a
    y_pred = (y_pred + a).clip(1, 6).round()
    qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic")

    return 'QWK', qwk, True


def qwk_obj(y_true, y_pred):
    labels = y_true + a
    pred_score = y_pred + a
    pred_score = pred_score.clip(1, 6)
    f = 1 / 2 * np.sum((pred_score - labels) ** 2)
    g = 1 / 2 * np.sum((pred_score - a) ** 2 + b)
    df_ = pred_score - labels
    dg = pred_score - a
    grad = (df_ / g - f * dg / g ** 2) * len(labels)
    hess = np.ones(len(labels))

    return grad, hess


if __name__ == "__main__":
    DATA_DIR = "/Users/dream/myProjects/kaggle-competition-AES2/data/raw/learning-agency-lab-automated-essay-scoring-2"
    # 载入训练集和测试集，同时对full_text数据使用\n\n字符分割为列表，重命名为paragraph
    columns = [(pl.col("full_text").str.split(by="\n\n").alias("paragraph"))]
    data_train = pl.read_csv(os.path.join(DATA_DIR, "train.csv")).with_columns(columns)
    data_test = pl.read_csv(os.path.join(DATA_DIR, "test.csv")).with_columns(columns)
    # 段落特征工程
    train_features = feature_engineer(preprocess_paragraph(data_train), "paragraph")
    train_features['score'] = data_train['score']
    # 句子特征工程
    train_features = train_features.merge(
        feature_engineer(preprocess_sentence(data_train), "sentence"),
        on='essay_id', how='left'
    )
    # 单词特征工程
    train_features = train_features.merge(
        feature_engineer(preprocess_word(data_train), "word"),
        on='essay_id', how='left'
    )
    # TFIDF特征工程
    # 改进：
    vectorizer = TfidfVectorizer(
        tokenizer=lambda x_: x_,
        preprocessor=lambda x_: x_,
        token_pattern=None,
        strip_accents='unicode',
        analyzer='word',
        ngram_range=(1, 3),
        min_df=0.05,
        max_df=0.95,
        sublinear_tf=True,
    )
    # 将全部数据集都填充进TfidfVectorizer里，这可能会造成泄露和过于乐观的CV分数
    train_tfidf = vectorizer.fit_transform([i for i in data_train['full_text']])
    dense_matrix = train_tfidf.toarray()
    df = pd.DataFrame(dense_matrix)
    tfidf_columns = [f'tfidf_{i}' for i in range(len(df.columns))]
    df.columns = tfidf_columns
    df['essay_id'] = train_features['essay_id']
    train_features = train_features.merge(df, on='essay_id', how='left')
    feature_names = list(filter(lambda x_: x_ not in ['essay_id', 'score'], train_features.columns))

    a = 2.948
    b = 1.092
    models = []
    oof = []
    x = train_features
    y = train_features['score'].values
    k_fold = KFold(n_splits=5, random_state=42, shuffle=True)
    callbacks = [log_evaluation(period=25), early_stopping(stopping_rounds=75, first_metric_only=True)]
    # 训练模型
    for fold_id, (trn_idx, val_idx) in tqdm(enumerate(k_fold.split(x.copy(), y.copy().astype(str)))):
        model = lgb.LGBMRegressor(
            objective=qwk_obj,
            metrics='None',
            learning_rate=0.1,
            max_depth=5,
            num_leaves=10,
            colsample_bytree=0.5,
            reg_alpha=0.1,
            reg_lambda=0.8,
            n_estimators=1024,
            random_state=42,
            verbosity=- 1)
        # 分别取出5 fold分割的训练集和验证集
        X_train = train_features.iloc[trn_idx][feature_names]
        Y_train = train_features.iloc[trn_idx]['score'] - a
        X_val = train_features.iloc[val_idx][feature_names]
        Y_val = train_features.iloc[val_idx]['score'] - a
        print('\nFold_{} Training ================================\n'.format(fold_id + 1))

        lgb_model = model.fit(X_train,
                              Y_train,
                              eval_names=['train', 'valid'],
                              eval_set=[(X_train, Y_train), (X_val, Y_val)],
                              eval_metric=quadratic_weighted_kappa,
                              callbacks=callbacks, )
        pred_val = lgb_model.predict(
            X_val, num_iteration=lgb_model.best_iteration_)
        df_tmp = train_features.iloc[val_idx][['essay_id', 'score']].copy()
        df_tmp['pred'] = pred_val + a
        oof.append(df_tmp)
        models.append(model.booster_)
        lgb_model.booster_.save_model(f'fold_{fold_id}.txt')
    df_oof = pd.concat(oof)
    acc = accuracy_score(df_oof['score'], df_oof['pred'].clip(1, 6).round())
    kappa = cohen_kappa_score(df_oof['score'], df_oof['pred'].clip(1, 6).round(), weights="quadratic")
    print('acc: ', acc)
    print('kappa: ', kappa)

    # 在测试数据上测试
    test_features = feature_engineer(preprocess_paragraph(data_test), "paragraph")
    test_features = test_features.merge(
        feature_engineer(preprocess_sentence(data_test), "sentence"),
        on='essay_id', how='left'
    )
    test_features = test_features.merge(
        feature_engineer(preprocess_word(data_test), "word"),
        on='essay_id', how='left'
    )

    test_tfidf = vectorizer.transform([i for i in data_test['full_text']])
    dense_matrix = test_tfidf.toarray()
    df = pd.DataFrame(dense_matrix)
    tfidf_columns = [f'tfidf_{i}' for i in range(len(df.columns))]
    df.columns = tfidf_columns
    df['essay_id'] = test_features['essay_id']
    test_features = test_features.merge(df, on='essay_id', how='left')
    feature_names = list(filter(lambda x_: x_ not in ['essay_id', 'score'], test_features.columns))

    prediction = test_features[['essay_id']].copy()
    prediction['score'] = 0
    pred_test = models[0].predict(test_features[feature_names]) + a
    for i in range(4):
        pred_now = models[i + 1].predict(test_features[feature_names]) + a
        pred_test = np.add(pred_test, pred_now)

    pred_test = pred_test / 5
    pred_test = pred_test.clip(1, 6).round()
    prediction['score'] = pred_test
    prediction.to_csv('submission.csv', index=False)
