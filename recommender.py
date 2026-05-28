#20212663 김재경
import sys
import subprocess

def _ensure(package, import_name=None):
    import importlib
    name = import_name or package
    try:
        importlib.import_module(name)
    except ImportError:
        print(f"[setup] Installing {package}...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])

_ensure('numpy')
_ensure('pandas')

import numpy as np
import pandas as pd


def load_base(file_name):
    data = pd.read_csv(file_name, sep='\t', header=None, names=['user_id', 'item_id', 'rating', 'timestamp'])
    return data


def load_test(file_name):
    data = pd.read_csv(file_name, sep='\t', header=None, names=['user_id', 'item_id', 'rating', 'timestamp'])
    return data


def create_user_item_table(data):
    """행=user_id, 열=item_id, 값=rating (미평가 항목은 NaN)"""
    return data.pivot_table(index='user_id', columns='item_id', values='rating')


def _cosine_sim(vectors, index):
    """벡터 행렬로부터 코사인 유사도 DataFrame을 반환한다."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    normalized = vectors / np.where(norms == 0, 1, norms)
    sim_matrix = normalized @ normalized.T
    np.fill_diagonal(sim_matrix, 1.0)
    return pd.DataFrame(sim_matrix, index=index, columns=index)


def compute_svd_similarity(user_item_table, global_mean, user_bias, item_bias, k=20, n_iter=10):
    """
    Iterative SVD로 user/item latent vector를 추출하고 PCC 유사도를 반환한다.

    ① baseline(u,i) = global_mean + bias_u + bias_i 로 결측 초기화
    ② SVD 수행 → 행렬 재구성
    ③ 원래 결측 위치만 재구성값으로 교체
    ④ ②~③을 n_iter회 반복 → 결측 추정이 점진적으로 정확해짐
    """
    users = user_item_table.index
    items = user_item_table.columns

    # baseline 행렬: (n_users, n_items)
    u_bias_arr = np.array([user_bias.get(u, 0.0) for u in users])
    i_bias_arr = np.array([item_bias.get(i, 0.0) for i in items])
    baseline = global_mean + u_bias_arr[:, None] + i_bias_arr[None, :]

    known_mask = ~user_item_table.isna().values
    known_values = np.where(known_mask, user_item_table.values, 0.0)

    # 결측 위치를 baseline으로 초기화
    filled = np.where(known_mask, known_values, baseline)

    for _ in range(n_iter):
        U, S, Vt = np.linalg.svd(filled, full_matrices=False)
        reconstructed = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
        # 실제 평가 위치는 원본 유지, 결측 위치만 재구성값으로 업데이트
        filled = np.where(known_mask, known_values, reconstructed)

    # 최종 SVD로 latent vector 추출 및 재구성 행렬 생성
    U, S, Vt = np.linalg.svd(filled, full_matrices=False)
    sqrt_s = np.sqrt(S[:k])

    user_vectors = U[:, :k] * sqrt_s
    item_vectors = (Vt[:k, :] * sqrt_s[:, None]).T

    user_sim = _cosine_sim(user_vectors, users)
    item_sim = _cosine_sim(item_vectors, items)

    recon = np.clip(U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :], 1.0, 5.0)
    recon_df = pd.DataFrame(recon, index=users, columns=items)

    return user_sim, item_sim, recon_df


def compute_biases(train_data, global_mean, lambda_u=25, lambda_i=25):
    """
    정규화된 user/item bias를 계산한다.
    평가 수가 적을수록 bias가 0(global mean)으로 수축되어 과적합을 방지한다.

    bias_u = Σ(r_ui - μ) / (n_u + λ)
    bias_i = Σ(r_ui - μ) / (n_i + λ)
    """
    user_bias = {}
    for uid, group in train_data.groupby('user_id'):
        n = len(group)
        user_bias[uid] = (group['rating'] - global_mean).sum() / (n + lambda_u)

    item_bias = {}
    for iid, group in train_data.groupby('item_id'):
        n = len(group)
        item_bias[iid] = (group['rating'] - global_mean).sum() / (n + lambda_i)

    return user_bias, item_bias


def predict_rating(user_id, item_id, user_item_table, similarity_df, user_means,
                   global_mean, user_bias, item_bias, k=30):
    """
    Bias-corrected mean-centered user-based CF 예측.
    baseline(u, i) = global_mean + bias_u + bias_i 를 기준으로
    이웃의 baseline 대비 편차를 가중 평균하여 예측한다.
    """
    if user_id not in user_item_table.index:
        return global_mean

    b_u = user_bias.get(user_id, 0.0)
    b_i = item_bias.get(item_id, 0.0)
    baseline_ui = global_mean + b_u + b_i

    if item_id not in user_item_table.columns:
        return float(np.clip(baseline_ui, 1.0, 5.0))

    item_col = user_item_table[item_id]
    raters = item_col.dropna().index.difference([user_id])
    raters = [u for u in raters if u in similarity_df.columns]

    if not raters:
        return float(np.clip(baseline_ui, 1.0, 5.0))

    sims = similarity_df.loc[user_id, raters]
    top = sims[sims > 0].nlargest(k)

    if top.empty:
        return float(np.clip(baseline_ui, 1.0, 5.0))

    # 이웃의 baseline 대비 편차를 가중 평균
    numerator = sum(
        top[v] * (user_item_table.loc[v, item_id] - (global_mean + user_bias.get(v, 0.0) + b_i))
        for v in top.index
    )
    denominator = top.abs().sum()

    if denominator == 0:
        return float(np.clip(baseline_ui, 1.0, 5.0))

    pred = baseline_ui + numerator / denominator
    return float(np.clip(pred, 1.0, 5.0))


def predict_rating_item(user_id, item_id, user_item_table, item_sim,
                        global_mean, user_bias, item_bias, k=30):
    """
    Item-based CF 예측.
    user_u가 평가한 아이템 중 item_i와 유사한 top-k를 찾아
    baseline 대비 편차를 가중 평균하여 예측한다.
    """
    b_u = user_bias.get(user_id, 0.0)
    b_i = item_bias.get(item_id, 0.0)
    baseline_ui = global_mean + b_u + b_i

    if user_id not in user_item_table.index or item_id not in item_sim.index:
        return float(np.clip(baseline_ui, 1.0, 5.0))

    # user_u가 실제로 평가한 아이템 (item_i 제외)
    user_row = user_item_table.loc[user_id].dropna()
    rated_items = user_row.index.difference([item_id])
    rated_items = [j for j in rated_items if j in item_sim.columns]

    if not rated_items:
        return float(np.clip(baseline_ui, 1.0, 5.0))

    sims = item_sim.loc[item_id, rated_items]
    top = sims[sims > 0].nlargest(k)

    if top.empty:
        return float(np.clip(baseline_ui, 1.0, 5.0))

    numerator = sum(
        top[j] * (user_item_table.loc[user_id, j] - (global_mean + b_u + item_bias.get(j, 0.0)))
        for j in top.index
    )
    denominator = top.abs().sum()

    if denominator == 0:
        return float(np.clip(baseline_ui, 1.0, 5.0))

    pred = baseline_ui + numerator / denominator
    return float(np.clip(pred, 1.0, 5.0))


def predict_rating_svd(user_id, item_id, recon_df, global_mean, user_bias, item_bias):
    """
    SVD 재구성 행렬에서 직접 예측값을 읽는다.
    학습 데이터에 없는 user/item은 baseline으로 fallback한다.
    """
    if user_id in recon_df.index and item_id in recon_df.columns:
        return float(recon_df.loc[user_id, item_id])
    b_u = user_bias.get(user_id, 0.0)
    b_i = item_bias.get(item_id, 0.0)
    return float(np.clip(global_mean + b_u + b_i, 1.0, 5.0))


def compute_rmse(predictions, actuals):
    """predictions, actuals: (user_id, item_id, rating) 튜플 리스트"""
    pred_map = {(u, i): r for u, i, r in predictions}
    sq_sum, count = 0.0, 0
    for u, i, actual in actuals:
        if (u, i) in pred_map:
            sq_sum += (pred_map[(u, i)] - actual) ** 2
            count += 1
    return (sq_sum / count) ** 0.5 if count > 0 else float('inf')


def main():
    train_file = sys.argv[1]
    test_file = sys.argv[2]

    train_data = load_base(train_file)
    test_data = load_test(test_file)

    print("Building user-item table...")
    user_item_table = create_user_item_table(train_data)
    user_means = user_item_table.mean(axis=1).to_dict()
    global_mean = train_data['rating'].mean()

    user_bias, item_bias = compute_biases(train_data, global_mean)

    print("Computing SVD-based similarities (iterative)...")
    user_sim, item_sim, recon_df = compute_svd_similarity(user_item_table, global_mean, user_bias, item_bias)

    # 앙상블 가중치 (합 = 1.0)
    alpha = 0.25   # user-based CF
    beta  = 0.45   # item-based CF
    gamma = 0.30   # SVD 재구성

    print("Predicting ratings...")
    output_file = train_file + '_prediction.txt'
    predictions = []
    with open(output_file, 'w') as f:
        for _, row in test_data.iterrows():
            uid = int(row['user_id'])
            iid = int(row['item_id'])
            pred_user = predict_rating(uid, iid, user_item_table, user_sim, user_means,
                                       global_mean, user_bias, item_bias)
            pred_item = predict_rating_item(uid, iid, user_item_table, item_sim,
                                            global_mean, user_bias, item_bias)
            pred_svd  = predict_rating_svd(uid, iid, recon_df, global_mean, user_bias, item_bias)
            pred = alpha * pred_user + beta * pred_item + gamma * pred_svd
            predictions.append((uid, iid, pred))
            f.write(f"{uid}\t{iid}\t{pred:.4f}\n")

    print(f"Saved to {output_file}")

    actuals = [(int(r['user_id']), int(r['item_id']), float(r['rating'])) for _, r in test_data.iterrows()]
    rmse = compute_rmse(predictions, actuals)
    print(f"RMSE: {rmse:.4f}")


if __name__ == '__main__':
    main()
