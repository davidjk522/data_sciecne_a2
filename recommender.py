#20212663 김재경
import sys
import subprocess

def _ensure(package, import_name=None):
    # 패키지가 설치되어 있지 않으면 pip로 자동 설치한다
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
    # 학습 데이터를 탭 구분자로 읽어 DataFrame으로 반환한다
    data = pd.read_csv(file_name, sep='\t', header=None, names=['user_id', 'item_id', 'rating', 'timestamp'])
    return data


def load_test(file_name):
    # 테스트 데이터를 탭 구분자로 읽어 DataFrame으로 반환한다
    # 테스트 데이터의 rating은 예측에 사용하지 않고 RMSE 계산에만 쓴다
    data = pd.read_csv(file_name, sep='\t', header=None, names=['user_id', 'item_id', 'rating', 'timestamp'])
    return data


def create_user_item_table(data):
    """행=user_id, 열=item_id, 값=rating (미평가 항목은 NaN)"""
    # pivot_table을 통해 (user x item) 행렬을 만든다
    # 해당 user가 해당 item을 평가하지 않은 경우 NaN으로 남긴다
    return data.pivot_table(index='user_id', columns='item_id', values='rating')


def _cosine_sim(vectors, index):
    """벡터 행렬로부터 코사인 유사도 DataFrame을 반환한다."""
    # 각 벡터를 L2 norm으로 나눠 단위 벡터로 정규화한다
    # norm이 0인 경우(영벡터) 분모를 1로 처리해 ZeroDivisionError를 방지한다
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    normalized = vectors / np.where(norms == 0, 1, norms)

    # 정규화된 벡터끼리 내적 → 코사인 유사도 행렬 (n x n)
    # cos(θ) = u·v / (||u|| ||v||) = normalized_u · normalized_v (정규화 후 내적)
    sim_matrix = normalized @ normalized.T

    # 자기 자신과의 유사도는 정확히 1.0으로 고정 (부동소수점 오차 제거)
    np.fill_diagonal(sim_matrix, 1.0)
    return pd.DataFrame(sim_matrix, index=index, columns=index)


def compute_svd_similarity(user_item_table, global_mean, user_bias, item_bias, k=20, n_iter=10):
    """
    Iterative SVD로 user/item latent vector를 추출하고 PCC 유사도를 반환한다.

    ① baseline(u,i) = global_mean + bias_u + bias_i 로 missing value 초기화
    ② SVD 수행 → 행렬 재구성
    ③ 원래 missing value들을 재구성값으로 교체
    ④ ②~③을 n_iter회 반복 → missing value가 점진적으로 정확해짐
    """
    users = user_item_table.index
    items = user_item_table.columns

    # baseline 행렬: (n_users, n_items)
    # broadcasting을 활용해 모든 (u, i) 쌍의 baseline을 한 번에 계산한다
    # baseline(u,i) = μ + bias_u + bias_i
    u_bias_arr = np.array([user_bias.get(u, 0.0) for u in users])
    i_bias_arr = np.array([item_bias.get(i, 0.0) for i in items])
    baseline = global_mean + u_bias_arr[:, None] + i_bias_arr[None, :]

    # known_mask: 실제로 평가가 존재하는 위치는 True, missing value는 False
    known_mask = ~user_item_table.isna().values
    # known_values: 실제 rating은 그대로, missing value 자리는 0으로 채운다 (baseline으로 덮어쓸 것이라 무관)
    known_values = np.where(known_mask, user_item_table.values, 0.0)

    # missing value 위치를 baseline으로 초기화
    # 실제 평가 자리는 원본 rating을 유지하고, missing value 자리만 baseline으로 채운다
    filled = np.where(known_mask, known_values, baseline)

    for _ in range(n_iter):
        # SVD 수행: filled = U @ diag(S) @ Vt
        U, S, Vt = np.linalg.svd(filled, full_matrices=False)
        # 상위 k개의 singular value/vector만 사용해 low-rank 근사 행렬을 재구성한다
        # k가 클수록 정확하지만 노이즈도 포함될 수 있다
        reconstructed = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
        # 실제 평가 위치는 원본 유지, missing value 위치만 재구성값으로 업데이트
        # 반복할수록 missing value 추정이 점점 정확해진다
        filled = np.where(known_mask, known_values, reconstructed)

    # 최종 SVD로 latent vector 추출 및 재구성 행렬 생성
    U, S, Vt = np.linalg.svd(filled, full_matrices=False)

    # singular value를 square_root해서 user(U)와 item(Vt)에 균등 분배
    # SVD: A ≈ U @ diag(S) @ Vt
    #
    # singular value의 가중치를 user/item 양쪽에 균등하게 반영하기 위해 sqrt(S)를 사용한다:
    # user_vectors = U * sqrt(S)
    # item_vectors = Vt.T * sqrt(S)
    #
    # 두 벡터를 내적하면 원래 SVD 근사가 복원된다:
    # user_vectors @ item_vectors.T = U @ diag(S) @ Vt ≈ A
    sqrt_s = np.sqrt(S[:k])

    # user latent vector: (n_users, k) — 각 user의 잠재 선호도 표현
    user_vectors = U[:, :k] * sqrt_s
    # item latent vector: (n_items, k) — 각 item의 잠재 특성 표현
    item_vectors = (Vt[:k, :] * sqrt_s[:, None]).T

    # latent vector 간 코사인 유사도로 user-user / item-item 유사도 행렬을 만든다
    # 잠재 공간에서 유사도를 계산하므로 원본 행렬의 희소성(sparsity)에 강하다
    user_sim = _cosine_sim(user_vectors, users)
    item_sim = _cosine_sim(item_vectors, items)

    # 최종 재구성 행렬: MF 예측값으로 직접 사용한다 (rating 범위 [1,5]로 클리핑)
    reconstructed = np.clip(U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :], 1.0, 5.0)
    recon_df = pd.DataFrame(reconstructed, index=users, columns=items)

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
        # 분모에 λ를 더해 평가 수가 적은 user의 bias를 0 쪽으로 수축시킨다
        # 예: 평가 2개인 user는 bias가 크게 줄고, 평가 200개인 user는 거의 그대로
        user_bias[uid] = (group['rating'] - global_mean).sum() / (n + lambda_u)

    item_bias = {}
    for iid, group in train_data.groupby('item_id'):
        n = len(group)
        # user bias와 동일한 원리로 item bias를 정규화한다
        item_bias[iid] = (group['rating'] - global_mean).sum() / (n + lambda_i)

    return user_bias, item_bias


def predict_rating(user_id, item_id, user_item_table, similarity_df, user_means,
                   global_mean, user_bias, item_bias, k_neighbor=30):
    """
    Bias-corrected mean-centered user-based CF 예측.
    baseline(u, i) = global_mean + bias_u + bias_i 를 기준으로
    이웃의 baseline 대비 편차를 가중 평균하여 예측한다.
    """
    # 학습 데이터에 없는 user는 global mean으로 fallback
    if user_id not in user_item_table.index:
        return global_mean

    b_u = user_bias.get(user_id, 0.0)
    b_i = item_bias.get(item_id, 0.0)
    # baseline: user 성향과 item 인기도를 반영한 기준 예측값
    baseline_ui = global_mean + b_u + b_i

    # 학습 데이터에 없는 item은 baseline으로 fallback
    if item_id not in user_item_table.columns:
        return float(np.clip(baseline_ui, 1.0, 5.0))

    # item_id 열에서 실제 평가한 user들만 추출 (자기 자신 제외)
    item_col = user_item_table[item_id]
    raters = item_col.dropna().index.difference([user_id])
    raters = [u for u in raters if u in similarity_df.columns]

    # 해당 item을 평가한 이웃이 없으면 baseline으로 fallback
    if not raters:
        return float(np.clip(baseline_ui, 1.0, 5.0))

    # user_id와 각 이웃 간의 유사도를 가져와 양수인 것만 top-k 선택
    # 유사도가 음수인 이웃(취향 반대)은 예측을 왜곡하므로 제외한다
    sims = similarity_df.loc[user_id, raters]
    top = sims[sims > 0].nlargest(k_neighbor)

    if top.empty:
        return float(np.clip(baseline_ui, 1.0, 5.0))

    # 이웃의 baseline 대비 편차를 가중 평균
    # 편차 = 이웃의 실제 rating - 이웃의 baseline
    # 이를 통해 이웃이 "평균보다 얼마나 좋아했는가"를 반영한다
    # pred(u,i) = baseline(u,i) + Σ sim(u,v)*(r(v,i)-baseline(v,i)) / Σ|sim(u,v)|
    numerator = sum(
        top[v] * (user_item_table.loc[v, item_id] - (global_mean + user_bias.get(v, 0.0) + b_i))
        for v in top.index
    )
    denominator = top.abs().sum()

    if denominator == 0:
        return float(np.clip(baseline_ui, 1.0, 5.0))

    pred = baseline_ui + numerator / denominator
    # rating은 1~5 범위이므로 범위를 벗어난 예측값을 클리핑한다
    return float(np.clip(pred, 1.0, 5.0))


def predict_rating_item(user_id, item_id, user_item_table, item_sim,
                        global_mean, user_bias, item_bias, k_neighbor=30):
    """
    Item-based CF 예측.
    user_u가 평가한 아이템 중 item_i와 유사한 top-k를 찾아
    baseline 대비 편차를 가중 평균하여 예측한다.
    """
    b_u = user_bias.get(user_id, 0.0)
    b_i = item_bias.get(item_id, 0.0)
    # baseline: user 성향과 item 인기도를 반영한 기준 예측값
    baseline_ui = global_mean + b_u + b_i

    # 학습 데이터에 없는 user/item은 baseline으로 fallback
    if user_id not in user_item_table.index or item_id not in item_sim.index:
        return float(np.clip(baseline_ui, 1.0, 5.0))

    # user_u가 실제로 평가한 아이템 (item_i 제외)
    # 이 중에서 item_i와 유사한 아이템을 이웃으로 사용한다
    user_row = user_item_table.loc[user_id].dropna()
    rated_items = user_row.index.difference([item_id])
    rated_items = [j for j in rated_items if j in item_sim.columns]

    if not rated_items:
        return float(np.clip(baseline_ui, 1.0, 5.0))

    # item_id와 각 후보 아이템 간의 유사도를 가져와 양수인 것만 top-k 선택
    sims = item_sim.loc[item_id, rated_items]
    top = sims[sims > 0].nlargest(k_neighbor)

    if top.empty:
        return float(np.clip(baseline_ui, 1.0, 5.0))

    # user_u가 유사 아이템 j에 준 rating에서 baseline(u,j)를 빼 편차를 구하고
    # item 유사도로 가중 평균한다
    # pred(u,i) = baseline(u,i) + Σ sim(i,j)*(r(u,j)-baseline(u,j)) / Σ|sim(i,j)|
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
    # recon_df는 iterative SVD의 최종 재구성 행렬로
    # 전체 행렬 구조의 잠재 패턴을 반영한 전역적(global) 예측값이다
    # CF가 이웃 부족으로 예측이 약할 때 이를 보완하는 역할을 한다
    if user_id in recon_df.index and item_id in recon_df.columns:
        return float(recon_df.loc[user_id, item_id])
    # 학습 데이터에 없는 경우 baseline으로 fallback
    b_u = user_bias.get(user_id, 0.0)
    b_i = item_bias.get(item_id, 0.0)
    return float(np.clip(global_mean + b_u + b_i, 1.0, 5.0))


def compute_rmse(predictions, actuals):
    """predictions, actuals: (user_id, item_id, rating) 튜플 리스트"""
    # 예측값을 (user_id, item_id) 키로 빠르게 조회하기 위해 dict로 변환
    pred_map = {(u, i): r for u, i, r in predictions}
    sq_sum, count = 0.0, 0
    for u, i, actual in actuals:
        if (u, i) in pred_map:
            # RMSE = sqrt( Σ(pred - actual)² / n )
            sq_sum += (pred_map[(u, i)] - actual) ** 2
            count += 1
    return (sq_sum / count) ** 0.5 if count > 0 else float('inf')


def main():
    train_file = sys.argv[1]
    test_file = sys.argv[2]

    train_data = load_base(train_file)
    test_data = load_test(test_file)

    print("Building user-item table")
    user_item_table = create_user_item_table(train_data)
    # user_means: 각 user의 평균 rating (CF fallback에 사용)
    user_means = user_item_table.mean(axis=1).to_dict()
    # global_mean: 전체 평균 rating (cold-start fallback의 최후 수단)
    global_mean = train_data['rating'].mean()

    # 정규화된 user/item bias 계산
    # bias는 SVD 결측 초기화와 CF 예측의 baseline에 모두 사용된다
    user_bias, item_bias = compute_biases(train_data, global_mean)

    print("Computing SVD")
    # iterative SVD로 유사도 행렬(user_sim, item_sim)과 재구성 행렬(recon_df)을 동시에 생성
    user_sim, item_sim, recon_df = compute_svd_similarity(user_item_table, global_mean, user_bias, item_bias)

    # 앙상블 가중치 (합 = 1.0)
    # 세 예측값을 선형 결합해 최종 예측을 만든다
    alpha = 0.25   # user-based CF
    beta  = 0.45   # item-based CF
    gamma = 0.30   # SVD 재구성

    print("Predicting ratings")
    output_file = train_file + '_prediction.txt'
    predictions = []
    with open(output_file, 'w') as f:
        for _, row in test_data.iterrows():
            uid = int(row['user_id'])
            iid = int(row['item_id'])
            # 세 가지 예측값을 각각 계산한다
            pred_user = predict_rating(uid, iid, user_item_table, user_sim, user_means,
                                       global_mean, user_bias, item_bias)
            pred_item = predict_rating_item(uid, iid, user_item_table, item_sim,
                                            global_mean, user_bias, item_bias)
            pred_svd  = predict_rating_svd(uid, iid, recon_df, global_mean, user_bias, item_bias)
            # 앙상블: 가중 평균으로 최종 예측값을 산출한다
            pred = alpha * pred_user + beta * pred_item + gamma * pred_svd
            predictions.append((uid, iid, pred))
            f.write(f"{uid}\t{iid}\t{pred:.4f}\n")

    print(f"Saved to {output_file}")

    # 테스트 데이터의 실제 rating과 비교해 RMSE를 계산하고 출력한다
    actuals = [(int(r['user_id']), int(r['item_id']), float(r['rating'])) for _, r in test_data.iterrows()]
    rmse = compute_rmse(predictions, actuals)
    print(f"RMSE: {rmse:.4f}")


if __name__ == '__main__':
    main()
