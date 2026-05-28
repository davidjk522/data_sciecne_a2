#20212663 김재경
import sys
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


def compute_svd_similarity(user_item_table, k=20):
    """
    SVD로 user latent vector를 추출한 뒤 코사인 유사도를 계산한다.
    잠재 공간에서 유사도를 계산하므로 순수 CF보다 희소성에 강하다.
    """
    user_means = user_item_table.mean(axis=1)
    # 결측값을 user 평균으로 채워 SVD 입력 행렬 구성
    filled = user_item_table.T.fillna(user_means).T.values

    U, S, Vt = np.linalg.svd(filled, full_matrices=False)
    # user latent vector: U @ sqrt(S) → (n_users, k)
    user_vectors = U[:, :k] * np.sqrt(S[:k])

    # 코사인 유사도
    norms = np.linalg.norm(user_vectors, axis=1, keepdims=True)
    normalized = user_vectors / np.where(norms == 0, 1, norms)
    sim_matrix = normalized @ normalized.T
    np.fill_diagonal(sim_matrix, 1.0)

    return pd.DataFrame(sim_matrix, index=user_item_table.index, columns=user_item_table.index)


def predict_rating(user_id, item_id, user_item_table, similarity_df, user_means, global_mean, k=30):
    """
    Mean-centered user-based CF 예측.
    유사도는 SVD 잠재 공간 기반 코사인 유사도를 사용한다.
    """
    if user_id not in user_item_table.index:
        return global_mean

    user_mean = user_means.get(user_id, global_mean)

    if item_id not in user_item_table.columns:
        return user_mean

    # 해당 item을 실제로 평가한 이웃 후보
    item_col = user_item_table[item_id]
    raters = item_col.dropna().index.difference([user_id])
    raters = [u for u in raters if u in similarity_df.columns]

    if not raters:
        return user_mean

    sims = similarity_df.loc[user_id, raters]
    top = sims[sims > 0].nlargest(k)

    if top.empty:
        return user_mean

    # mean-centered 가중 평균
    numerator = sum(
        top[v] * (user_item_table.loc[v, item_id] - user_means.get(v, global_mean))
        for v in top.index
    )
    denominator = top.abs().sum()

    if denominator == 0:
        return user_mean

    pred = user_mean + numerator / denominator
    return float(np.clip(pred, 1.0, 5.0))


def main():
    train_file = sys.argv[1]
    test_file = sys.argv[2]

    train_data = load_base(train_file)
    test_data = load_test(test_file)

    print("Building user-item table...")
    user_item_table = create_user_item_table(train_data)
    user_means = user_item_table.mean(axis=1).to_dict()
    global_mean = train_data['rating'].mean()

    print("Computing SVD-based user similarities...")
    similarity_df = compute_svd_similarity(user_item_table, k=20)

    print("Predicting ratings...")
    output_file = train_file + '_prediction.txt'
    with open(output_file, 'w') as f:
        for _, row in test_data.iterrows():
            uid = int(row['user_id'])
            iid = int(row['item_id'])
            pred = predict_rating(uid, iid, user_item_table, similarity_df, user_means, global_mean)
            f.write(f"{uid}\t{iid}\t{round(pred)}\n")

    print(f"Saved to {output_file}")


if __name__ == '__main__':
    main()
