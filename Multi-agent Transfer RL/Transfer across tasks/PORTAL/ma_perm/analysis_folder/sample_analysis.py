import numpy as np
import random
from sklearn.mixture import GaussianMixture
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from openpyxl import Workbook
from analysis_folder.utils.MultiDimensionScaling import MDS

def train_ubm(data, n_components=64, n_init=1, verbose=2):
    """
    Train a GMM on the data to form an Universal Background Model,
    that will be later used to adapt per-policy means.

    Note: Hardcoded to use diagonal covariance matrices
        as otherwise computing will take too long.

    Parameters:
        data (np.ndarray): Array of shape (N, D), containing data
            from various policies to be used to create a model
            of a "general policy".
        n_components (int): Number of components in the UBM
        n_init (int): Fed to GaussianMixture
        verbose (int): Fed to GaussianMixture
    Returns:
        ubm (sklearn.mixture.GaussianMixture): Trained GMM model
    """
    ubm = GaussianMixture(
        n_components=n_components,
        covariance_type="diag",
        verbose=verbose,
        n_init=n_init,
        max_iter=300
    )
    ubm.fit(data)
    return ubm


def save_ubm(path, ubm, means, stds, trajectory_indeces=np.nan, **additional_items):
    """
    Save sklearn UBM GMM into a numpy arrays for
    easier transfer between sklearn versions etc.

    Parameters:
        path (str): Where to store the UBM
        ubm (sklearn.mixture.GaussianMixture): Trained GMM model
        means, stds (ndarray): Means and stds of variables to
            be stored along UBM for normalization purposes
        trajectory_indeces (ndarray): (num_policies, num_trajs)
            array, that tells which trajectories were used to train
            this UBM. Used when trajectories are sampled.
        **additional_items: Additional items that will be added to the numpy
            archive.
    """
    np.savez(
        path,
        ubm_means=ubm.means_,
        ubm_weights=ubm.weights_,
        # Probably no need to store all of these, but oh well
        ubm_covariances=ubm.covariances_,
        ubm_precisions=ubm.precisions_,
        ubm_precisions_cholesky=ubm.precisions_cholesky_,
        means=means,
        stds=stds,
        trajectory_indeces=trajectory_indeces,
        **additional_items
    )


def load_ubm(path):
    """
    Load UBM stored with save_ubm, returning
    GMM object and normalization vectors

    Parameters:
        path (str): Where to load UBM from
    Returns:
        ubm (sklearn.mixture.GaussianMixture): Trained GMM model
        means, stds (ndarray): Means and stds of variables to
            be stored along UBM for normalization purposes
    """
    data = np.load(path)
    n_components = data["ubm_means"].shape[0]
    cov_type = "diag" if data["ubm_covariances"].ndim == 2 else "full"
    ubm = GaussianMixture(n_components=n_components, covariance_type=cov_type)
    ubm.means_ = data["ubm_means"]
    ubm.weights_ = data["ubm_weights"]
    ubm.covariances_ = data["ubm_covariances"]
    ubm.precisions_ = data["ubm_precisions"]
    ubm.precisions_cholesky_ = data["ubm_precisions_cholesky"]
    means = data["means"]
    stds = data["stds"]
    return ubm, means, stds


def trajectories_to_supervector(states, ubm, relevance_factor=16):
    """
    Take a trained UBM and states visited by a policy and create
    a fixed-length supervector to represent that policy
    based on the data.

    Current implementation MAP-adapts UBM means to data and
    then concatenates all these means

    Parameters:
        states (np.ndarray): (N, D) array
        ubm (sklearn.mixture.GaussianMixture): Trained GMM model
        relevance_factor (int): Relevance factor from [2]
    Returns:
        np.ndarray of shape (M,): 1D and fixed-length vector
            representing policy that created the data
    """
    # Using the notation in [1]
    # Score each data point to all components,
    # get (N, K) matrix (K = number of components)
    state_probs = ubm.predict_proba(states)

    # n, or "how close each point is each component"
    # (K, )
    state_prob_sums = np.sum(state_probs, axis=0)

    # \alpha, or weight of how much means should be moved, per component
    # (K, )
    alpha = state_prob_sums / (state_prob_sums + relevance_factor)

    # \tilde x, or the new means based on state
    # they are like expectations, except weighted
    # by how probable it is they came from that centroid
    # (K, D)
    tilde_x = np.zeros_like(ubm.means_)
    # Do each component individually to make this bit easier
    # to read and save on memory
    for k in range(ubm.n_components):
        tilde_x[k] = np.sum(states * state_probs[:, k, None], axis=0) / (state_prob_sums[k] + 1e-6)

    # MAP-adapt means
    # (K, D)
    adapted_means = alpha[..., None] * tilde_x + (1 - alpha[..., None]) * ubm.means_

    # Create pi-vector (supervector) of means
    # (K * D, )
    pi_vector = adapted_means.ravel()

    return pi_vector


def adapted_gmm_distance(means1, means2, precisions, weights):
    """
    Calculate upper-bound of KL-divergence of two MAP-adapted
    GMMs, as in [3] equation (6).

    Parameters:
        means1 (ndarray): Array of (K, D) of adapted means
        means2 (ndarray): Array of (K, D) of adapted means
        precisions (ndarray): Array of (K, D), an inverse of a
            diagonal covariance matrix (1/Sigma)
        weights (ndarray): Array of (K,), weights of the
            components
    Returns
        distance (float): Upper-bound of KL-divergence for the
            two GMMs specified by the two mean-matrices
    """

    mean_diff = means1 - means2

    # We can get rid of the matrix operations
    # since precisions are diagonal
    dist = 0.5 * np.sum(weights * np.sum(mean_diff * mean_diff * precisions, axis=1))

    return dist

def construct_dataset(n):
    states = []
    for _ in range(3):
        state = np.random.rand(int(n), 64)
        states.append(state)
    states = np.array(states)
    np.save(f'states_each_task_{int(n)}.npy', states)
    states = states.reshape((-1, 64))
    np.save(f'states_all_task_{int(n)}.npy', states)
    print('constructing dataset finished')

def load_datas(file_paths):

    data = []
    for file_path in file_paths:
        data.append(np.load(file_path))
    names = []
    # file_name should be like 'sv_5m_vs_6m_20000.npz'
    for file_path in file_paths:
        file_name = file_path.split('/')[-1]
        file_name_splits = file_name.split('_')
        name = "_".join(file_name_splits[1:-1])
        names.append(name)
    return data, names

def evaluate_distance(file_paths, save_dir, target):
    """
    :param
    file_paths: policy_supervector filepath, e.g. 'xx/xx.npz'
    save_dir: dir to save
    target: the target to compare with
    """
    data, names = load_datas(file_paths)
    target_idx = -1
    for i, name in enumerate(names):
        if target == name:
            target_idx = i


    if target_idx == -1:
        print('no target env in env list')
        return

    covariances = data[target_idx]["covariances"]
    weights = data[target_idx]["weights"]

    env_covariances = covariances
    env_precisions = 1 / env_covariances
    env_weights = weights
    mean_shape = env_covariances.shape

    def distance_metric(policy_supervector_1, policy_supervector_2):
        means_1 = policy_supervector_1.reshape(mean_shape)
        means_2 = policy_supervector_2.reshape(mean_shape)
        return adapted_gmm_distance(means_1, means_2, env_precisions, env_weights)

    rows = []
    # n = len(data)
    # for i, d in enumerate(data):
    #     for j in range(i+1, n):
    #         print(f'{i}, {j}')
    #         temp_dist = distance_metric(data[i]["pivector"], data[j]["pivector"])
    #         print(f'distance between {names[i]} and {names[j]} is {temp_dist}')
    #         rows.append([names[i], names[j], temp_dist])

    for i, d in enumerate(data):
        # 选择课程时不考虑当前任务和目标任务
        one_row = [names[i], names[target_idx], distance_metric(data[i]["pivector"], data[target_idx]["pivector"])]
        rows.append(one_row)
    sorted_rows = sorted(rows, key = lambda x:x[2])
    sorted_rows.insert(0, ['subtask', 'maintask', 'distance'])
    wb = Workbook()
    sheet = wb.active
    for row in sorted_rows:
        sheet.append(row)
    # if os.path.exists(f'{save_dir}/distance.xlsx'):
    #     print(f"evaluate distance skipped, because file exists: {save_dir}/distance.xlsx")
    wb.save(f'{save_dir}/distance.xlsx')
    print("evaluate distance finished")

    # get distance_matrix
    # distance_matrix = np.zeros((len(data), len(data)))
    # for i in range(len(data)):
    #     for j in range(len(data)):
    #         distance_matrix[i][j] = distance_metric(data[i]["pivector"], data[j]["pivector"])
    # return distance_matrix
    return sorted_rows[1:]



def save_everything(dir, raw_files):
    names = [raw_file.split('/')[-1][11:-4] for raw_file in raw_files]
    # names = [raw_file[:-4] for raw_file in raw_files]
    datas = []
    for data_path in raw_files:
        datat = np.load(os.path.join(dir, data_path))
        # for shape like (10000, 5, 64)
        # datat = np.mean(datat, axis=1)
        datas.append(datat)
    datas = np.array(datas)

    pre_dir = os.path.dirname(dir)
    data_all = np.concatenate(datas, axis=0)
    # mean, std of all state
    mean = data_all.mean(axis=0)
    std = data_all.std(axis=0)
    ubm = train_ubm(data_all)
    make_path(f'{pre_dir}/ubm/ubm.npz')
    save_ubm(f'{pre_dir}/ubm/ubm', ubm, mean, std)
    ubm, _, _ = load_ubm(f'{pre_dir}/ubm/ubm.npz')
    supervectors = []
    for states in datas:
        supervectors.append(trajectories_to_supervector(states, ubm))
    make_path(f"{pre_dir}/ubm/sv/check.npz")
    for sv, name in zip(supervectors, names):
        np.savez(f"{pre_dir}/ubm/sv/sv_{name}.npz",
                 pivector=sv,
                 covariances=ubm.covariances_,
                 weights=ubm.weights_
                 )

    print('save everything finished')

def evaluate_tsne(file_paths):
    """
    :param file_paths: policy_supervector filepath, e.g. 'policy_supervector_100000_0.npz'
    """
    data = []
    for file_path in file_paths:
        data.append(np.load(file_path))


    covariances = data[0]["covariances"]
    weights = data[0]["weights"]

    env_covariances = covariances
    env_precisions = 1 / env_covariances
    env_weights = weights
    mean_shape = env_covariances.shape

    env_policy_supervectors = []
    for sv in data:
        env_policy_supervectors.append(sv["pivector"])
    # env_policy_supervectors = np.stack(env_policy_supervectors)
    def distance_metric(policy_supervector_1, policy_supervector_2):
        means_1 = policy_supervector_1.reshape(mean_shape)
        means_2 = policy_supervector_2.reshape(mean_shape)
        return adapted_gmm_distance(means_1, means_2, env_precisions, env_weights)

    tsne = TSNE(metric=distance_metric)
    pi_points = tsne.fit_transform(env_policy_supervectors)
    color = [i/len(pi_points) for i in range(len(pi_points))]
    plt.scatter(pi_points[:, 0], pi_points[:, 1],
                c=color,
                # vmin=min(x.min() for x in env_rewards),
                # vmax=max(x.max() for x in env_rewards),
                cmap="plasma",
                )
    plt.legend()
    plt.show()

import os
def make_path(save_path):
    save_dir = os.path.dirname(save_path)
    if not os.path.exists(save_dir):
        recur_dir(save_dir)


def recur_dir(path):
    dir_path = os.path.dirname(path)
    if dir_path and not os.path.exists(dir_path):
        recur_dir(dir_path)
    os.mkdir(path)


def do_ubm(dir, raw_files, target_env):
    #  raw_data.npy to supervector.npz and save it in dir/ubm/supervector
    save_everything(dir, raw_files)
    pre_dir = os.path.dirname(dir)
    names = [raw_file.split('/')[-1][11:-4] for raw_file in raw_files]
    # names = [raw_file[:-4] for raw_file in raw_files]
    sv_paths = [f'{pre_dir}/ubm/sv/sv_{p}.npz' for p in names]
    save_dir = f"{pre_dir}/ubm"
    sorted_rows = evaluate_distance(file_paths=sv_paths, save_dir=save_dir, target=target_env)
    return sorted_rows

import argparse
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--policy', type=str, help='policy')
    parser.add_argument('--last_dir', type=str, help='dir')
    parser.add_argument('--base_dir', type=str, help='base_dir')
    parser.add_argument('--target_env', type=str, help='target env')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    policy = args.policy
    last_dir = args.last_dir
    base_dir = args.base_dir
    target_env = args.target_env
    dir = f"{base_dir}/{policy}/{last_dir}"
    data_files = os.listdir(dir)
    data_files = sorted(data_files, key = lambda x :os.path.getctime(os.path.join(dir, x)))

    sorted_rows = do_ubm(dir, data_files, target_env)
    exit(0)

