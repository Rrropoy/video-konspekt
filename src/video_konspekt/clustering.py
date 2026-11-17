import numpy as np


def assign_scenes(vectors: np.ndarray, centers: np.ndarray) -> np.ndarray:
    squared_distances = (
        np.sum(centers * centers, axis=1)[None, :]
        - 2 * (vectors @ centers.T)
    )
    # The squared norm of a scene is constant across its candidate centers.
    return squared_distances.argmin(axis=1)


def fit_kmeans(
    vectors: np.ndarray, count: int, seed: int, max_iterations: int = 100
) -> tuple[np.ndarray, np.ndarray]:
    distinct = np.unique(vectors, axis=0, return_index=True)[1]
    count = min(count, len(distinct))
    rng = np.random.default_rng(seed)
    centers = vectors[rng.choice(np.sort(distinct), size=count, replace=False)].copy()
    previous_labels = None

    for _ in range(max_iterations):
        labels = assign_scenes(vectors, centers)
        if previous_labels is not None and np.array_equal(labels, previous_labels):
            break
        sums = np.zeros_like(centers)
        np.add.at(sums, labels, vectors)
        sizes = np.bincount(labels, minlength=count)
        occupied = sizes > 0
        centers[occupied] = sums[occupied] / sizes[occupied, None]
        previous_labels = labels

    return centers, assign_scenes(vectors, centers)


def select_representatives(
    vectors: np.ndarray, centers: np.ndarray, labels: np.ndarray
) -> list[int]:
    representatives = []
    for cluster_id, center in enumerate(centers):
        members = np.flatnonzero(labels == cluster_id)
        if members.size:
            difference = vectors[members] - center
            distances = np.sum(difference * difference, axis=1)
            representatives.append(int(members[distances.argmin()]))
    return sorted(representatives)
