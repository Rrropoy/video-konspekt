from concurrent.futures import ThreadPoolExecutor
from functools import partial

import numpy as np


def assign_scenes(vectors: np.ndarray, centers: np.ndarray) -> np.ndarray:
    squared_distances = (
        np.sum(centers * centers, axis=1)[None, :]
        - 2 * (vectors @ centers.T)
    )
    # The squared norm of a scene is constant across its candidate centers.
    return squared_distances.argmin(axis=1)


def assign_and_sum(
    vectors: np.ndarray, centers: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels = assign_scenes(vectors, centers)
    sums = np.zeros_like(centers)
    np.add.at(sums, labels, vectors)
    sizes = np.bincount(labels, minlength=len(centers))
    return labels, sums, sizes


def fit_kmeans(
    vectors: np.ndarray,
    count: int,
    seed: int,
    max_iterations: int = 100,
    *,
    executor: ThreadPoolExecutor | None = None,
    workers: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    # Use the same precision in both modes to limit differences from summation order.
    vectors = np.asarray(vectors, dtype=np.float64)
    distinct = np.unique(vectors, axis=0, return_index=True)[1]
    count = min(count, len(distinct))
    rng = np.random.default_rng(seed)
    centers = vectors[rng.choice(np.sort(distinct), size=count, replace=False)].copy()
    blocks = np.array_split(vectors, min(workers, len(vectors))) if executor else []
    previous_labels = None

    for _ in range(max_iterations):
        if executor:
            results = list(executor.map(partial(assign_and_sum, centers=centers), blocks))
            block_labels, block_sums, block_sizes = zip(*results, strict=True)
            labels = np.concatenate(block_labels)
            sums = np.sum(block_sums, axis=0)
            sizes = np.sum(block_sizes, axis=0)
        else:
            labels, sums, sizes = assign_and_sum(vectors, centers)

        if previous_labels is not None and np.array_equal(labels, previous_labels):
            return centers, labels
        occupied = sizes > 0
        centers[occupied] = sums[occupied] / sizes[occupied, None]
        previous_labels = labels

    if executor:
        labels = np.concatenate(
            list(executor.map(partial(assign_scenes, centers=centers), blocks))
        )
    else:
        labels = assign_scenes(vectors, centers)
    return centers, labels


def closest_scene(
    cluster_id: int, vectors: np.ndarray, centers: np.ndarray, labels: np.ndarray
) -> int | None:
    members = np.flatnonzero(labels == cluster_id)
    if not members.size:
        return None
    difference = vectors[members] - centers[cluster_id]
    distances = np.sum(difference * difference, axis=1)
    return int(members[distances.argmin()])


def select_representatives(
    vectors: np.ndarray,
    centers: np.ndarray,
    labels: np.ndarray,
    *,
    executor: ThreadPoolExecutor | None = None,
) -> list[int]:
    select = partial(closest_scene, vectors=vectors, centers=centers, labels=labels)
    groups = range(len(centers))
    representatives = executor.map(select, groups) if executor else map(select, groups)
    return sorted(index for index in representatives if index is not None)
