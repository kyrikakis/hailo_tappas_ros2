import numpy as np
import json

def gallery_get_xtensor(matrix: np.ndarray) -> np.ndarray:
    return np.squeeze(matrix)

def gallery_one_dim_dot_product(array1: np.ndarray, array2: np.ndarray) -> float:
    if array1.ndim > 1 or array2.ndim > 1:
        raise ValueError("One of the arrays has more than 1 dimension")
    if array1.shape[0] != array2.shape[0]:
        raise ValueError("Arrays are with different shape")
    return np.sum(array1 * array2)

def get_distance(embeddings_queue: list[np.ndarray], matrix: np.ndarray) -> float:
    new_embedding = gallery_get_xtensor(matrix)
    max_thr = 0.0
    for embedding in embeddings_queue:
        thr = gallery_one_dim_dot_product(embedding, new_embedding)
        max_thr = max(thr, max_thr)
    return 1.0 - max_thr

def get_embeddings_distances(embeddings: list[list[np.ndarray]], matrix: np.ndarray) -> np.ndarray:
    distances = [get_distance(queue, matrix) for queue in embeddings]
    return np.array(distances)

def match_embedding(new_embedding: np.ndarray, embeddings: list[list[np.ndarray]], similarity_thr: float) -> int:
    if not embeddings:
        return 0  # New ID
    distances = get_embeddings_distances(embeddings, new_embedding)
    closest_id = np.argmin(distances)
    if distances[closest_id] > similarity_thr:
        return len(embeddings)  # New ID
    else:
        return closest_id

def load_gallery_from_json(file_path: str) -> tuple[list[list[np.ndarray]], list[str]]:
    with open(file_path, "r") as f:
        data = json.load(f)
    embeddings = []
    names = []
    for item in data:
        names.append(item['name'])
        embeddings.append([np.array(item['embedding'])]) #create the 2d array of embeddings
    return embeddings, names

def save_gallery_to_json(file_path: str, embeddings: list[list[np.ndarray]], names: list[str]) -> None:
    data = []
    for i, queue in enumerate(embeddings):
      if len(queue) > 0:
        data.append({"name": names[i], "embedding": queue[0].tolist()})
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)