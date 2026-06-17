from PIL import Image
import config
import numpy as np


def resize_image(image):
    """
    Изменяет изображения для подачи в классификатор
    Args:
        image: PIL IMAGE
    returns:
        :PIL IMAGE
    """

    temp_image = image.copy()
    target_size = config.CLASSIFIER_PIC_SIZE_MAX
    width, height = temp_image.size
    if width > height:
        scale = target_size / width
    else:
        scale = target_size / height

    new_width = int(width * scale)
    new_height = int(height * scale)
    temp_image = temp_image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    result = Image.new("RGB", (target_size, target_size), (0, 0, 0))
    x = (target_size - new_width) // 2
    y = (target_size - new_height) // 2
    result.paste(temp_image, (x, y))

    return result


def crop_bounding_box(image, bbox):
    """
    Вырезает часть изображения по указанному bbox
    Args:
        image: PIL IMAGE
        bbox: LIST [x1, y1, x2, y2]
    returns:
        temp_image: PIL IMAGE
    """

    x1, y1, x2, y2 = [int(coord) for coord in bbox]

    x1 = max(0, min(x1, image.width - 1))
    y1 = max(0, min(y1, image.height - 1))
    x2 = max(0, min(x2, image.width))
    y2 = max(0, min(y2, image.height))

    temp_image = image.crop((x1, y1, x2, y2))
    return temp_image


def top_n_closest_embs(emb, emb_matrix, top_n=5):
    """
    Возвращает позиции и косинусное расстояние
    ближайших к данному вектору векторов базы
    Args:
        emb (np.array): вектор для поиска ближайшего соседа в базе (dim)
        emb_matrix (np.array): база векторов (N, dim)
        top_n (int, optional): искомое число ближайших векторов. Defaults to 5.
    returns:
        result (tuple): (top_n ближайших позиций в базе, top_n ближайших расстояний)
        np.array([]), np.array([]): если база пуста
    """
    if len(emb_matrix) == 0:
        return np.array([]), np.array([])

    top_n = min(top_n, len(emb_matrix))

    cosine_similarity = np.dot(emb_matrix, emb)
    top_n_positions = np.argsort(-cosine_similarity)[:top_n]
    top_n_similarities = cosine_similarity[top_n_positions]

    return top_n_positions, top_n_similarities
