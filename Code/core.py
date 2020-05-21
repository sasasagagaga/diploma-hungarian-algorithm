import os
from typing import Union, Tuple, List

import numpy as np
import torch
import PIL.Image


def list_all_files(dir_name: str) -> List[str]:
    """
    Получает все файлы в заданной директории.

    Parameters
    ----------
    dir_name : str
        Директория, из которой берутся файлы.

    Returns
    -------
    list of str
        Список файлов в директории.
    """
    return [f for f in os.listdir(dir_name) if os.path.isfile(os.path.join(dir_name, f))]


# Основные папки
image_dir = os.getcwd() + '/Images/'
style_image_dir = f'{image_dir}/style'
content_image_dir = f'{image_dir}/content'
result_image_dir = f'{image_dir}/result'
model_dir = os.getcwd() + '/Models/'


# Количество стилей и контентов
num_styles = len(list_all_files(style_image_dir))
num_contents = len(list_all_files(content_image_dir))


# Параметры для обучения
USE_GPU = True
if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
# dtype = torch.float32


# Базовый размер минимальной стороны изображения
img_size = 300


# Вводим тип для изображений
Picture = Union[torch.Tensor, PIL.Image.Image]

# Введем тип для пары массивов с индексами (idx, idy)
IdxIdy = Tuple[np.ndarray, np.ndarray]
