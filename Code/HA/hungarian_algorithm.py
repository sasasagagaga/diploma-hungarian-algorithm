
import os
import ctypes
import platform

import numpy as np

from .. import core


# Загрузка библиотеки
if platform.system() != 'Windows':
    raise SystemExit('Библиотека для работы с венгерским алгоритмом пока что поддерживается только для Windows')

# shared_lib_suffix = 'dll' if platform.system() == 'Windows' else 'so'
dll_name = f'libhungarian_algorithm.dll'  # {shared_lib_suffix}
dll_abspath = os.path.dirname(os.path.abspath(__file__)) + os.path.sep + dll_name
HA_dll = ctypes.CDLL(dll_abspath)

hungarian_algorithm_C_wrapper = HA_dll.hungarian_algorithm

hungarian_algorithm_C_wrapper.restype = None
hungarian_algorithm_C_wrapper.argtypes = [
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int)
]


def __hungarian_algorithm(cost_matrix: np.ndarray) -> core.IdxIdy:
    """
    Обертка для венгерского алгоритма, написанного на C. Принимает на вход
    матрицу размера N x M, причем ожидает, что N <= M.

    Parameters
    ----------
    cost_matrix : np.ndarray
        Матрица стоимостей, над которой решается задача о назначениях.
        Количество строк в ней должно быть не больше числа столбцов.

    Returns
    -------
    idx, idy : np.ndarrays
        Индексы сопоставленных точек.
    """
    n, m = cost_matrix.shape

    assert n <= m, 'n должно быть не больше m!'

    zeroed_cost_matrix = np.zeros((n + 1, m + 1), dtype=np.float64)
    zeroed_cost_matrix[1:, 1:] = cost_matrix
    zeroed_cost_matrix = zeroed_cost_matrix.ravel()

    ind = np.zeros(max(n, m) + 1, dtype=np.int)
    ind_len = ctypes.c_int(-1)

    hungarian_algorithm_C_wrapper(
        zeroed_cost_matrix.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(n),
        ctypes.c_int(m),
        ind.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        ctypes.byref(ind_len)
    )

    n = ind_len.value
    return np.arange(n), ind[1:][:n] - 1


def hungarian_algorithm(cost_matrix):
    """
    Реализация венгерского алгоритма.

    Parameters
    ----------
    cost_matrix : np.ndarray
        Матрица стоимостей, над которой решается задача о назначениях.

    Returns
    -------
    idx, idy : np.ndarrays
        Индексы сопоставленных точек.
    """
    swapped = False
    if cost_matrix.shape[0] > cost_matrix.shape[1]:
        cost_matrix = cost_matrix.T
        swapped = True

    idx, idy = __hungarian_algorithm(cost_matrix)
    if swapped:
        idx, idy = idy, idx

    return idx, idy
