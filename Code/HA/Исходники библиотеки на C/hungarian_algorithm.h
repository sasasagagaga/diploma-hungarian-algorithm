// Hungarian algorithm


#ifndef CTYPES_HUNGARIAN_ALGORITHM_H
#define CTYPES_HUNGARIAN_ALGORITHM_H


#define data_type double

void hungarian_algorithm_implementation(
        data_type* a, int n, int m,
        int* ind, int* ind_len,
        data_type* u, data_type* v,
        int* p, int* way,
        data_type* minv, char* used
);

void hungarian_algorithm(
        data_type* arr, int n, int m,
        int* ind, int* ind_len
);


#endif //CTYPES_HUNGARIAN_ALGORITHM_H
