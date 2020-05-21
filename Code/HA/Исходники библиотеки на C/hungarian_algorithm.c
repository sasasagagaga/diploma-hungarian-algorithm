#pragma GCC optimize("O3","unroll-loops","omit-frame-pointer","inline") //  Optimization flags
#pragma GCC option("arch=native","tune=native","no-zero-upper") //  Enable AVX
#pragma GCC target("avx")  //   Enable AVX


#include "hungarian_algorithm.h"

#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <stdbool.h>
//#include <time.h>


#define IND(i, j) ((i) * (m + 1) + (j))

const data_type INF = 1e200;

void hungarian_algorithm(data_type* arr, int n, int m, int* ind, int* ind_len)
{
    assert(n <= m);

    //int* ans = (int*)malloc((n + 1) * sizeof(int));

    //data_type** a = (data_type**)malloc((n + 1) * sizeof(data_type*));
    //for (int i = 0; i < n + 1; ++i) {
    //    a[i] = arr + i * (m + 1);
    //}

    data_type*  u    = (data_type*) calloc(n + 1, sizeof(data_type));
    data_type*  v    = (data_type*) calloc(m + 1, sizeof(data_type));
    int*        p    = (int*)       calloc(m + 1, sizeof(int)      );
    int*        way  = (int*)       calloc(m + 1, sizeof(int)      );
    data_type*  minv = (data_type*) calloc(m + 1, sizeof(data_type));
    char*       used = (char*)      calloc(m + 1, sizeof(char)     );


    hungarian_algorithm_implementation(arr, n, m, ind, ind_len, u, v, p, way, minv, used);


    free(used);
    free(minv);
    free(way);
    free(p);
    free(v);
    free(u);
    //free(a);

    //free(ans);
}

void hungarian_algorithm_implementation(
        data_type* a, int n, int m,
        //data_type** a, int n, int m,
        int* ind, int* ind_len,
        data_type* u, data_type* v,
        int* p, int* way,
        data_type* minv, char* used
)
{
    //for (int i = 0; i <= n; ++i) {
    //    for (int j = 0; j <= m; ++j) {
    //        printf("%f ", a[i][j]);
    //    }
    //    puts("");
    //}
    //puts("");

    for (int i = 1; i <= n; ++i) {
        p[0] = i;
        int j0 = 0;

        for (int j = 0; j <= m; ++j) {
            minv[j] = INF;
        }
        memset(used, 0, m + 1);

        do {
            used[j0] = true;
            int j1, i0 = p[j0];
            data_type delta = INF;

            for (int j = 1; j <= m; ++j) {
                //++cnt_operations;
                if (!used[j]) {
                    //data_type cur = a[i0][j] - u[i0] - v[j];
                    data_type cur = a[IND(i0, j)] - u[i0] - v[j];
                    if (cur < minv[j]) {
                        minv[j] = cur;
                        way[j] = j0;
                    }
                    if (minv[j] < delta) {
                        delta = minv[j];
                        j1 = j;
                    }
                }
            }

            for (int j = 0; j <= m; ++j) {
                //++cnt_operations;
                if (used[j]) {
                    u[p[j]] += delta;
                    v[j] -= delta;
                }
                else {
                    minv[j] -= delta;
                }
            }
            j0 = j1;
        } while (p[j0] != 0);

        do {
            int j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
        } while (j0);
    }

    for (int j = 1; j <= m; ++j) {
        ind[p[j]] = j;
    }
    *ind_len = n;
}
