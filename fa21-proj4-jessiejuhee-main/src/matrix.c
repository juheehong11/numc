#include "matrix.h"
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

// Include SSE intrinsics
#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
#include <immintrin.h>
#include <x86intrin.h>
#endif

/* Below are some intel intrinsics that might be useful
 * void _mm256_storeu_pd (double * mem_addr, __m256d a)
 * __m256d _mm256_set1_pd (double a)
 * __m256d _mm256_set_pd (double e3, double e2, double e1, double e0)
 * __m256d _mm256_loadu_pd (double const * mem_addr)
 * __m256d _mm256_add_pd (__m256d a, __m256d b)
 * __m256d _mm256_sub_pd (__m256d a, __m256d b)
 * __m256d _mm256_fmadd_pd (__m256d a, __m256d b, __m256d c)
 * __m256d _mm256_mul_pd (__m256d a, __m256d b)
 * __m256d _mm256_cmp_pd (__m256d a, __m256d b, const int imm8)
 * __m256d _mm256_and_pd (__m256d a, __m256d b)
 * __m256d _mm256_max_pd (__m256d a, __m256d b)
*/

/* Generates a random double between low and high */
double rand_double(double low, double high) {
    double range = (high - low);
    double div = RAND_MAX / range;
    return low + (rand() / div);
}

/* Generates a random matrix */
void rand_matrix(matrix *result, unsigned int seed, double low, double high) {
    srand(seed);
    for (int i = 0; i < result->rows; i++) {
        for (int j = 0; j < result->cols; j++) {
            set(result, i, j, rand_double(low, high));
        }
    }
}

/*
 * Returns the double value of the matrix at the given row and column.
 * You may assume `row` and `col` are valid. Note that the matrix is in row-major order.
 */
double get(matrix *mat, int row, int col) {
    //int rsize = mat->rows;//(*mat).rows;
    int csize = mat->cols;//(*mat).cols;
    double * d = mat->data;//(*mat).data;
    //return *(d + row*csize + col);
    return d[csize * row + col];
}

/*
 * Sets the value at the given row and column to val. You may assume `row` and
 * `col` are valid. Note that the matrix is in row-major order.
 */
void set(matrix *mat, int row, int col, double val) {
    //int rsize = mat->rows;//(*mat).rows;
    int csize = mat->cols;//(*mat).cols;
    double * d = mat->data;//(*mat).data;
    d[csize * row + col] = val;
}

/*
 * Allocates space for a matrix struct pointed to by the double pointer mat with
 * `rows` rows and `cols` columns. You should also allocate memory for the data array
 * and initialize all entries to be zeros. `parent` should be set to NULL to indicate that
 * this matrix is not a slice. You should also set `ref_cnt` to 1.
 * You should return -1 if either `rows` or `cols` or both have invalid values. Return -2 if any
 * call to allocate memory in this function fails.
 * Return 0 upon success.
 */
int allocate_matrix(matrix **mat, int rows, int cols) {
    // Task 1.2 TODO
    // HINTS: Follow these steps.
    // 1. Check if the dimensions are valid. Return -1 if either dimension is not positive.
    // 2. Allocate space for the new matrix struct. Return -2 if allocating memory failed.
    // 3. Allocate space for the matrix data, initializing all entries to be 0. Return -2 if allocating memory failed.
    // 4. Set the number of rows and columns in the matrix struct according to the arguments provided.
    // 5. Set the `parent` field to NULL, since this matrix was not created from a slice.
    // 6. Set the `ref_cnt` field to 1.
    // 7. Store the address of the allocated matrix struct at the location `mat` is pointing at.
    // 8. Return 0 upon success.
    if (rows <= 0 || cols <= 0) {
	return -1;
    }
    //struct matrix *space;
    //space = (struct matrix *) malloc(sizeof(struct matrix));
    matrix *space;
    space = (matrix *) malloc(sizeof(matrix));
    if (space == NULL) {
	return -2;
    }
    double *data;
    data = (double *) calloc(rows * cols, sizeof(double));
    if (data == NULL) {
	return -2;
    }
    space->rows = rows;
    space->cols = cols;
    space->data = data;
    space->parent = NULL;
    space->ref_cnt = 1;
    *mat = space;
    return 0;
}

/*
 * You need to make sure that you only free `mat->data` if `mat` is not a slice and has no existing slices,
 * or that you free `mat->parent->data` if `mat` is the last existing slice of its parent matrix and its parent
 * matrix has no other references (including itself).
 */
void deallocate_matrix(matrix *mat) {
    // Task 1.3 TODO
    // HINTS: Follow these steps.
    // 1. If the matrix pointer `mat` is NULL, return.
    // 2. If `mat` has no parent: decrement its `ref_cnt` field by 1. If the `ref_cnt` field becomes 0, then free `mat` and its `data` field.
    // 3. Otherwise, recursively call `deallocate_matrix` on `mat`'s parent, then free `mat`.
    if (mat == NULL) {
	return;
    }
    if ((*mat).parent == NULL) {
	(*mat).ref_cnt = (*mat).ref_cnt - 1;
	if ((*mat).ref_cnt == 0) {
	    //free mat and data
	    free((*mat).data);
	    free(mat);
	}
    } else {
	//deallocate_matrix(mat.parent)
	//free mat
	deallocate_matrix((*mat).parent);
	free(mat);
    }
}

/*
 * Allocates space for a matrix struct pointed to by `mat` with `rows` rows and `cols` columns.
 * Its data should point to the `offset`th entry of `from`'s data (you do not need to allocate memory)
 * for the data field. `parent` should be set to `from` to indicate this matrix is a slice of `from`
 * and the reference counter for `from` should be incremented. Lastly, do not forget to set the
 * matrix's row and column values as well.
 * You should return -1 if either `rows` or `cols` or both have invalid values. Return -2 if any
 * call to allocate memory in this function fails.
 * Return 0 upon success.
 * NOTE: Here we're allocating a matrix struct that refers to already allocated data, so
 * there is no need to allocate space for matrix data.
 */
int allocate_matrix_ref(matrix **mat, matrix *from, int offset, int rows, int cols) {
    // Task 1.4 TODO
    // HINTS: Follow these steps.
    // 1. Check if the dimensions are valid. Return -1 if either dimension is not positive.
    // 2. Allocate space for the new matrix struct. Return -2 if allocating memory failed.
    // 3. Set the `data` field of the new struct to be the `data` field of the `from` struct plus `offset`.
    // 4. Set the number of rows and columns in the new struct according to the arguments provided.
    // 5. Set the `parent` field of the new struct to the `from` struct pointer.
    // 6. Increment the `ref_cnt` field of the `from` struct by 1.
    // 7. Store the address of the allocated matrix struct at the location `mat` is pointing at.
    // 8. Return 0 upon success.
    if (rows <= 0 || cols <= 0) {
	return -1;
    }
    //struct matrix *space;
    //space = (struct matrix *) malloc(sizeof(struct matrix));
    matrix *space;
    space = (matrix *) malloc(sizeof(matrix));
    if (space == NULL) {
	return -2;
    }
    double *original_data = from->data;
    double *d = original_data + offset;
    space->rows = rows;
    space->cols = cols;
    space->data = d;
    space->parent = from;
    from->ref_cnt = from->ref_cnt + 1;
    space->ref_cnt = 1;
    *mat = space;
    return 0;
}

/*
 * Sets all entries in mat to val. Note that the matrix is in row-major order.
 */
void fill_matrix(matrix *mat, double val) {
    int r = mat->rows;
    int c = mat->cols;
    double * d = mat->data;
    //double * d;
    __m256d temp = _mm256_set1_pd (val);
    #pragma omp parallel for
    for (int i = 0; i < r*c/4 * 4; i += 4) {
	_mm256_storeu_pd (d+i, temp);
    }
    for (int i = r*c/4 * 4; i< r*c; i++) {
	d[i] = val;
    }
    /*
    for (int i = 0; i < r*c; i++) {
	d[i] = val;
    }*/
}

/*
 * Store the result of taking the absolute value element-wise to `result`.
 * Return 0 upon success.
 * Note that the matrix is in row-major order.
 */
int abs_matrix(matrix *result, matrix *mat) {
    int matrix_len = mat->rows * mat->cols;

    __m256d neg_one =_mm256_set1_pd(-1.0);
    #pragma omp parallel for
    for (int i = 0; i < matrix_len/4 * 4; i += 4) {
	__m256d original = _mm256_loadu_pd(mat->data + i);
	__m256d negated = _mm256_mul_pd(original, neg_one);
	_mm256_storeu_pd(result->data + i, _mm256_max_pd(original, negated));
    }
    for (int i = matrix_len/4 * 4; i < matrix_len; i++) {
	double elem = mat->data[i];
	if (elem < 0) {
	    result->data[i] = -1 * elem;
	} else {
	    result->data[i] = elem;
	}
    }
    return 0;
    /*
    if (matrix_len > 0) {
	for(int i = 0; i < matrix_len; i++) {
	    double elem = mat->data[i];
	    if (elem < 0) {
		result->data[i] = -1 * elem;
	    } else {
		result->data[i] = elem;
	    }
	}
	return 0;
    } else {
	...
    } */
}

/*
 * (OPTIONAL)
 * Store the result of element-wise negating mat's entries to `result`.
 * Return 0 upon success.
 * Note that the matrix is in row-major order.
 */
int neg_matrix(matrix *result, matrix *mat) {
    // Task 1.5 TODO
}

/*
 * Store the result of adding mat1 and mat2 to `result`.
 * Return 0 upon success.
 * You may assume `mat1` and `mat2` have the same dimensions.
 * Note that the matrix is in row-major order.
 */
int add_matrix(matrix *result, matrix *mat1, matrix *mat2) {
    int matrix_len = mat1->rows * mat1->cols;
    #pragma omp parallel for
    for (int i = 0; i < matrix_len/4 * 4; i+=4) {
	__m256d first = _mm256_loadu_pd(mat1->data + i);
	__m256d second = _mm256_loadu_pd(mat2->data + i);
	_mm256_storeu_pd(result->data + i, _mm256_add_pd(first, second));
    }
    #pragma omp parallel for
    for (int i = matrix_len/4 *4; i < matrix_len; i++) {
	result->data[i] = mat1->data[i] + mat2->data[i];
    }
    return 0;
    /*
    if (matrix_len > 0) {
	for (int i = 0; i < matrix_len; i++) {
	    result->data[i] = mat1->data[i] + mat2->data[i];
	}
	return 0;
    }
    return -1;
    */
}

/*
 * (OPTIONAL)
 * Store the result of subtracting mat2 from mat1 to `result`.
 * Return 0 upon success.
 * You may assume `mat1` and `mat2` have the same dimensions.
 * Note that the matrix is in row-major order.
 */
int sub_matrix(matrix *result, matrix *mat1, matrix *mat2) {
    // Task 1.5 TODO
}

/*
 * Store the result of multiplying mat1 and mat2 to `result`.
 * Return 0 upon success.
 * Remember that matrix multiplication is not the same as multiplying individual elements.
 * You may assume `mat1`'s number of columns is equal to `mat2`'s number of rows.
 * Note that the matrix is in row-major order.
 */
int mul_matrix(matrix *result, matrix *mat1, matrix *mat2) {
    int c = mat1->rows;
    int r = mat2->cols;
    int middle = mat1->cols;
    int unroll = 4; //4 and 13 do give a perfect power benchmark
    // tried 4, 8, 9, 16, 12, 14, 5, 7, 10, 11, 13, 15 in this order. 
    //6 left to try. 

    double * r_data = result->data;
    double * mat1_data = mat1->data;
    double * mat2_data = mat2->data;
    fill_matrix(result, 0);

    if (c*r <= 100) {
        for (int j = 0; j < c; j++) {
            for (int k = 0; k < middle; k++) {
                __m256d b = _mm256_broadcast_sd(mat1_data + k + j * middle);
                for (int i = 0; i < r/4 * 4; i += 4) {
                    //__m256d t = _mm256_add_pd(_mm256_mul_pd(_mm256_loadu_pd(mat2_data + i + k*r), b), _mm256_loadu_pd(r_data + i + j*r));
                    __m256d t = _mm256_fmadd_pd(_mm256_loadu_pd(mat2_data + i + k*r), b, _mm256_loadu_pd(r_data + i + j*r));
                    _mm256_storeu_pd(r_data + i +j*r, t);
                }
                double bb = mat1_data[k + j*middle];
                for (int i = r/4 * 4; i < r; i++) {
                //for (int i = bound; i < r; i += 1) {
                    result->data[i + j*r] += mat2_data[i + k*r]*bb;
                }
            }
        }
        return 0; 
    }

    #pragma omp parallel for
    for (int j = 0; j < c; j++) {
        for (int k = 0; k < middle; k++) {
            __m256d b = _mm256_broadcast_sd(mat1_data + k + j * middle);
            //__m256d c[unroll];
            __m256d one;
            __m256d two;
            __m256d three;
            __m256d four;
            int bound = r/(4*unroll) * (4*unroll);
            for (int i = 0; i < bound; i+= 16) {
                one = _mm256_loadu_pd(r_data + i + 0 + j * r);
                two = _mm256_loadu_pd(r_data + i + 4 + j * r);
                three = _mm256_loadu_pd(r_data + i + 8 + j * r);
                four = _mm256_loadu_pd(r_data + i + 12 + j * r);

                one = _mm256_fmadd_pd(_mm256_loadu_pd(mat2_data + i + 0 + k * r), b, one);
                two = _mm256_fmadd_pd(_mm256_loadu_pd(mat2_data + i + 4 + k * r), b, two);
                three = _mm256_fmadd_pd(_mm256_loadu_pd(mat2_data + i + 8 + k * r), b, three);
                four = _mm256_fmadd_pd(_mm256_loadu_pd(mat2_data + i + 12 + k * r), b, four);

                _mm256_storeu_pd(r_data + i + 0 +j*r, one);
                _mm256_storeu_pd(r_data + i + 4 +j*r, two);
                _mm256_storeu_pd(r_data + i + 8 +j*r, three);
                _mm256_storeu_pd(r_data + i + 12 +j*r, four);
                /*
                for (int x = 0; x < unroll; x++) {
                    c[x] = _mm256_loadu_pd(r_data + i + x*4 + j * r);
                }
                for (int x = 0; x < unroll; x++) {
                    //c[x] = _mm256_add_pd(_mm256_mul_pd(_mm256_loadu_pd(mat2_data + i + x*4 + k * r), b), c[x]);   
                    c[x] = _mm256_fmadd_pd(_mm256_loadu_pd(mat2_data + i + x*4 + k * r), b, c[x]);
                }
                for (int x = 0; x < unroll; x++) {
                    _mm256_storeu_pd(r_data + i + x*4 +j*r, c[x]);
                }
                */
                
            }
            /*for (int i = r/(4*unroll)  * (4*unroll); i < r/4 * 4; i += 4) {
                //__m256d t = _mm256_add_pd(_mm256_mul_pd(_mm256_loadu_pd(mat2_data + i + k*r), b), _mm256_loadu_pd(r_data + i + j*r));
                __m256d t = _mm256_fmadd_pd(_mm256_loadu_pd(mat2_data + i + k*r), b, _mm256_loadu_pd(r_data + i + j*r));
                _mm256_storeu_pd(r_data + i +j*r, t);
            }*/
            double bb = mat1_data[k + j*middle];
            //for (int i = r/4 * 4; i < r; i++) {
            for (int i = bound; i < r; i += 1) {
                result->data[i + j*r] += mat2_data[i + k*r]*bb;
            }
        }
    }
    return 0; 
    /*
    for (int i = 0; i < r; i += 4) {
	for (int j = 0; j < c; j++) {
	    __m256d c0 = _mm256_set1_pd(0.0);//{0, 0, 0, 0};
            for (int k = 0; k < middle; k++) {
	    	c0 = _mm256_add_pd(c0, _mm256_mul_pd(_mm256_load_pd(mat1 + i+k*r), _mm256_broadcast_sd(mat2 + k+j*middle)));
		//c0 = _mm256_add_pd(c0, _mm256_mul_pd(_mm256_load_pd(mat1 + i * middle + k), _mm256_broadcast_sd(mat2 + k * c + j)));	
		//c0 = _mm256_fmadd_pd ?
	    }
	    _mm256_store_pd(result + i+j*c, c0);
	    //_mm256_store_pd(result + i*c + j, c0);
	}
    }
    return 0; */
    
    /*
    fill_matrix(result, 0);
    for (int i = 0; i < r; i++){
	for (int k = 0; k < middle; k++) {
	    for (int j = 0; j < c; j++) {
		result->data[i*c+j] += mat1->data[i*middle + k] * mat2->data[k*c + j];
	    }
	}
    }
    return 0; */
    
}

/*
 * Store the result of raising mat to the (pow)th power to `result`.
 * Return 0 upon success.
 * Remember that pow is defined with matrix multiplication, not element-wise multiplication.
 * You may assume `mat` is a square matrix and `pow` is a non-negative integer.
 * Note that the matrix is in row-major order.
 */
int copy_matrix(matrix *d, matrix *s) {
    if (s->rows == d->rows && s->cols == d->cols) {
	int n = s->rows * s->cols;
	for (int i = 0; i < n; i++) {
	    d->data[i] = s->data[i];
	}
	return 0;
    }
    return -1;
}

int pow_matrix(matrix *result, matrix *mat, int pow) {
    result->rows = mat->rows;
    result->cols = mat->cols;
    int n = result->rows;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (i == j) {
                result->data[i*n+j] = 1;
            } else {
                result->data[i*n+j] = 0;
            }
        }
    }
    if (pow == 0) {
        return 0;
    } else if (pow == 1){
        copy_matrix(result, mat);
        return 0;
    } else {
        matrix *temp = NULL; // = (matrix *) malloc(sizeof(matrix));
        matrix *temp2 = NULL;
        int alloc = allocate_matrix(&temp, result->rows, result->cols);
        int alloc2 = allocate_matrix(&temp2, result->rows, result->cols);
        if (alloc == 0 && alloc2 == 0) {
            copy_matrix(temp, result); //temp is a var for result
            copy_matrix(temp2, mat); //temp2 is a var for "x" or mat**power
            while (pow > 1) {
                if (pow % 2 == 0) {
                    int t2 = mul_matrix(mat, temp2, temp2);
                    if (t2 != 0)
                        return -1;
                    copy_matrix(temp2, mat);
                    pow = pow / 2;
                } else {
                    int t = mul_matrix(result, temp, mat); 
                    if (t != 0) 
                        return -1;
                    copy_matrix(temp, result);
                    int t2 = mul_matrix(mat, temp2, temp2);
                    if (t2 != 0)
                        return -1;
                    copy_matrix(temp2, mat);
                    pow = (pow - 1)/2;
                }
            }
            mul_matrix(result, temp, temp2);
            return 0;
	    } else {
            return -1;
        }
    }

    /*
    if (pow == 0) {
	    for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (i == j) {
                    result->data[i*n+j] = 1;
                } else {
                    result->data[i*n+j] = 0;
                }
	        }
	    }
	    return 0;
    } else if (pow == 1) {
        copy_matrix(result, mat);
        return 0;
    } else if (pow > 1) {
        matrix *temp = NULL; // = (matrix *) malloc(sizeof(matrix));
        int alloc = allocate_matrix(&temp, result->rows, result->cols);
        if (alloc == 0) {
                mul_matrix(result, mat, mat); //takes care of mat^2, stores it in result
            copy_matrix(temp, result);
            for (int i = 0; i < pow-2; i++) {
                int t = mul_matrix(result, temp, mat); 
                if (t != 0) {
                return -1;
                }
            copy_matrix(temp, result);
            }
	    }  
	    return 0;
    } else{
	    return -1;
    } */
}
