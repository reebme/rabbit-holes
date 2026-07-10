#include<stdio.h>
#include<stdlib.h>
#include<limits.h>
#include<time.h>

/*
* 1 is available
* 0 is missing
*/

const unsigned int BYTE_SIZE = 8;

int Bernoulli(double p);
void print_matrix(int no_rows, int no_cols, int matrix[no_rows][no_cols]);
void store_matrix(int no_rows, int no_cols, int matrix[no_rows][no_cols], FILE *output);
void fill_random_matrix(int no_rows, int no_cols, int matrix[no_rows][no_cols], double missing);
void print_bits(unsigned int mask, unsigned int len);
unsigned int next_mask(unsigned int previous);
bool is_complete(int no_rows, int no_cols, int matrix[no_rows][no_cols], unsigned int row_mask, unsigned int col_mask);
void experiment(unsigned int no_rows, unsigned int no_cols, double missing, unsigned int seed);
void better_experiment(unsigned int no_rows, unsigned int no_cols, double missing, unsigned int seed);

int main(void){
    //const unsigned int no_rows[] = {28};
    //const unsigned int no_cols[] = {31};
    //double missing[] = {0.05, 0.1, 0.2, 0.5, 0.7, 0.9};
    const unsigned int no_rows[] = {17, 20};
    const unsigned int no_cols[] = {18, 25};
    //double missing[] = {0.05, 0.1, 0.2, 0.5, 0.7, 0.9};
    //const unsigned int no_rows[] = {5, 7, 9, 12, 15};
    //const unsigned int no_cols[] = {3, 8, 9, 15, 12};
    double missing[] = {0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95};
    // const unsigned int no_rows[] = {7};
    // const unsigned int no_cols[] = {9};
    // double missing[] = {0.5};
    
    if(sizeof(no_rows) != sizeof(no_cols)){
        printf("Wrong matrix sizes.");
        exit(-1);
    }

    int no_matrices = sizeof(no_rows)/sizeof(no_rows[0]);
    int no_missing = sizeof(missing)/sizeof(missing[0]);
   
    struct timespec start, end; 
  
    // if one wants more than one run
    // with the same params  
    int T = 10;

    for(int t = 0; t < T; t++){
        unsigned int seed = (unsigned int)time(NULL); 
        srand(seed); 
        for(int i = 0; i < no_matrices; i++){
            for(int j = 0; j < no_missing; j++){
                /*
                printf("%d rows, %d columns, %f missing.\n", no_rows[i], no_cols[i], missing[j]);
                clock_gettime(CLOCK_MONOTONIC, &start);

                experiment(no_rows[i], no_cols[i], missing[j], seed);

                clock_gettime(CLOCK_MONOTONIC, &end);
                double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
                printf("experiment: %.6f seconds\n", elapsed);

                clock_gettime(CLOCK_MONOTONIC, &start);
                */
                better_experiment(no_rows[i], no_cols[i], missing[j], seed);
                /*
                clock_gettime(CLOCK_MONOTONIC, &end);
                elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
                printf("better_experiment: %.6f seconds\n", elapsed);
                */
            }
        }
    }
   
    return 0;
}

int Bernoulli(double p){
    if((double)rand() / RAND_MAX < p)
        return 1;
    else
        return 0;
}

void print_matrix(int no_rows, int no_cols, int matrix[no_rows][no_cols]){
    for(int i = 0; i < no_rows; i++){
        for(int j = 0; j < no_cols; j++){
            printf("%d", matrix[i][j]);
        }
        printf("\n");
    }
    printf("\n");
}

void store_matrix(int no_rows, int no_cols, int matrix[no_rows][no_cols], FILE *output){
    for(int i = 0; i < no_rows; i++){
        for(int j = 0; j < no_cols; j++){
            fprintf(output, "%d", matrix[i][j]);
        }
        fprintf(output, "\n");
    }
    fprintf(output, "\n");
}

void fill_random_matrix(int no_rows, int no_cols, int matrix[no_rows][no_cols], double missing){
    for(int i = 0; i < no_rows; i++){
        for(int j = 0; j < no_cols; j++){
            matrix[i][j] = Bernoulli(1 - missing);
        }
    }
}

void print_bits(unsigned int mask, unsigned int len){
    int max_size = sizeof(unsigned int) * BYTE_SIZE;
    
    if(len > max_size){
        printf("Length longer than the mask\n");
        return;
    }
    
    for(int i=len-1; i >=0; i--){
        printf("%d", mask >> i & 1);
    }
    printf("\n");
}

unsigned int next_mask(unsigned int previous){
    // Gosper's hack:
    unsigned int lowest_bit = previous & -previous;
    unsigned int ripple = previous + lowest_bit;
    return (((ripple ^ previous) >> 2) / lowest_bit) | ripple;
}

bool is_complete(int no_rows, int no_cols, int matrix[no_rows][no_cols], unsigned int row_mask, unsigned int col_mask){
    //print_matrix(no_rows, no_cols, matrix);
    //print_bits(row_mask, no_rows);
    //print_bits(col_mask, no_cols);
    for(int i = 0; i < no_rows; i++){
        if((row_mask & (1u << i)) == 0) continue;

        for(int j = 0; j < no_cols; j++){
            if((col_mask & (1u << j)) == 0) continue;

            if(matrix[i][j] == 0) return false;
        }
    }
    return true;
}

void experiment(unsigned int no_rows, unsigned int no_cols, double missing, unsigned int seed){
    // file output disabled
    // it creates huge files

    int matrix[no_rows][no_cols];
    
    // fill the matrix with 0s and 1s
    // according to missingness pattern
    fill_random_matrix(no_rows, no_cols, matrix, missing);
    //print_matrix(no_rows, no_cols, matrix);
    
    //rows
    for(int i = no_rows; i > 0; i--){
        // iterate through all no_rows long values
        // with i bits set
        // using Gosper's hack
        // last i bits set
        unsigned int set_rows = (1u << i) - 1;
        unsigned int limit_rows = (1u << no_rows);
        while (set_rows < limit_rows){
            //print_bits(set_rows, no_rows);
            // columns
            for(int j = no_cols; j > 0; j--){
                unsigned int set_cols = (1u << j) - 1;
                unsigned int limit_cols = (1u << no_cols);
                while (set_cols < limit_cols){
                    //printf("\t");
                    //print_bits(set_cols, no_cols);
                    is_complete(no_rows, no_cols, matrix, set_rows, set_cols);
                         
                    set_cols = next_mask(set_cols);
                }
            }
            //printf("\n");
            set_rows = next_mask(set_rows);
        }
        //printf("\n");
    }
}

/*
* It can be improved by choosing smaller of row and columns number
* and iterating over all possible subsets of the smaller number.
* It makes complexity of that part 2^(min(n,m)) - 1
* further reducing execution time.
*/
void better_experiment(unsigned int no_rows, unsigned int no_cols, double missing, unsigned int seed){
    int matrix[no_rows][no_cols];
    
    // fill the matrix with 0s and 1s
    // according to missingness pattern
    fill_random_matrix(no_rows, no_cols, matrix, missing);
    // print_matrix(no_rows, no_cols, matrix);
    
    char filename[100];
    snprintf(
        filename,
        sizeof(filename),
        "../results/benchmark_matrices/r%u_c%u_m%02d_%u.txt",
        no_rows,
        no_cols,
        (int)(missing * 100),
        seed
    ); 

    FILE *output = fopen(filename, "w");
    if (output == NULL) {
        perror("Could not open file.\n");
        return;
    }

    fprintf(output, "%u %u\n", no_rows, no_cols);
    store_matrix(no_rows, no_cols, matrix, output);

    unsigned int row_masks[no_rows];

    for(int i = 0; i < no_rows; i++){
        row_masks[i] = 0;
        for(int j = 0; j < no_cols; j++){
            if(matrix[i][j] == 1)
                row_masks[i] |= 1u << j;
        }
    }

    int best_area = 0;
    // [0] rowmask
    // [1] colmask
    int solutions_capacity = 10;
    unsigned int (*solutions)[2] = malloc(solutions_capacity * sizeof(*solutions));
    int solutions_cnt = 0;

    //for all subsets of rows
    for(int i = no_rows; i > 0; i--){
        // iterate through all no_rows long values
        // with i bits set
        // using Gosper's hack
        unsigned int set_rows = (1u << i) - 1;
        unsigned int limit_rows = (1u << no_rows);
        while (set_rows < limit_rows){
            // calculate full columns
            unsigned int full_columns = (1u << no_cols) - 1u;
            for(int row = 0; row < no_rows; row++){
                if(((1u << row) & set_rows) == 0)
                    continue;
                full_columns &= row_masks[row];
            }
            // if a complete submatrix a found
            if(full_columns != 0){
                //print_bits(full_columns, no_cols);
                // calculate no of full columns
                int best_col_cnt = __builtin_popcount(full_columns);
                //printf("%d\n", best_col_cnt);
                // calculate the area of the complete submatrix
                int temp_area = i * best_col_cnt;

                if(temp_area > best_area){
                    //better solution was found
                    solutions_cnt = 0;
                    solutions[solutions_cnt][0] = set_rows;
                    solutions[solutions_cnt][1] = full_columns;
                    solutions_cnt += 1;
                    best_area = temp_area;
                } else if(temp_area == best_area){
                    //add solution to solutions list
                    if(solutions_cnt >= solutions_capacity){
                        int temp_capacity = solutions_capacity + 10;
                        unsigned int (*temp)[2] = realloc(solutions, temp_capacity * sizeof(*solutions));
                        if(temp == NULL){
                            fprintf(stderr, "Memory allocation failed.\nExiting.\n");
                            free(solutions);
                            exit(-1);
                        }
                        solutions = temp;
                        solutions_capacity = temp_capacity;
                    }
                        
                    solutions[solutions_cnt][0] = set_rows;
                    solutions[solutions_cnt][1] = full_columns;
                    solutions_cnt += 1;
                }
            }
            
            set_rows = next_mask(set_rows);
        }
    } 
    // write solutions to a file
    for(int s = 0; s < solutions_cnt; s++){
        fprintf(output, "%d %d\n", solutions[s][0], solutions[s][1]);
    }

    free(solutions); 
    fclose(output); 
}
