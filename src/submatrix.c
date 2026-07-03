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

int main(void){
    unsigned int seed = (unsigned int)time(NULL); 
    srand(seed); 

    const unsigned int no_rows = 5;
    const unsigned int no_cols = 3;

    int matrix[no_rows][no_cols];

    // fill the matrix with 0s and 1s
    // according to missingness pattern
    double missing = 0.6;

    fill_random_matrix(no_rows, no_cols, matrix, missing);
    //print_matrix(no_rows, no_cols, matrix);
    
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
        return 1;
    }

    fprintf(output, "%u %u\n", no_rows, no_cols);
    store_matrix(no_rows, no_cols, matrix, output);

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
                    if(is_complete(no_rows, no_cols, matrix, set_rows, set_cols))
                        fprintf(output, "%u %u\n", set_rows, set_cols);
                    
                    set_cols = next_mask(set_cols);
                }
            }
            //printf("\n");
            set_rows = next_mask(set_rows);
        }
        //printf("\n");
    }
   
    fclose(output); 
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
}

void store_matrix(int no_rows, int no_cols, int matrix[no_rows][no_cols], FILE *output){
    for(int i = 0; i < no_rows; i++){
        for(int j = 0; j < no_cols; j++){
            fprintf(output, "%d", matrix[i][j]);
        }
        fprintf(output, "\n");
    }

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
