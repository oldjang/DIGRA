# DIGRA

## Quick Start

### Compile and Run

```bash
cmake . && make
```

Running example benchmark on SIFT dataset:
```bash
./Digra 10000 1000 128 10 100 8 [path_to_sift_base.fvecs] [path_to_sift_query.fvecs] [path_to_sift_base.data]
```

Parameter description:

1. `10000`: Number of base vectors
2. `1000`: Number of query vectors
3. `128`: Vector dimension
4. `10`: Number of nearest neighbors to return
5. `100`: The size of result set during index building
6. `8`: The degree of the graph index 
7. `[path_to_sift_base.fvecs]`: Path to the SIFT base vectors file
8. `[path_to_sift_query.fvecs]`: Path to the SIFT query vectors file
9. `[path_to_sift_base.data]`: Path to the SIFT base data file

The `.data` file is a text file containing key-value pairs. Each line represents an entry in the format:

```
key value
```
