#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <array>
#include <chrono>
#include <fstream>
#include <iostream>
#include <string>

using namespace std;

const int ROWS = 21;
const int COLS = 31;
const int GRID_SIZE = ROWS * COLS;
// (1, 0) down, (0, 1) right, (-1, 0) up, (0, -1) left
const array<array<int, 2>, 4> directions = {{{1, 0}, {0, 1}, {-1, 0}, {0, -1}}};
const int INF = 1e9;

using Grid = array<int, GRID_SIZE>;

__device__ const int d_ROWS = 21;
__device__ const int d_COLS = 31;
__device__ const int d_directions[4][2] = {{1, 0}, {0, 1}, {-1, 0}, {0, -1}};

// Define file paths based on environment
#ifdef KAGGLE
const string INPUT_PATH = "/kaggle/input/buglab-grid/";
const string OUTPUT_PATH = "/kaggle/working/";
#else
const string INPUT_PATH = "";
const string OUTPUT_PATH = "";
#endif

// Function to check CUDA errors
#define cudaCheckError(ans) \
    { cudaAssert((ans), __FILE__, __LINE__); }

inline void cudaAssert(cudaError_t code, const char *file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

// Check if the maze is solvable (CPU version)
bool is_solvable(const Grid &grid) {
    if (grid[COLS + 1] == 1 || grid[(ROWS - 2) * COLS + COLS - 2] == 1) {
        return false;
    }

    int queue[ROWS * COLS][2];
    int front = 0, rear = 0;
    bool visited[ROWS * COLS] = {false};

    queue[rear][0] = 1;
    queue[rear][1] = 1;
    rear++;
    visited[1 * COLS + 1] = true;

    while (front < rear) {
        int x = queue[front][0];
        int y = queue[front][1];
        front++;

        if (x == ROWS - 2 && y == COLS - 2) {
            return true;
        }

        for (int i = 0; i < 4; i++) {
            int nx = x + directions[i][0];
            int ny = y + directions[i][1];
            if (nx >= 0 && nx < ROWS && ny >= 0 && ny < COLS && grid[nx * COLS + ny] == 0 &&
                !visited[nx * COLS + ny]) {
                queue[rear][0] = nx;
                queue[rear][1] = ny;
                rear++;
                visited[nx * COLS + ny] = true;
            }
        }
    }

    return false;
}

// Calculate the score of the maze (CPU version)
int score(const Grid &grid) {
    if (!is_solvable(grid)) {
        return 0;
    }

    int visits[ROWS * COLS] = {0};
    int x = 1, y = 1;
    int current_dir = 0;
    int steps = 0;

    while (x != ROWS - 2 || y != COLS - 2) {
        steps++;
        visits[x * COLS + y]++;

        int valid_moves[4] = {0};
        int visit_counts[4] = {0};

        for (int i = 0; i < 4; ++i) {
            int nx = x + directions[i][0];
            int ny = y + directions[i][1];
            if (nx >= 0 && nx < ROWS && ny >= 0 && ny < COLS && grid[nx * COLS + ny] == 0) {
                valid_moves[i] = 1;
                visit_counts[i] = visits[nx * COLS + ny];
            } else {
                valid_moves[i] = 0;
                visit_counts[i] = INF;
            }
        }

        if (valid_moves[0] || valid_moves[1] || valid_moves[2] || valid_moves[3]) {
            int min_visits = INF;
            for (int i = 0; i < 4; i++) {
                if (visit_counts[i] < min_visits) {
                    min_visits = visit_counts[i];
                }
            }

            int possible_dirs[4] = {0};
            for (int i = 0; i < 4; i++) {
                if (visit_counts[i] == min_visits && valid_moves[i]) {
                    possible_dirs[i] = 1;
                }
            }

            int next_dir;
            if (possible_dirs[current_dir]) {
                next_dir = current_dir;
            } else if (possible_dirs[0]) {
                next_dir = 0;
            } else if (possible_dirs[1]) {
                next_dir = 1;
            } else if (possible_dirs[2]) {
                next_dir = 2;
            } else {
                next_dir = 3;
            }

            x += directions[next_dir][0];
            y += directions[next_dir][1];
            current_dir = next_dir;
        } else {
            return 0;
        }
    }
    return steps;
}

// Check if the maze is solvable (GPU version)
__device__ bool is_solvable_gpu(const int *grid) {
    if (grid[1 * d_COLS + 1] == 1 || grid[(d_ROWS - 2) * d_COLS + d_COLS - 2] == 1) {
        return false;
    }

    int queue[d_ROWS * d_COLS][2];
    int front = 0, rear = 0;
    bool visited[d_ROWS * d_COLS] = {false};

    queue[rear][0] = 1;
    queue[rear][1] = 1;
    rear++;
    visited[1 * d_COLS + 1] = true;

    while (front < rear) {
        int x = queue[front][0];
        int y = queue[front][1];
        front++;

        if (x == d_ROWS - 2 && y == d_COLS - 2) {
            return true;
        }

        for (int i = 0; i < 4; i++) {
            int nx = x + d_directions[i][0];
            int ny = y + d_directions[i][1];
            if (nx >= 0 && nx < d_ROWS && ny >= 0 && ny < d_COLS && grid[nx * d_COLS + ny] == 0 &&
                !visited[nx * d_COLS + ny]) {
                queue[rear][0] = nx;
                queue[rear][1] = ny;
                rear++;
                visited[nx * d_COLS + ny] = true;
            }
        }
    }

    return false;
}

// Calculate the score of the maze (GPU version)
__device__ int score_gpu(const int *grid) {
    if (!is_solvable_gpu(grid)) {
        return 0;
    }

    int visits[d_ROWS * d_COLS] = {0};
    int x = 1, y = 1;
    int current_dir = 0;
    int steps = 0;

    while (x != d_ROWS - 2 || y != d_COLS - 2) {
        ++steps;
        ++visits[x * d_COLS + y];

        int valid_moves[4] = {0};
        int visit_counts[4] = {0};

        for (int i = 0; i < 4; ++i) {
            int nx = x + d_directions[i][0];
            int ny = y + d_directions[i][1];
            if (nx >= 0 && nx < d_ROWS && ny >= 0 && ny < d_COLS && grid[nx * d_COLS + ny] == 0) {
                valid_moves[i] = 1;
                visit_counts[i] = visits[nx * d_COLS + ny];
            } else {
                valid_moves[i] = 0;
                visit_counts[i] = INF;
            }
        }

        if (valid_moves[0] || valid_moves[1] || valid_moves[2] || valid_moves[3]) {
            int min_visits = INF;
            for (int i = 0; i < 4; i++) {
                if (visit_counts[i] < min_visits) {
                    min_visits = visit_counts[i];
                }
            }

            int possible_dirs[4] = {0};
            for (int i = 0; i < 4; i++) {
                if (visit_counts[i] == min_visits && valid_moves[i]) {
                    possible_dirs[i] = 1;
                }
            }

            int next_dir;
            if (possible_dirs[current_dir]) {
                next_dir = current_dir;
            } else if (possible_dirs[0]) {
                next_dir = 0;
            } else if (possible_dirs[1]) {
                next_dir = 1;
            } else if (possible_dirs[2]) {
                next_dir = 2;
            } else {
                next_dir = 3;
            }

            x += d_directions[next_dir][0];
            y += d_directions[next_dir][1];
            current_dir = next_dir;
        } else {
            return 0;
        }
    }

    return steps;
}

// Mutate the grid on GPU
__device__ void mutate_grid(int *grid, curandState *state) {
    const int mask_size = 4;
    int start_i = curand(state) % (d_ROWS - 1 - mask_size) + 1;
    int start_j = curand(state) % (d_COLS - 1 - mask_size) + 1;

    for (int i = start_i; i < start_i + mask_size; ++i) {
        for (int j = start_j; j < start_j + mask_size; ++j) {
            int index = i * d_COLS + j;
            grid[index] ^= curand(state) & 1; // 0.5 probability to flip the bit
        }
    }
}

// Main optimization kernel
__global__ void optimize_kernel(int *d_grid, int *d_best_score, curandState *states, int iterations_per_thread) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    curandState localState = states[idx];

    int local_grid[d_ROWS * d_COLS];
    int best_local_grid[d_ROWS * d_COLS];
    for (int i = 0; i < d_ROWS * d_COLS; ++i) {
        local_grid[i] = d_grid[i];
        best_local_grid[i] = d_grid[i];
    }

    for (int iter = 0; iter < iterations_per_thread; ++iter) {
        mutate_grid(local_grid, &localState);
        int new_score = score_gpu(local_grid);

        if (new_score > *d_best_score) {
            atomicMax(d_best_score, new_score);
            if (new_score == *d_best_score) {
                for (int i = 0; i < d_ROWS * d_COLS; ++i) {
                    d_grid[i] = local_grid[i];
                    best_local_grid[i] = local_grid[i];
                }
            }
        } else {
            for (int i = 0; i < d_ROWS * d_COLS; ++i) {
                local_grid[i] = best_local_grid[i];
            }
        }
    }

    // Save the updated random state
    states[idx] = localState;
}

// Initialize CUDA random states
__global__ void init_random_states(curandState *states, unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(seed, idx, 0, &states[idx]);
}

// Read grid from file
Grid read_grid_from_file(const string &filename) {
    ifstream fin(filename);
    string line;
    Grid grid = {};
    int row = 0;
    while (getline(fin, line) && row < ROWS) {
        for (int col = 0; col < COLS; ++col) {
            grid[row * COLS + col] = (line[col] == '#') ? 1 : 0;
        }
        ++row;
    }
    return grid;
}

// Write grid to file
void write_grid_to_file(const Grid &grid, const string &filename) {
    ofstream fout(filename);
    for (int i = 0; i < ROWS; ++i) {
        for (int j = 0; j < COLS; ++j) {
            fout << (grid[i * COLS + j] == 1 ? '#' : '.');
        }
        fout << '\n';
    }
}

// Copy grid from CPU to GPU
void copy_grid_to_device(const Grid &host_grid, int *device_grid) {
    cudaCheckError(cudaMemcpy(device_grid, host_grid.data(), GRID_SIZE * sizeof(int), cudaMemcpyHostToDevice));
}

// Copy grid from GPU to CPU
void copy_grid_from_device(int *device_grid, Grid &host_grid) {
    cudaCheckError(cudaMemcpy(host_grid.data(), device_grid, GRID_SIZE * sizeof(int), cudaMemcpyDeviceToHost));
}

int main() {
    auto total_start = chrono::high_resolution_clock::now();
    int step = 0;
    // Read current best grid
    Grid best_grid = read_grid_from_file(INPUT_PATH + "grid.txt");
    int best_score = score(best_grid);

    int num_devices;
    cudaGetDeviceCount(&num_devices);
    cout << "Number of devices: " << num_devices << endl;

    // Allocate memory on GPU
    const int MAX_DEVICES = 2;
    int *d_grid[MAX_DEVICES], *d_best_score[MAX_DEVICES];
    for (int dev = 0; dev < num_devices; ++dev) {
        cudaSetDevice(dev);
        cudaCheckError(cudaMalloc(&d_grid[dev], GRID_SIZE * sizeof(int)));
        cudaCheckError(cudaMalloc(&d_best_score[dev], sizeof(int)));
    }

    // Set kernel launch parameters
    const int num_blocks = 32;
    const int threads_per_block = 256;
    const int iterations_per_thread = 10;

    // Allocate memory for random number generator states
    curandState *d_states[num_devices];
    for (int dev = 0; dev < num_devices; ++dev) {
        cudaSetDevice(dev);
        cudaCheckError(cudaMalloc(&d_states[dev], num_blocks * threads_per_block * sizeof(curandState)));
    }

    // Initialize CUDA random states
    for (int dev = 0; dev < num_devices; ++dev) {
        cudaSetDevice(dev);
        init_random_states<<<num_blocks, threads_per_block>>>(d_states[dev], time(NULL));
    }
    cudaCheckError(cudaDeviceSynchronize());

    while (true) {
        ++step;
        auto step_start = chrono::high_resolution_clock::now();

        int overall_best_score = best_score;
        Grid overall_best_grid = best_grid;

        // Copy data to GPU
        for (int dev = 0; dev < num_devices; ++dev) {
            cudaSetDevice(dev);
            cudaCheckError(cudaMemcpy(d_grid[dev], best_grid.data(), GRID_SIZE * sizeof(int), cudaMemcpyHostToDevice));
            cudaCheckError(cudaMemcpy(d_best_score[dev], &best_score, sizeof(int), cudaMemcpyHostToDevice));
        }

        // Run optimization for each device
        for (int dev = 0; dev < num_devices; ++dev) {
            cudaSetDevice(dev);
            optimize_kernel<<<num_blocks, threads_per_block>>>(d_grid[dev], d_best_score[dev], d_states[dev],
                                                               iterations_per_thread);
        }
        cudaCheckError(cudaDeviceSynchronize());

        // Compare results from all devices
        for (int dev = 0; dev < num_devices; ++dev) {
            cudaSetDevice(dev);
            int device_best_score;
            Grid device_best_grid;
            cudaCheckError(cudaMemcpy(&device_best_score, d_best_score[dev], sizeof(int), cudaMemcpyDeviceToHost));
            cudaCheckError(
                cudaMemcpy(device_best_grid.data(), d_grid[dev], GRID_SIZE * sizeof(int), cudaMemcpyDeviceToHost));

            if (device_best_score > overall_best_score) {
                overall_best_score = device_best_score;
                overall_best_grid = device_best_grid;
            }
        }

        // Update best result
        if (overall_best_score > best_score) {
            best_score = overall_best_score;
            best_grid = overall_best_grid;
            write_grid_to_file(best_grid, OUTPUT_PATH + "grid.txt");
        }

        auto step_end = chrono::high_resolution_clock::now();
        chrono::duration<double> step_duration = step_end - step_start;
        chrono::duration<double> total_duration = step_end - total_start;

        cout << "Step " << step << ": Best score = " << best_score << endl;
        cout << "Total time: " << total_duration.count() << " seconds" << endl;
        cout << "Time for this step: " << step_duration.count() << " seconds" << endl << endl;
    }

    // Free memory (this code will never be reached due to the infinite loop)
    for (int dev = 0; dev < num_devices; ++dev) {
        cudaSetDevice(dev);
        cudaCheckError(cudaFree(d_grid[dev]));
        cudaCheckError(cudaFree(d_best_score[dev]));
        cudaCheckError(cudaFree(d_states[dev]));
    }

    return 0;
}
