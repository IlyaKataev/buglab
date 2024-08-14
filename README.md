# Buglab / Bug Game solution

[https://buglab.ru/](https://buglab.ru/)

## Problem Statement

This morning, a bug fell from the sky. You immediately realized that it's hungry and wants to eat you. You possess a single spell
that instantly constructs a maze for this bug, and you want to build the most difficult-to-navigate maze to give yourself enough time
to escape and save your life...

If we represent the maze as a rectangular matrix of 21 rows and 31 columns consisting of "#" (wall) or "." (empty space) symbols,
the Bug starts its movement from cell (1,1) (in 0-indexing) and searches for the exit from the maze located in cell (19,29) (also in 0-indexing).
The spell, of course, won't work if it's impossible to escape from the maze.

The maze boundaries (columns 0 and 30, and rows 0 and 20) must be walls, and cells (1,1) and (19,29) must be open spaces.
You can fill the remaining cells as you wish, ensuring the exit is reachable from the starting position.

The bug moves non-optimally, following these rules:

1. The bug considers four directions of movement.
2. From these directions, the bug chooses the directions with the least number of visits.
3. If the current direction is one of the least visited, the bug maintains its current direction of movement.
4. Otherwise, the bug chooses the first of the least visited directions in the following order:
   - Down
   - Right
   - Up
   - Left

For example, if the bug's current direction is downward, and the minimum number of visits is to the right and left,
the bug will go right because "right" has a higher priority than "left".

It should be noted that, moving according to this algorithm, the bug will always reach the exit when an exit exists.

The current direction of movement at the initial moment (before the first step) is downward.

Your task is to create a maze that maximizes the number of steps the bug takes to reach the exit.

## Solution Overview

This solution uses CUDA-accelerated parallel processing to simulate bug moving. It uses a greedy genetic algorithm approach,
continuously mutating and scoring mazes.

The mutation process involves applying a random square bitmask of size `mask_size` at a random location on the grid.

## Performance

Kaggle environment:
 - T4 x2
 - Maze with score 14734414
 - Benchmark mode

```
Total time: 647.239 seconds
Operations per second: 3.72982e+10
```

## Score

This solution achieves a score of 14734414

## Usage

To compile the solution, use a C++ compiler with CUDA support. For use on Kaggle, compile with the `-DKAGGLE` flag:

```
nvcc -DKAGGLE -o main main.cu
```

This flag sets the input and output directories as they are on Kaggle. If compiling without this flag, the input and output
will be in the same directory as the executable file. 

When running the program, you may need to adjust the `num_blocks` and `threads_per_block` variables for optimal performance on different GPUs.

### Running the program:

```
./main
```

The program starts by reading the initial maze from the `grid.txt`. As the program runs, it continually updates `grid.txt` with the current best maze.
