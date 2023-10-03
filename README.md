# Compressed data operations on GPU
Amethyst contains 5 different algorithms that run on CUDA 11.x capable devices.
These algorithms are: filter, binary operation, horizontal operation, groupBy and compaction.

To build and run all experiments:

./build.sh
./run.sh

Results are stored in /repro
/data contains measurements from a RTX 8000 GPU, which are also used for the paper.
Additional data is also present for 64 MiB and 256 MiB per array.
In /visualisation addtional visualisations for all benchmarks but compaction can be found for 64 MiB und 256 MiB
