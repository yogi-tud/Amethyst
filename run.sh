cmake -E make_directory repro
#all other experiments
./build/gpu_elementstuffing   > repro/1gib

#compaction experiments:
./build/gpu_elementstuffing  -m 0 -s 0.01 > repro/uni_0.01
./build/gpu_elementstuffing  -m 1 -s 0.01 > repro/sc_0.01
./build/gpu_elementstuffing  -m 2 -s 0.01 > repro/mc_0.01

./build/gpu_elementstuffing  -m 0 -s 0.1 > repro/uni_0.1
./build/gpu_elementstuffing  -m 1 -s 0.1 > repro/sc_0.1
./build/gpu_elementstuffing  -m 2 -s 0.1 > repro/mc_0.1

./build/gpu_elementstuffing  -m 0 -s 0.50 > repro/uni_0.5
./build/gpu_elementstuffing  -m 1 -s 0.50 > repro/sc_0.5
./build/gpu_elementstuffing  -m 2 -s 0.50 > repro/mc_0.5

./build/gpu_elementstuffing  -m 0 -s 0.90 > repro/uni_0.9
./build/gpu_elementstuffing  -m 1 -s 0.90 > repro/sc_0.9
./build/gpu_elementstuffing  -m 2 -s 0.90 > repro/mc_0.9
