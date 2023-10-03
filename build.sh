cmake .
cmake --build . --target clean -- -j 12
cmake --build . --target gpu_elementstuffing -- -j 12