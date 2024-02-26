cmake -E make_directory build
cd build
cmake ..
cmake --build . --target clean -- -j 12
cmake --build . --target gpu_elementstuffing --config Release -j 12
