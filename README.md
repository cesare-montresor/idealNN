# idealNN
IdealNN is a simple Neural Network framework written in C++ that aims to provide pytorch-like APIs.     
The library is mainly intended for educational purposes to demistify the complexities behind neural network frameworks.    
_NOTE: The framework is called IdealNN in the same sense of 'ideal filters' in the frequency domain._

----

## Setup & Build
In order to simplify setup and build and install bash scripts have been provided.

## Release
Build the project in release, build the docs and run make install.
```bash
./make_test.sh
```

### Setup
Install all dependencies and create the build directories.
```bash
./setup_env.sh
```

## Test
Build the project in debug and run tests.
```bash
./make_test.sh
```

## Valgind
Build the project in debug and run valgrind on tests to check for memory leaks.
```bash
./make_test.sh
```

## Debug
Build the project for debugging purposes.
```bash
./make_debug.sh
```

---

## Dependencies
Here the breakdown of dependencies based on the task. 
### Required
```bash
sudo apt install libeigen3-dev 
```

### Test, Coverage & Analysis
```bash
sudo apt install valgrind
sudo apt install gcovr
sudo apt install cloc
```

### Documentation
```bash
sudo apt install doxygen 
sudo apt install texlive texlive-font-utils
sudo apt install graphviz
```

## Documentation
     
API: https://cesare-montresor.github.io/idealNN/     
Writeup: https://github.com/cesare-montresor/idealNN/blob/main/README.pdf     
    
## Resources and Articles

#### Project
- https://www.linkedin.com/pulse/what-general-c-project-structure-like-herbert-elwood-gilliland-iii
- https://hiltmon.com/blog/2013/07/03/a-simple-c-plus-plus-project-structure/
- https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2018/p1204r0.html
- https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines


#### Neural Network
- https://www.youtube.com/watch?v=44tFKZhPyP0
- https://www.youtube.com/watch?v=i94OvYb6noo <3
- https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html
- https://www.youtube.com/watch?v=tIeHLnjs5U8
- https://www.youtube.com/watch?v=09c7bkxpv9I
- https://www.youtube.com/watch?v=MswxJw-8PvE
- https://forums.fast.ai/t/gradients-for-softmax-are-tiny-solved/18970/11
- http://ufldl.stanford.edu/tutorial/supervised/SoftmaxRegression/
- https://iamtrask.github.io/2015/07/12/basic-python-network/

