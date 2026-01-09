# TinyTorch (Notes and other commentary)

## TODOs:
0. pybind setup so that python can process lists and provide raw data to cpp
1. copy, move constructors to be created
2. proper destructor
3. test build
4. __repr__ equivalent
5. Outdated CMakeLists.txt file referring to an old test build, correct it post python bindings.


## Notes
- Couldn't make recursive templates initializer_list for initialization of tensors of arbitary depth in C++. Resorting to Python frontend which analyses a numpy array or Python list and forwards raw data and other metadata to the C++ backend.
- **C++ Quirks found**: Templated methods of a class to be defined in the .hpp file itself.

### Acknowledgement
I am trying to use as little as AI-tools/chatbots (they are damn good) for this. I think it is a restrictive habit, one no coder should have (much like Instagram scrolling, fries your brain and your ability to think). However, wherever I shall use AI-generated code, I shall add a comment regarding the credits.

I am also trying my best to add credits to the code pieces I borrow from the internet (Stackoverflow, etc.).

- CMake Tutorial [[link]](https://youtu.be/NGPo7mz1oa4?si=0pXX-n7rTgySSJtS)
- Pybind11 [[link]](https://pybind11.readthedocs.io/en/stable/index.html)

#### Gibberish-नामा

This is my attempt to a mini-PyTorch. I have an existing Python only implementation of an autograd engine [[link]](https://github.com/bihany-harsh/tinytorch/). One of the primary reasons I am not continuing that project is that it is quite bad code and I do not have the patience to re-write a lot of stuff (it just feels tirish, it works for the most part, one can do basic gradient descent, but that's about it. I have discovered some flaws in the core Tensor API there, and if I get some motivation for the same I shall make it clean) and secondly I think I quite lack C++ design skills, and this I think is a good way to improve them a bit. Regardless, I hope I stick with this for a while.
