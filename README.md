The goal of this assignment is to implement a multi-layer neural network model using Tensorflow (without using Keras) .

The neural network model in this assignment is a multi-layer of neurons with multiple nodes in each layer.
Your code should be using Tensorflow and numpy. DO NOT use Keras or tf.layers() or any other high-level API, package or library.
DO NOT alter/change the name of the function or the parameters of the function.
You may introduce additional functions (helper functions) as needed. All the helper functions should be put in the same file with multi_layer_nn_tensorflow()  function.
The comments in the multi_layer_nn_tensorflow()   function provide additional information that should help with your implementation.
The "Assignment_02_tests.py" file includes a very minimal set of unit tests for the multi_layer_nn.py file. The assignment grade will be based on your code passing these tests (and other additional tests).
You may modify the "Assignment_02_tests.py" to include more tests. You may also add additional tests to help you during development of your code.
DO NOT submit the "Assignment_02_tests.py" file when submitting your Assignment_02
You may run these tests using the command:      py.test --verbose Assignment_02_tests.py
Submit the output of running the test cases as a separate text file as part of your submission.
 
The following is roughly what your output should look like if all tests pass 
  
Assignment_02_tests.py::test_random_weight_init PASSED                   [ 10%]
Assignment_02_tests.py::test_weight_update_mse PASSED                    [ 20%]
Assignment_02_tests.py::test_weight_update_cross_entropy PASSED          [ 30%]
Assignment_02_tests.py::test_weight_update_svm PASSED                    [ 40%]
Assignment_02_tests.py::test_assign_weights_by_value PASSED              [ 50%]
Assignment_02_tests.py::test_error_output_dimensions PASSED              [ 60%]
Assignment_02_tests.py::test_error_vals_mse PASSED                       [ 70%]
Assignment_02_tests.py::test_error_vals_cross_entropy PASSED             [ 80%]
Assignment_02_tests.py::test_initial_validation_output PASSED            [ 90%]
Assignment_02_tests.py::test_many_layers PASSED                          [100%]
 
======================== 10 passed, 1 warning in 5.80s ======================== 
Process finished with exit code 0
