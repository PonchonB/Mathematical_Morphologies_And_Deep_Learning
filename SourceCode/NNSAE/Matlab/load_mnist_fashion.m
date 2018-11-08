function [x_train, x_test] = load_mnist_fashion(path)

A=importdata(path);
x_train = A.x_train_fashion_MNIST;
x_test = A.x_test_fashion_MNIST;

end

