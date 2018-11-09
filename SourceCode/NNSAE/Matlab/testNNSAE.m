function testNNSAE()

close all;
clear all;

%%%%%%%%%%%%%%%%%% Configuration %%%%%%%%%%%%%%%%%%%%%%%%%%
%% data parameters
numSamples = 60000;  %number of images
width = 28;          %image width = height


%% network parameters
inpDim = width^2;           %number of input/output neurons
netDim = 100;         

alpha = 1;          %alpha [0..1] is the decay rate for negative weights (alpha = 1 guarantees non-negative weights)
beta = 0;           %beta [0..1] is the decay rate for positive weights

%uncomment the following two lines for a symmetric decay function:
%alpha = 1e-6;          
%beta = 1e-6;

numEpochs = 500;     %number of sweeps through data for learning
lrateRO = 0.01;     %learning rate for synaptic plasticity of the read-out layer
lrateIP = 0.001;    %learning rate for intrinsic plasticity


%%%%%%%%%%%%%%%%%% Execution %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% data loading
data_path = '../../../../fashion_MNIST_matlab/fashion_MNIST_x_train.mat';
[x_train, x_test] = load_mnist_fashion(data_path);
%rescale data for better numeric performance
x_train = 0.25 * x_train;
x_test = 0.25 * x_test;


%% network creation
net = NNSAE(inpDim, netDim);
net.lrateRO = lrateRO;
net.lrateIP = lrateIP;
net.decayN = alpha;
net.decayP = beta;

net.init();

%% training
for e=1:numEpochs
    disp(['epoch ' num2str(e) '/' num2str(numEpochs)]);
    net.train(x_train);
end

save('NNSAE', 'net')

%% evaluating
x_rec_test = net.apply(x_test);
save('x_rec_test', 'x_rec_test')
clear('x_rec_test')

h_test = net.getEncoding(x_test);
save('h_test', 'h_test')
clear('h_test')

W=net.W;
save('W','W')
clear('W')

x_rec_train = net.apply(x_train);
save('x_rec_train', 'x_rec_train')
clear('x_rec_train')

h_train = net.getEncoding(x_train);
save('h_train', 'h_train')
clear('h_train')

