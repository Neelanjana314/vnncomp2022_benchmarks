%% load vnncomp 2022 benchmark NNs

%% cifar2020
net_cifar2020_1 = importONNXNetwork('./onnx/cifar10_2_255_simplified.onnx', 'OutputDataFormats',"BC"); %reshape
nnvnet1 = matlab2nnv(net_cifar2020_1);
%for pgd
net_cifar2020_5 = importONNXNetwork('./onnx/convBigRELU__PGD.onnx','InputDataFormats',"BCSS", 'OutputDataFormats',"BC");
nnvnet5 = matlab2nnv(net_cifar2020_5);

%% cifar100_tinyimagenet_resnet
net_cifar100_tiny1 = importONNXNetwork('./onnx/CIFAR100_resnet_large.onnx', 'OutputDataFormats',"BC");
nnvnet = matlab2nnv(net_cifar100_tiny1);

%% cifar_biasfied
net_bias = importONNXNetwork('./onnx/cifar_bias_field_'+string(i)+'.onnx','InputDataFormats',"BC", 'OutputDataFormats',"BC");%no reshape
nnvnet = matlab2nnv(net_bias);

%% mnist_fc: onnx to matlab:
net_fc = importONNXNetwork('./onnx/mnist-net_256x2.onnx','InputDataFormats',"SSC", 'OutputDataFormats',"BC"); %no reshape
nnvnet = matlab2nnv(net_fc);

%% oval21
net_oval1 = importONNXNetwork('./onnx/cifar_base_kw.onnx', 'OutputDataFormats',"BC"); %reshape
nnvnet = matlab2nnv(net_oval1);

%% reachprob: onnx to matlab
net_reach1 = importONNXNetwork('./onnx/gcas.onnx', 'OutputDataFormats',"BC"); % no reshape
nnvnet = matlab2nnv(net_reach1);

%% rl_bench: onnx to matlab
net_rl1 = importONNXNetwork('./onnx/cartpole.onnx','InputDataFormats',"BC", 'OutputDataFormats',"BC"); % no reshape
nnvnet = matlab2nnv(net_rl1);

%% sri_a: onnx to matlab
net_sria = importONNXNetwork('./onnx/resnet_3b2_bn_mixup_adv_4.0_bs128_lr-1.onnx', 'OutputDataFormats',"BC"); % reshape
nnvnet = matlab2nnv(net_sria);

%% sri_a: onnx to matlab
net_srib = importONNXNetwork('./onnx/resnet_3b2_bn_mixup_ssadv_4.0_bs128_lr-1_v2.onnx', 'OutputDataFormats',"BC"); % reshape
nnvnet = matlab2nnv(net_srib);