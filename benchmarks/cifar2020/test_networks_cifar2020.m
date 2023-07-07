%% load cifar100_tiny networks

%% network 1
net_cifar2020_1 = importONNXNetwork('./onnx/cifar10_2_255_simplified.onnx', 'OutputDataFormats',"BC"); %reshape
nnvnet1 = matlab2nnv(net_cifar2020_1);
properties1 = load_vnnlib('./vnnlib/cifar10_spec_idx_0_eps_0.00784_n1.vnnlib');

test1 = properties1.lb;
test1 = python_reshape(test1,net_cifar2020_1.Layers(1,1).InputSize);
pred_net1 = predict(net_cifar2020_1,test1);
pred_nnv1 = nnvnet1.evaluate(test1);

if all(abs(pred_nnv1 - pred_net1')<0.0001 == 1)
    fprintf("prediction value is same for matnet and nnvnet 1\n")
end

Im1 = ImageStar(test1,test1);
reachOptions = struct;
reachOptions.reachMethod = 'approx-star';
R1  = nnvnet1.reach(Im1,reachOptions);
[lb1,ub1]=getRanges(R1);

if all(abs(squeeze(lb1) - pred_nnv1)<0.001 == 1)
    fprintf("reachset value is same for nnvnet predict and nnvnet reach 1 \n")
end


% %% network 2
% net_cifar100_tiny2 = importONNXNetwork('./onnx/cifar10_2_255.onnx', 'OutputDataFormats',"BC"); %reshape
% nnvnet2 = matlab2nnv(net_cifar100_tiny2);
% properties2 = load_vnnlib('./vnnlib/CIFAR100_resnet_medium_prop_idx_115_sidx_2873_eps_0.0039.vnnlib');
% 
% test2 = properties2.lb;
% test2 = python_reshape(test2,net_cifar100_tiny2.Layers(1,1).InputSize);
% pred_net2 = predict(net_cifar100_tiny2,test2);
% pred_nnv2 = nnvnet2.evaluate(test2);
% 
% if all(abs(pred_nnv2 - pred_net2)<0.0001 == 1)
%     fprintf("prediction value is same for matnet and nnvnet 2 \n")
% end
% 
% Im2 = ImageStar(test2,test2);
% R2  = nnvnet2.reach(Im2,reachOptions);
% [lb2,ub2]=getRanges(R2);
% 
% if all(abs(lb2 - pred_nnv2)<0.001 == 1)
%     fprintf("reachset value is same for nnvnet predict and nnvnet reach 2\n")
% end


%% network 3
net_cifar2020_3 = importONNXNetwork('./onnx/cifar10_8_255_simplified.onnx', 'OutputDataFormats',"BC"); %reshape
nnvnet3 = matlab2nnv(net_cifar2020_3);
properties3 = load_vnnlib('./vnnlib/cifar10_spec_idx_1_eps_0.03137_n1.vnnlib');

test3 = properties3.lb;
test3 = python_reshape(test3,net_cifar2020_3.Layers(1,1).InputSize);
pred_net3 = predict(net_cifar2020_3,test3);
pred_nnv3 = nnvnet3.evaluate(test3);

if all(abs(pred_nnv3 - pred_net3')<0.0001 == 1)
    fprintf("prediction value is same for matnet and nnvnet 3\n")
end

Im3 = ImageStar(test3,test3);
R3  = nnvnet3.reach(Im3,reachOptions);
[lb3,ub3]=getRanges(R3);

if all(abs(squeeze(lb3) - pred_nnv3)<0.001 == 1)
    fprintf("reachset value is same for nnvnet predict and nnvnet reach 3\n")
end


% %% network 4
% net_cifar100_tiny4 = importONNXNetwork('./onnx/cifar10_8_255.onnx', 'OutputDataFormats',"BC"); %reshape
% nnvnet4 = matlab2nnv(net_cifar100_tiny4);
% properties4 = load_vnnlib('./vnnlib/CIFAR100_resnet_super_prop_idx_319_sidx_4416_eps_0.0039.vnnlib');
% 
% test4 = properties4.lb;
% test4 = python_reshape(test4,net_cifar100_tiny4.Layers(1,1).InputSize);
% pred_net4 = predict(net_cifar100_tiny4,test4);
% pred_nnv4 = nnvnet4.evaluate(test4);
% 
% if all(abs(pred_nnv4 - pred_net4)<0.0001 == 1)
%     fprintf("prediction value is same for matnet and nnvnet 4\n")
% end
% 
% Im4 = ImageStar(test4,test4);
% R4  = nnvnet4.reach(Im4,reachOptions);
% [lb4,ub4]=getRanges(R4);
% 
% if all(abs(lb4 - pred_nnv4)<0.001 == 1)
%     fprintf("reachset value is same for nnvnet predict and nnvnet reach 4\n")
% end


%% network 5
net_cifar2020_5 = importONNXNetwork('./onnx/convBigRELU__PGD.onnx','InputDataFormats',"BCSS", 'OutputDataFormats',"BC");
nnvnet5 = matlab2nnv(net_cifar2020_5);
properties5 = load_vnnlib('./vnnlib/cifar10_spec_idx_1_eps_0.00784.vnnlib');

test5 = properties5.lb;
test5 = python_reshape(test5,net_cifar2020_5.Layers(1,1).InputSize);
pred_net5 = predict(net_cifar2020_5,test5);
pred_nnv5 = nnvnet5.evaluate(test5);

if all(abs(pred_nnv5 - pred_net5')<0.0001 == 1)
    fprintf("prediction value is same for matnet and nnvnet 5\n")
end

Im5 = ImageStar(test5,test5);
R5  = nnvnet5.reach(Im5,reachOptions);
[lb5,ub5]=getRanges(R5);

if all(abs(squeeze(lb5) - pred_nnv5)<0.001 == 1)
    fprintf("reachset value is same for nnvnet predict and nnvnet reach 5\n")
end