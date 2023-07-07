%% load sri networks

%% network 1
net_sri1 = importONNXNetwork('./onnx/resnet_3b2_bn_mixup_adv_4.0_bs128_lr-1.onnx', 'OutputDataFormats',"BC");
nnvnet1 = matlab2nnv(net_sri1);
properties1 = load_vnnlib('./vnnlib/cifar10_spec_idx_232_eps_0.00350.vnnlib');

test1 = properties1.lb;
test1 = python_reshape(test1,net_sri1.Layers(1,1).InputSize);
pred_net1 = predict(net_sri1,test1);
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
    fprintf("sriset value is same for nnvnet predict and nnvnet sri 1 \n")
end