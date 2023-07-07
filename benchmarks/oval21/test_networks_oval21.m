%% load oval networks

%% network 1
net_oval1 = importONNXNetwork('./onnx/cifar_base_kw.onnx', 'OutputDataFormats',"BC"); %reshape
nnvnet1 = matlab2nnv(net_oval1);
properties1 = load_vnnlib('./vnnlib/cifar_base_kw-img8194-eps0.018300653594771243.vnnlib');

test1 = properties1.lb;
test1 = python_reshape(test1,net_oval1.Layers(1,1).InputSize);
pred_net1 = predict(net_oval1,test1);
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


%% network 2
net_oval2 = importONNXNetwork('./onnx/cifar_deep_kw.onnx', 'OutputDataFormats',"BC"); %reshape
nnvnet2 = matlab2nnv(net_oval2);
properties2 = load_vnnlib('./vnnlib/cifar_deep_kw-img4405-eps0.036732026143790855.vnnlib');

test2 = properties2.lb;
test2 = python_reshape(test2,net_oval2.Layers(1,1).InputSize);
pred_net2 = predict(net_oval2,test2);
pred_nnv2 = nnvnet2.evaluate(test2);

if all(abs(pred_nnv2 - pred_net2')<0.0001 == 1)
    fprintf("prediction value is same for matnet and nnvnet 2 \n")
end

Im2 = ImageStar(test2,test2);
R2  = nnvnet2.reach(Im2,reachOptions);
[lb2,ub2]=getRanges(R2);

if all(abs(squeeze(lb2) - pred_nnv2)<0.001 == 1)
    fprintf("reachset value is same for nnvnet predict and nnvnet reach 2\n")
end


%% network 3
net_oval3 = importONNXNetwork('./onnx/cifar_wide_kw.onnx', 'OutputDataFormats',"BC"); %reshape
nnvnet3 = matlab2nnv(net_oval3);
properties3 = load_vnnlib('./vnnlib/cifar_wide_kw-img6432-eps0.034771241830065365.vnnlib');

test3 = properties3.lb;
test3 = python_reshape(test3,net_oval3.Layers(1,1).InputSize);
pred_net3 = predict(net_oval3,test3);
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