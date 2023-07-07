%% load reach networks

%% network 1
net_reach1 = importONNXNetwork('./onnx/gcas.onnx', 'OutputDataFormats',"BC");
nnvnet1 = matlab2nnv(net_reach1);
properties1 = load_vnnlib('./vnnlib/gcas_0.vnnlib');

test1 = properties1.lb;
pred_net1 = predict(net_reach1,test1');
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
net_reach2 = importONNXNetwork('./onnx/robot.onnx', 'OutputDataFormats',"BC"); %reshape
nnvnet2 = matlab2nnv(net_reach2);
properties2 = load_vnnlib('./vnnlib/robot_0.vnnlib');

test2 = properties2.lb;
pred_net2 = predict(net_reach2,test2');
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
net_reach3 = importONNXNetwork('./onnx/vdp.onnx', 'OutputDataFormats',"BC"); %reshape
nnvnet3 = matlab2nnv(net_reach3);
properties3 = load_vnnlib('./vnnlib/vdp_0.vnnlib');

test3 = properties3.lb;
pred_net3 = predict(net_reach3,test3');
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