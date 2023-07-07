%% load cifar100_tiny networks
reachOptions = struct;
reachOptions.reachMethod = 'approx-star';
%% network 2
for i = 1 : 3
    net{i} = importONNXNetwork('./onnx/mnist-net_256x'+string(i*2)+'.onnx','InputDataFormats',"SSC", 'OutputDataFormats',"BC");
    nnvnet{i} = matlab2nnv(net{i});
    properties{i} = load_vnnlib('./vnnlib/prop_0_0.03.vnnlib');

    test{i} = properties{i}.lb;
    pred_net{i} = predict(net{i},test{i}');  
    pred_nnv{i} = nnvnet{i}.evaluate(test{i});

    if all(abs(pred_nnv{i} - pred_net{i}')<0.0001 == 1)
        fprintf("prediction value is same for matnet and nnvnet"+ string(i)+ "\n")
    end

    Im{i} = ImageStar(test{i},test{i});
    R{i}  = nnvnet{i}.reach(Im{i},reachOptions);
    [lb{i},ub{i}]=getRanges(R{i});
    
    if all(abs(squeeze(lb{i}) - pred_nnv{i})<0.001 == 1)
        fprintf("reachset value is same for nnvnet predict and nnvnet reach"+ string(i)+ "\n")
    end
end