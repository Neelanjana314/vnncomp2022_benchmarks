%% load cifar100_tiny networks
reachOptions = struct;
reachOptions.reachMethod = 'approx-star';
%% network 2
for i = 0 : 71
    net{i+1} = importONNXNetwork('./onnx/cifar_bias_field_'+string(i)+'.onnx','InputDataFormats',"BC", 'OutputDataFormats',"BC");
    nnvnet{i+1} = matlab2nnv(net{i+1});
    properties{i+1} = load_vnnlib('./vnnlib/prop_'+string(i)+'.vnnlib');

    test{i+1} = properties{i+1}.lb;
    pred_net{i+1} = predict(net{i+1},test{i+1}');  
    pred_nnv{i+1} = nnvnet{i+1}.evaluate(test{i+1});

    if all(abs(squeeze(pred_nnv{i+1}) - pred_net{i+1}')<0.0001 == 1)
        fprintf("prediction value is same for matnet and nnvnet"+ string(i)+ "\n")
    end

    Im{i+1} = ImageStar(test{i+1},test{i+1});
    R{i+1}  = nnvnet{i+1}.reach(Im{i+1},reachOptions);
    [lb{i+1},ub{i+1}]=getRanges(R{i+1});
    
    if all(abs(lb{i+1} - pred_nnv{i+1})<0.001 == 1)
        fprintf("reachset value is same for nnvnet predict and nnvnet reach"+ string(i)+ "\n")
    end
end