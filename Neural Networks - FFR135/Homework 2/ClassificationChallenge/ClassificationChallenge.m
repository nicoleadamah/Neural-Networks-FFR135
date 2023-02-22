%ANN HW2 2022 Classification Challenge Machine Nicole Adamah
close 
clear 
clc
xTest2 = loadmnist2();
[xTrain, tTrain, xVal, tVal, xTest, tTest] = LoadMNIST(3);

%% Neural-network algorithm
layers = [
    imageInputLayer([28 28 1])
    
    convolution2dLayer(3,8,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,16,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,32,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    fullyConnectedLayer(10)
    softmaxLayer
    classificationLayer];

options = trainingOptions('sgdm', ...
    'Momentum',0.9,...
    'InitialLearnRate',0.02, ...
    'MaxEpochs',3, ...
    'Shuffle','every-epoch', ...
    'MiniBatchSize',64, ...
    'ValidationData',{xVal tVal}, ...
    'ValidationFrequency',30, ...
    'ValidationPatience',5,...
    'Verbose',false, ...
    'Plots','training-progress');

network = trainNetwork(xTrain,tTrain,layers,options);
P = classify(network,xTest);
accuracy1 = sum(P == tTest)/numel(tTest);
P_xtest2 = classify(network,xTest2);
%% Print accuracy, plot and save as a csv-file
predicted = (char(P_xtest2));
disp(accuracy1)

n = randperm(10000,20);
for i = 1:20
    subplot(4,5,i);
    colormap(gray(256))
    image(xTest2(:,:,:,n(i)))
    set(gca,'XTick',[], 'YTick', [])
    set(gcf,'Position',[700 700 700 700])
    title("Predicted: " + str2double(predicted(n(i))))
end
writematrix(P_xtest2,"classifications.csv")
