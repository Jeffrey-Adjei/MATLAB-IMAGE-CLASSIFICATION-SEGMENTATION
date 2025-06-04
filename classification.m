dataDir = fullfile('17flowers');

[imdsTrain, imdsValidation] = loadClassificationImages(dataDir);

inputSize = [256 256 3];

augimdsTrain = augmentedImageDatastore(inputSize, imdsTrain);
augimdsValidation = augmentedImageDatastore(inputSize, imdsValidation);

disp(imdsTrain);
disp(imdsValidation);

countEachLabel(imdsTrain)
countEachLabel(imdsValidation)

layers = [
    imageInputLayer(inputSize)

    convolution2dLayer(3, 8, 'padding', 'same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2)


    convolution2dLayer(3, 16, 'padding', 'same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2)


    convolution2dLayer(3, 32, 'padding', 'same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, "Stride", 2)


    fullyConnectedLayer(17)
    softmaxLayer
    classificationLayer
    ];


options = trainingOptions('sgdm', ...
    'InitialLearnRate', 0.001, ...
    'MaxEpochs', 10, ...
    'MiniBatchSize', 32, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', augimdsValidation, ...
    'ValidationFrequency', 30, ...
    'Verbose', false, ...
    'Plots', 'training-progress');

net = trainNetwork(augimdsTrain, layers, options);