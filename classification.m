%% 1. Load and Split the Dataset
dataDir = fullfile('17flowers');

%% Create image datastore
imds = imageDatastore(dataDir, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

%% Shuffle and split the datastore (80% training, 20% validation)
imds = shuffle(imds);
[imdsTrain, imdsValidation] = splitEachLabel(imds, 0.8, 'randomized');

%% 2. Data Augmentation
inputSize = [256 256 3];
imageAugmenter = imageDataAugmenter( ...
    'RandRotation', [-15, 15], ...
    'RandXTranslation', [-15 15], ...
    'RandYTranslation', [-15 15], ...
    'RandXReflection', true, ...
    'RandYReflection', false, ...
    'RandScale', [0.9, 1.1]);

augimdsTrain = augmentedImageDatastore(inputSize, imdsTrain, ...
    'DataAugmentation', imageAugmenter);
augimdsValidation = augmentedImageDatastore(inputSize, imdsValidation);

%% 3. Define CNN Architecture
layers = [
    imageInputLayer(inputSize)

    convolution2dLayer(3, 8, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2)
    
    convolution2dLayer(3, 16, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2)

    dropoutLayer(0.5)

    fullyConnectedLayer(17)  % 17 flower classes
    softmaxLayer
    classificationLayer
];

%% 4. Training Options
options = trainingOptions('sgdm', ...
    'InitialLearnRate', 0.0007, ...
    'MaxEpochs', 30, ...
    'MiniBatchSize', 32, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', augimdsValidation, ...
    'ValidationFrequency', 30, ...
    'Verbose', false, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropPeriod', 10, ...
    'LearnRateDropFactor', 0.5, ...
    'Plots', 'training-progress');

%% 5. Train the Network
net = trainNetwork(augimdsTrain, layers, options);
%% 6. Evaluate the Model
predictedLabels = classify(net, augimdsValidation)
trueLabels = imdsValidation.Labels

valAccuracy = mean(predictedLabels == trueLabels);
fprintf('Validation accuracy: %.2f%%\n', valAccuracy * 100);

figure;
confusionchart(trueLabels, predictedLabels);
title('Confusion Matrix - Validation Set');

%% 7. Save the Model
