dataDir = fullfile('17flowers');

[imdsTrain, imdsValidation] = loadClassificationImages(dataDir);

inputSize = [256 256 3];

augimdsTrain = augmentedImageDatastore(inputSize, imdsTrain);
augimdsValidation = augmentedImageDatastore(inputSize, imdsValidation);


