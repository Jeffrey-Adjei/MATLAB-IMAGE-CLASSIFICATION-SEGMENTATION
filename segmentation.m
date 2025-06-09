%% Set folder paths
imageFolder = fullfile("daffodilSeg", "ImagesRsz256");
labelFolder = fullfile("daffodilSeg", "LabelsRsz256");

%% Define classes and RGB labels
classes = ["flower", "background"];
labelIDs = {
    [128 0 0];  % flower
    [0 0 0; 0 128 0; 0 0 128; 128 128 0]  % all background variants
};

%% Create datastores
imds = imageDatastore(imageFolder);
pxds = pixelLabelDatastore(labelFolder, classes, labelIDs);

%% Pair them for training
trainingData = combine(imds, pxds);


