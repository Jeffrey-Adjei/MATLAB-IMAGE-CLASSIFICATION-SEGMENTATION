function [imdsTrain, imdsValidation] = loadClassificationImages(dataDir)
    % Create an image datastore without reading them into to memory
    imds = imageDatastore(dataDir, ...
        'IncludeSubfolders',true, ...
        'LabelSource','foldernames');
    
    % Shuffle the datastore to avoid class-wise grouping bias
    imds = shuffle(imds);

    % Split the image dataset into 80% training and 20% validation 
    [imdsTrain, imdsValidation] = splitEachLabel(imds, 0.8, 'randomized');

end