%
% myLDA( data, labels, numberOfFeatures )
%
% Arguments: data: is an MxN matrix, where M is the number of examples and
%            n is the number of digits
%            labels: Mx1 vector
%            featuresToExtract: number of features to be extracted from the
%            data (max is numberOfUniqueClasses-1)
%
% Returns:  ldaData: the output data from the LDA algrorithm 
%           projectionVectors: the projection vectors from LDA
%           eigVec: the eigenvectors from LDA
%           eigVal the eigenvalues from LDA
%
% TODO add case where N > M (Use PCA to  reduce N to M-1 then continue with
% LDA)
function [ldaData,projectionVectors,SB,eigVal] = myLDA( data, labels, featuresToExtract)
    
    % Check the arguments
    if ~exist('data', 'var')
        error('Data argument required.');
    end
    if ~exist('labels', 'var')
        error('Labels argument required.');
    end
    
    % Convert data and labels to double
    data = double(data);
    labels = double(labels);
    
    % Get the number of features and examples of the data
    numberOfExamples = size(data,1);
    numberOfLabels = length(unique(labels));
    uniqueLabels = unique(labels);
     
    % LDA can produce at most (numberOfClasses - 1) new features
    if exist('featuresToExtract', 'var')
        if( featuresToExtract > (numberOfLabels-1))
            error('LDA can produce at most numberOfClasses-1 new features');
        end
    else 
        featuresToExtract = numberOfLabels-1;
    end

    
    
    % Step 1: Find the indices for each class
    for i = 1 : numberOfLabels
        classExamples{i} = data(labels == uniqueLabels(i),:)';
        meanClass{i} = mean(classExamples{i},2);
    end
    
    % Step 2: Calculate the within class scatter matrix
    %         This matrix gives the area each class covers
    % classExamples{i} is MxN matrix - M is the number of features
    %                                  N is the number of examples
    
    % Allocate space for SW
    SW = zeros(size(classExamples{1},1),size(classExamples{1},1));
    for i = 1 : numberOfLabels
        for j = 1 : size(classExamples{i},2)
            centeredData(j,:) = classExamples{i}(:,j) - meanClass{i};
        end
        SW = SW+ centeredData' * centeredData;
    end
    
    
    % Step 3: Calculate between class scatter matrix
    %         This matrix gives the entire area covered by the classes
    
    %Calculate the mean of the entire data
    meanData = mean(data,1);
    
    % Allocate space for SB
    SB = zeros(length(meanClass{1}));
    for i = 1 : numberOfLabels
        numberOfExamplesForClass = size(classExamples{i},2);
        MeanData = meanClass{i}' - meanData;
        SB = SB + numberOfExamplesForClass * (MeanData' * MeanData);
    end
    
    % Step 4: Calculate eigenvectors and eigenvalues of the
    % inv(SW) * SB - Matlab suggest to use SW\SB
    [eigVec, eigVal] = eig(SW\SB);
    
    % Get the eigenvalues 
    bestEigVal = sortrows(diag(eigVal),-1); 
    for i = 1 : featuresToExtract 
        projectionVectors(:,i) = eigVec(:,diag(eigVal) == bestEigVal(i));
    end
    
    for i = 1 : numberOfExamples
        centeredData(i,:) = data(i,:) - meanData;
    end
     
    ldaData = centeredData * projectionVectors;
end
   