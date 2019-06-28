function [trainSample, testSample] = splitSample(sample, trainSize, permute)
% Splits the sample into a training sample and a test sample
% trainSize is a number between 0 and 1 that decides which percentage of 
% the sample is to be used for training permute is a boolean value such 
% that if true then the sample is randomly permuted before being split, 
% if false then no permutation is used.

[n, ~] = size(sample);

if permute
    ind = randperm(n);
else
    ind = 1:n;
end
sample = sample(ind, :);
lengthTrainSample = floor(trainSize * n);
trainSample = sample(1:lengthTrainSample, :);
testSample = sample(lengthTrainSample + 1:end, :);

end

