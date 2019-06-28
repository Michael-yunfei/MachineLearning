function [foldVals, fold_length] = splitCrossValSample(sample, train_Y, k)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

classes = unique(train_Y);
train_length = sum(train_Y == classes(1));

fold_length = floor(train_length/k) * 2;
foldVals = struct();

rng('default');

ind_perm = randperm(fold_length * k);
train_sample = sample(ind_perm, :);
train_Y = train_Y(ind_perm);

for i = 1:k
    foldVals(i).features_vals = train_sample((i - 1) * fold_length + 1:i * fold_length, :);
    foldVals(i).Y_vals = train_Y((i - 1) * fold_length + 1:i * fold_length);
end

end

