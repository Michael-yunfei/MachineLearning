function [beta_perc, k] = fitPerceptron(featuresSet, groupSet, classLabels)

[numObs, numFeatures] = size(featuresSet);

classLabelsNumber = length(classLabels);
if classLabelsNumber > 2
    fprintf('perceptron algorithm works only for the binary classification');
    return;
end
Y = groupSet;
Y(Y == classLabels(1)) = -1;
Y(Y == classLabels(2)) = 1;

%perceptron classification







end

