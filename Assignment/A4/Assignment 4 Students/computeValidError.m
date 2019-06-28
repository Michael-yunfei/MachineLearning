function [outputArg1,outputArg2] = computeValidError(inputArg1, train_length_CV, lambda_num)
% computeValidError computes the validatio error for k-fold 
% cross-validation.

lambda_num = 100;
lambda = linspace(1.e-6, 1.e-1, lambda_num);
beta0 = ones(d + 1, 1);

CV_error = zeros(k, lambda_num);

for l_num = 1:lambda_num
    for i = 1:k
        set_of_ind_trainCV = [1:(i-1), i+1:((i>k)*(i-1) + (i<=k)*k)];
        ind_testCV = i;
        train_featuresSet_CV = [];
        train_Y_CV = [];
        for j = set_of_ind_trainCV
            train_featuresSet_CV = [train_featuresSet_CV; foldVals(j).features_vals];
            train_Y_CV = [train_Y_CV; foldVals(j).Y_vals];
        end
        test_featuresSet_CV = foldVals(ind_testCV).features_vals;
        test_Y_CV = foldVals(ind_testCV).Y_vals;

        % normalizing features
        [train_featuresSet_CV_norm, mu_featuresSet_CV, sigma_featuresSet_CV] = featureNormalize(train_featuresSet_CV);
        [test_featuresSet_CV_norm] = featureNormalize(test_featuresSet_CV, mu_featuresSet_CV, sigma_featuresSet_CV);
        
        % train and validate
        % l2
        funToOptimize = @(beta) getBernoulliLoglik(beta, [ones(fold_length * (k - 1), 1), train_featuresSet_CV_norm], train_Y_CV, classLabels, 'l2', lambda(l_num));
        % using the built-in function
        options = optimoptions('fminunc', 'Algorithm', 'trust-region', 'SpecifyObjectiveGradient', true, 'Display', 'none');
        [beta_l2, ~] = fminunc(funToOptimize, beta0, options);
        % compute the validation error
        [~, ~, logLVal_l2_nonReg] = getBernoulliLoglik(beta_l2, [ones(fold_length, 1), test_featuresSet_CV_norm], test_Y_CV, classLabels, 'l2', lambda(l_num));
        CV_error_l2(i, l_num) = logLVal_l2_nonReg;
        
    end    
end

end

