function [ ] = plotGD2(beta_path, X, Y)
% 2D plotting of the gradient descent path for 
% the OLS regression:

res = 10;

Lf  = @(v1,v2) sum((X*[v1;v2]-Y).^2);
gLF = @(v1,v2) 2/length(Y) * X' * (X*[v1;v2] - Y);

figure('Name', 'Plot ML - 2D');

beta1 = beta_path(end,:);
nrm = 1.5*sqrt(norm(beta1));

gridx1 = linspace(-1.5*nrm+beta1(1),1.5*nrm+beta1(1),100);
gridy1 = linspace(-1.5*nrm+beta1(2),1.5*nrm+beta1(2),100);
ll = zeros(100);
for i = 1:100
    for j = 1:100
        ll(i,j) = log(Lf(gridx1(i),gridy1(j)));
    end
end
surf(gridx1,gridy1,ll','EdgeColor','none','FaceAlpha',0.75)
hold on
for j = 1:min(size(beta_path,1), 100)
    plot3(beta_path(j,1),beta_path(j,2), ...
        log(Lf(beta_path(j,1),beta_path(j,2))), ...
                     '.r','MarkerSize',25)
end
grid on

xlabel('beta1')
ylabel('beta2')

end

