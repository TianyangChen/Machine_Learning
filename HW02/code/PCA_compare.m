function out1=PCA_compare(varargin)
% read data
data = csvread(varargin{1});
[row , col] = size(data);
% seprate data
X = data(:,1:col-1);
Y = data(:, col);
% Discriminant Analysis Classifiers, and compute 5-fold cross validation
% error
quadisc = fitcdiscr(X, Y, 'DiscrimType','quadratic');
cvmodel = crossval(quadisc,'kfold',5);
cverror = kfoldLoss(cvmodel);
fprintf('5-fold cross validation error is %f\n', cverror);
% implement PCA
[coef,score,latent,t2] = pca(X);
s=cumsum(latent)./sum(latent);
last_col = min(find(s>0.95));
transmatrix=coef(:,1:last_col);
X2 = X*transmatrix;
% Discriminant Analysis Classifiers After implementing PCA, and compute 5-fold cross validation
% error
quadisc2 = fitcdiscr(X2, Y, 'DiscrimType','quadratic');
cvmodel2 = crossval(quadisc2,'kfold',5);
cverror2 = kfoldLoss(cvmodel2);
fprintf('After PCA, 5-fold cross validation error is %f\n', cverror2);

end