function weight = logisticR(varargin)
%read data
data = csvread(varargin{1});
[row , col] = size(data);
X = data(:,1:col-1);
Y = data(:,col);
alpha = 0.001;
maxCycle = varargin{2};% set iteration times
weight = ones(col-1,1);
for i = 1:maxCycle
    h = sigmoid((X * weight)');
    error = (Y - h');
    weight = weight + alpha * X' * error;
end
end

function returnVals = sigmoid(inX)
returnVals = 1.0./(1.0+exp(-inX));
end