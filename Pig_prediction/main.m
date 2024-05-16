clc;
clear;
close all;
%% 
% 读取 Excel 文件
filename = 'data.xlsx';
data = xlsread(filename);

% 提取特征和目标变量
X = data(:, 2:4);  % 长宽高作为特征
y = data(:, 5);    % 体重作为目标变量


%% 
% % 你的猪的数据，假设是 (长, 宽, 高, 体重)
% data = [
%     1.5, 0.8, 0.6, 20;
%     1.7, 0.9, 0.7, 25;
%     1.6, 0.85, 0.65, 22;
%     % 添加更多数据...
% ];
% 
% % 提取特征和目标变量
% X = data(:, 1:3);  % 长宽高作为特征
% y = data(:, 4);    % 体重作为目标变量

% 初始化 RANSACRegressor 模型
ransac = robustfit(X, y);

% 预测体重
predicted_weights = ransac(1) + X * ransac(2:end);

% 输出拟合得到的模型参数
fprintf('RANSAC模型的截距: %f\n', ransac(1));
fprintf('RANSAC模型的系数: ');
disp(ransac(2:end));

% 可视化拟合结果
scatter(X(:, 1), y, 'DisplayName', 'Ground Truth');
hold on;
scatter(X(:, 1), predicted_weights, 'r', 'DisplayName', 'Predicted');
xlabel('Length');
ylabel('Weight');
legend();



%% 
% 假设已经拟合好了模型并得到了预测值 predicted_weights

% 计算均方根误差
mse = mean((predicted_weights - y).^2);
rmse = sqrt(mse);
fprintf('均方根误差（RMSE）: %f\n', rmse);

% 画出误差分布图
errors = predicted_weights - y;
figure;
histogram(errors, 'BinWidth', 1);  % 指定直方图的宽度为1
xlabel('Prediction Error');
ylabel('Frequency');
title('Prediction Error Distribution');



weightstd=std(data(:,5));