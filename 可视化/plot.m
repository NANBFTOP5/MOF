clc;
clear;
% close all
% figure; data = load('test.out');    
figure; data = load('dat.txt');    
[row, col] = size(data); 
mesh(data);xlim([1,col]);ylim([1,row]);
set(gcf, 'Position', get(0, 'Screensize'));
title('sph', 'FontSize', 24)

