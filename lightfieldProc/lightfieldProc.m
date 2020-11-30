clear all;
close all;
clc

% read in .lfr file, processed by Lytro desktop
% imgPath = 'data/IMG_3421.LFR';
imgPath = 'data/jacaranda.lfp';
LFP = LFReadLFP(imgPath);
imgRAW = LFP.RawImg;

img = demosaic(imgRAW,LFP.DemosaicOrder);                               % demosaic
img = double(img) ./ (2^LFP.Metadata.image.pixelPacking.bitsPerPixel);  % normalize by pixel bits

% color adjustment (non-linear)
ColMatrix = reshape(LFP.Metadata.image.color.ccm, 3,3);
ColBalance = [1,...
    LFP.Metadata.image.color.whiteBalanceGain.r,...
    LFP.Metadata.image.color.whiteBalanceGain.b];
Gamma = 1;
imgCorr = LFColourCorrect(img,ColMatrix,ColBalance,Gamma);

% adjust each lenslet pixel, assum span 16*16
% [H,W,C] = size(img);          % 5368*7728
% lenSize = 12;                 % lenslet size
% S = floor(H/lenSize);
% T = floor(W/lenSize);
% imgAdj = zeros(S*lenSize,T*lenSize,C);
% for s = 1:S-1
%     if mod(s,2) ~= 0
%         imgAdj(s*16:(s+1)*16,:,:) = img(s*16+8:(s+1)*16+8,1:end-8,:);
%     else
%         imgAdj(s*16:(s+1)*16,:,:) = img(s*16+8:(s+1)*16+8,9:end,:);
%     end
% end

% figure(1)
imshow(imgCorr)
% figure(2)
% imshow(imgAdj)
% save lightfield image, I: (u*s) * (v*t) * 3
% imwrite(img,"lightfield_adj.png")
% imwrite(imgCorr,"lightfield.png")

