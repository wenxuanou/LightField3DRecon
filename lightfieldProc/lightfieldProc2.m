close all;
clear all;
clc;
addpath './doda42-LFToolbox-5dd4a8a';
LFMatlabPathSetup;

% read in .lfr file, processed by Lytro desktop
% imgPath = 'data/Jacaranda.lfp';
imgPath = 'data/IMG_3422.lfr';
whiteImgPath = './data';

LFUtilProcessWhiteImages(whiteImgPath)

DecodeOptions = LFDefaultField( 'DecodeOptions', 'WhiteImageDatabasePath',...
    fullfile(whiteImgPath,'WhiteImageDatabase.mat'));
[LF, LFMetadata, WhiteImageMetadata, LensletGridModel, ~] = ...
        LFLytroDecodeImage(imgPath,DecodeOptions);


LF = LF(:,:,5:end,2:end-2,1:3);     % remove corrupted edges, remove 
% LF = LF(:,:,5:end,2:end-2,:);     % remove corrupted edges
save('lightfield.mat','LF')

% % rebuild raw image
% [U,V,S,T,C] = size(LF);
% imgRecon = zeros(U*S,V*T,3);
% 
% for s = 0:S-1
%     for t = 0:T-1
%         temp = LF(:,:,s+1,t+1,:);
%         imgRecon(s*U+1:(s+1)*U,t*V+1:(t+1)*V,:) = reshape(temp,[U,V,3]);
%     end
% end
% 
% imshow(imgRecon)
