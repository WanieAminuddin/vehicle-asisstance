tic; 
HOGdetector = vision.CascadeObjectDetector('HOG.xml');

%% Video Initialization
[filename, pathname] = uigetfile('*.*','Browse image'); %select Input video
file = strcat(pathname, filename);
im = imread (file);

%%

framegray = rgb2gray(im);
framegray = imadjust(framegray);

HOGD = step(HOGdetector,framegray);

RGB = insertShape(im, 'Rectangle', HOGD, 'LineWidth', 3);
imshow (RGB);      

toc;
Timespent = toc;