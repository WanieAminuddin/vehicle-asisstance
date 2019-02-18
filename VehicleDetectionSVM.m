load VehicleDetectionquadratic264

[filename, pathname] = uigetfile('*.*','Browse image'); %select Input video
file = strcat(pathname, filename);
im = imread (file);
[w h d] = size(im);
counter = 1;
level = 4;
scale = 0.5;

windowHeight = 64;
windowWidth = 64;

framegray = rgb2gray(im);
framegray = imadjust(framegray);

for y = 1 : windowWidth/2 : w - windowHeight
    for x = 1 : windowHeight/2 : h - windowWidth
      p1 = [x,y];
      p2 = [x + (windowWidth - 1), y + (windowHeight - 1)];
      po = [p1; p2] ;

      % Croped image and scan it.
      crop_px = [po(1,1) po(2,1)];
      crop_py  = [po(1,2) po(2,2)];

      topLeftRow = ceil(min(crop_px));
      topLeftCol = ceil(min(crop_py));

      bottomRightRow = ceil(max(crop_px));
      bottomRightCol = ceil(max(crop_py));

      cropedImage = framegray(topLeftCol:bottomRightCol,topLeftRow:bottomRightRow,:);

      % Get the feature vector from croped image using HOG descriptor
      featureVector{counter,:} = extractHOGFeatures(cropedImage);
      boxPoint{counter,:} = [x, y];
      counter = counter + 1;
      x = x + 1;
    end 
end

% label = ones(length(featureVector),1);
P = cell2mat(featureVector);

predictions = svmclassify(VehicleDetectionquadratic264, P); 
vehicle = find(ismember(predictions,'VEHICLE'));

vehiclepoint = cell2mat(boxPoint(vehicle));
vehiclepoint(:,3) = 100; vehiclepoint(:,4) = 100;

xmin = vehiclepoint(:,1);
ymin = vehiclepoint(:,2);
xmax = xmin + vehiclepoint(:,3) - 1;
ymax = ymin + vehiclepoint(:,4) - 1;

overlapRatio = bboxOverlapRatio(vehiclepoint, vehiclepoint);

n = size(overlapRatio,1);
overlapRatio(1 : n + 1 : n ^ 2) = 0;
g = graph(overlapRatio);
componentIndices = conncomp(g);

% Merge the boxes based on the minimum and maximum dimensions.
xmin = accumarray(componentIndices', xmin, [], @min);
ymin = accumarray(componentIndices', ymin, [], @min);
xmax = accumarray(componentIndices', xmax, [], @max);
ymax = accumarray(componentIndices', ymax, [], @max);

textBBoxes = [xmin ymin xmax - xmin + 1 ymax - ymin + 1];
numRegionsInGroup = histcounts(componentIndices);
textBBoxes(numRegionsInGroup < 1, :) = [];
textBBoxes(textBBoxes(:,4) > 200, :) = [];



RGB = insertShape(im, 'Rectangle', textBBoxes, 'LineWidth', 5);
imshow (RGB);      


