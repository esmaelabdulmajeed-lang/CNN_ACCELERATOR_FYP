%week 1 task

% Minimal 2-layer CNN prototype
clear; clc;

% Step 1: Input image (8x8 matrix)
input_img = [0 0 0 0 0 0 0 0;
             0 9 9 9 0 1 1 0;
             0 9 9 9 0 1 1 0;
             0 9 9 9 0 1 1 0;
             0 0 0 0 0 0 0 0;
             1 1 0 0 2 2 2 0;
             1 1 0 0 2 2 2 0;
             0 0 0 0 0 0 0 0];

% Step 2: Conv1 (3x3 vertical edge filter)
kernel1 = [-1 0 1;
           -1 0 1;
           -1 0 1];
out1 = zeros(6,6); % (8-3+1) = 6

for i = 1:6
    for j = 1:6
        patch = input_img(i:i+2, j:j+2);   % take 3x3 patch
        out1(i,j) = sum(sum(patch .* kernel1)); % dot product
    end
end

%% Step 3: ReLU1
out1(out1 < 0) = 0;

%% Step 4: Pool1 (2x2 max pooling)
pool1 = zeros(3,3);
for i = 1:3
    for j = 1:3
        block = out1(2*i-1:2*i, 2*j-1:2*j);
        pool1(i,j) = max(block(:));
    end
end

%% Step 5: Conv2 (3x3 horizontal edge filter)
kernel2 = [-1 -1 -1;
            0  0  0;
            1  1  1];
patch = pool1(1:3,1:3); % whole 3x3
out2 = sum(sum(patch .* kernel2)); % gives 1x1 output

%% Step 6: ReLU2
if out2 < 0
    out2 = 0;
end

%% Step 7: Fully Connected layer (3 outputs)
% Flatten -> here it's just one number out2
fc_weights = [0.5; -0.2; 0.8]; % random demo weights
fc_bias = [0.1; 0; -0.1];
fc_out = fc_weights * out2 + fc_bias;

%% Show results
disp('Conv1 output (6x6):');
disp(out1);

disp('Pool1 output (3x3):');
disp(pool1);

disp('Conv2 output (1x1):');
disp(out2);

disp('Final FC outputs (3 classes):');
disp(fc_out');
