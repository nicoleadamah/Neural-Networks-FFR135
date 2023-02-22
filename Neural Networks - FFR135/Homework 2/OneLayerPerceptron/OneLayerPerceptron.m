%ANN HW2 2022 One Layer Perceptron Nicole Adamah
close
clear
clc
% Load trainging and validation data
data = readmatrix('training_set.csv');
val = readmatrix('validation_set.csv');

x1 = normalize(data(:, 1));
x2 = normalize(data(:, 2));
x_inputs = [x1, x2];
t = data(:, 3);

x1_val = normalize(val(:, 1));
x2_val = normalize(val(:, 2));
x_val = [x1_val, x2_val];
t_val = val(:, 3);

% Initializing weights, thresholds and other parameters 
M1 = 15;
C = 1;
eta = 0.005;
epochs = 10^3;
w1 = randn([2, M1]);
w2 = randn([1, M1]);
t1 = zeros(1,M1); 
t2 = 0;



for epoch = 1:epochs
    for m = 1:length(x_inputs)
    %Iterating over the training set
    mu = randi(length(x_inputs));
    pattern = x_inputs(mu, :);
	% Calculations for hidden layer, V
    V = tanh((pattern * w1) - t1);
    % Calculations for Output
	Output = tanh(sum(w2 * V') - t2);
    
    % Back propagation in order to update the weights and thresholds
    delta2 = (t(mu) - Output)*(1 - (tanh(dot(w2, V) - t2)^2));    
    delta1 = (w2' * delta2) .* (1 - (tanh(pattern * w1 - t1).^2))';
	% Updating the weights and threshholds.
    w2 = w2 + (eta * delta2 * V);
    t2 = t2 - eta * delta2;
	w1 = w1 + eta * (delta1 * pattern)';
	t1 = t1 - eta * delta1';
    end
    % Iterating over the validation set
    pval = length(val);
    errorcalc = 0;
    for mu = 1:pval
        pattern = x_val(mu, :);
        V = tanh((pattern * w1) - t1);
        Output = tanh(sum(w2 * V') - t2);
        errorcalc = errorcalc + abs(sign(Output) - t_val(mu));
    end
    C = errorcalc/(2*pval);
    disp(epoch);
    disp(C) 
    
    if C < 0.12
        csvwrite('w1.csv', w1');
        csvwrite('w2.csv', w2');
        csvwrite('t1.csv', t1');
        csvwrite('t2.csv', t2);
        break
    end
end

