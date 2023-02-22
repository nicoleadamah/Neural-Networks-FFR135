%ANN HW2 2022 Restricted Boltzmann Machine Nicole Adamah
close  
clear 
clc

N = 3; %number of visible neurons
M = [1,2,4,8];%number of hidden neurons
eta = 0.005;
inputs = unique(nchoosek(repmat([-1,1], 1, 3), N), 'rows');%combinations for XOR inputs
P = [1/4 0 0 1/4 0 1/4 1/4 0]';% P(x) = 1/4 for x-inputs: 1, 4, 6, 7.
in = inputs([1, 4, 6, 7],:);
total_x = length(inputs);
minibatches = 20;
k = 200;
trials = 1000;
DKL = zeros(5, length(M));

for Counts = 1:5

for nrneurons=1:length(M)
        % Initilize weights, thresholds and states
        w = normrnd(0,1,[M(nrneurons),N]);
        t_v = zeros(1, N);
        t_h = zeros(M(nrneurons), 1);
        v = zeros(1,N);
        h = zeros(1, M(nrneurons));

        for trial = 1:trials
            dw = zeros(M(nrneurons), N);
            dt_v = zeros(1, N);
            dt_h = zeros(M(nrneurons), 1);
            for i = 1:minibatches
                mu = randi(4);
                v0 = in(mu,:);
                b_0 = (w * v0') - t_h;
                % Updating hidden neurons
                for j = 1:M(nrneurons)
                    Pr = 1/(1+exp(-2*b_0(j)));
                    r = rand(1);
                    if r < Pr
                        h(j) = 1;
                    else
                        h(j) = -1;
                    end 
                end
                % Updating visible neurons
                for t = 1: k
                    b_v = (h * w) - t_v; 
                    for j2 = 1:length(b_v)
                        Pr = 1/(1+exp(-2*b_v(j2)));
                        r = rand(1);
                        if r < Pr
                            v(j2) = 1;
                        else
                            v(j2) = -1;
                        end
                    end
                    % Updating hidden neurons
                    b_h = (v * w')' - t_h;
                    for j3 = 1:M(nrneurons)
                        Pr = 1/(1+exp(-2*b_h(j3)));
                        r = rand(1);
                        if r < Pr
                            h(j3) = 1;
                        else
                            h(j3) = -1;
                        end
                    end
                end
                dt_v = dt_v - eta*(v0-v);
                dt_h = dt_h - eta*(tanh(b_0)-tanh(b_h));
                dw = dw + eta*((tanh(b_0)*v0) - tanh(b_h)*v);
            end
            t_v = t_v + dt_v;
            t_h = t_h + dt_h;
            w = w + dw;
        end % Done with training

        % Iterating over all x-inputs
        outer = 10^3;
        inner = 10^2;
        Pb = zeros(total_x,1);
        T = outer*inner;
        % Outer, updating hidden neurons
        for i = 1:outer
            idx = randi(total_x);
            v = inputs(idx, :)';
            b_01 = (w * v) - t_h;
            for j = 1:M(nrneurons)
                Pr = 1/(1+exp(-2*b_01(j)));
                r = rand(1);
                if r < Pr
                    h(j) = 1;
                else
                    h(j) = -1;
                end
            end
            % Inner, updating visible neurons
            for i2 = 1:inner
                b_v1 = (h * w) - t_v;
                for j3 = 1:length(b_v1)   
                    Pr = 1/(1+exp(-2*b_v1(j3)));
                    r = rand(1);
                    if r < Pr
                        v(j3) = 1;
                    else
                        v(j3) = -1;
                    end
                end       
                % Updating hidden neurons
                b_h1 = (w * v) - t_h;
                for j4 = 1:M(nrneurons)  
                    Pr = 1/(1+exp(-2*b_h1(j4)));
                    r = rand(1);
                    if r < Pr
                        h(j4) = 1;
                    else
                        h(j4) = -1;
                    end
                end

                % Check for convergence, when the vectors are the same
                for j5 = 1 : total_x
                    x_val = inputs(j5,:);
                    if isequal(v', x_val)
                        Pb(j5) = Pb(j5) + 1/T;
                    end
                end
            end 
        end
    % Calculations for the Kullback-Leibler divergence    
    DKL_val = 0;
    for i = 1:total_x
        if (P(i) ~= 0)
            DKL_val = DKL_val + (P(i) * (log(P(i))-log(Pb(i))));
        end
    end
    DKL(Counts,nrneurons) = DKL_val;
end
disp(Counts)
end
%% Plotting the experimental DKL 
DKLPlot = zeros(1,4);
for i = 1:4
    DKLPlot(i) = mean(DKL(:,i));
end
figure
plot(M,DKLPlot,'ro')
hold on

% Calculating and plotting the theoretical DKL
M_i = 0:10;
DKL_real = zeros(length(M_i),1);
for i = 1 : length(M_i)
    if M_i(i) < 2^(N-1)-1
        DKL_real(i) = N - (log2(M_i(i)+1)) - (M_i(i)+1)/(2^(log2(M_i(i)+1)));
    else
        DKL_real(i) = 0;
    end
end
plot(M_i, DKL_real, 'k-', 'LineWidth',2)
title('Kullback-Leiber divergence theoretical vs experimental','FontSize',16)
legend('Experimental D_{KL}', 'Theoretical D_{KL}')
xlabel('M')
ylabel('D_{KL}')
hold on


 