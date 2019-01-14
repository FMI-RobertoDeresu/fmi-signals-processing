% IndependentComponenetAnalysis
function ICA = ICA()
    A = [0.56, 0.79, -0.37;
         -0.75, 0.65, 0.86;
         0.17, 0.32, -0.48];

    S_len = 15000;
    S = [read_signal('semnale/fajw2.txt', S_len); 
         read_signal('semnale/fkde0.txt', S_len);
         read_signal('semnale/fsmm0.txt', S_len)];

    X = A * S;
    X = zscore(X);
    iterations = 10;

    % Kullback-Leibler
    W_KullbackLeibler = KullbackLeibler(X, iterations);
    disp('W_KullbackLeibler: ');
    disp(W_KullbackLeibler);
    P_KullbackLeibler = PerformanceIndex(W_KullbackLeibler * A);
    disp(['P_KullbackLeibler: ',num2str(P_KullbackLeibler),'.']);
    disp(' ');    disp(' ');
    
    % Max entropy
    W_MaxEntropy = MaxEntropy(X, iterations);
    disp('W_MaxEntropy: ');
    disp(W_MaxEntropy);
    P_MaxEntropy = PerformanceIndex(W_MaxEntropy * A);
    disp(['P_MaxEntropy: ',num2str(P_MaxEntropy),'.']);
    disp(' ');    disp(' ');
    
    % Jade
    W_Jade = jade(X, 3);
    disp('W_Jade: ');
    disp(inv(W_Jade));
    P_Jade = PerformanceIndex(inv(W_Jade) * A);
    disp(['P_Jade: ',num2str(P_Jade),'.']);
    disp(' ');    disp(' ');
end

function [signal] = read_signal(filename, signal_length)
	fileId = fopen(filename, 'r');
    x = fscanf(fileId, '%f');
	x = x';
	signal = x(1:signal_length);
end

% Kullback-Leibler
function W = KullbackLeibler(X, iterations)
    [c, n] = size(X);
    niu = 0.01;
    
    W = eye(c);
    W1 = W;
    
    for iteration = 1:iterations
        for t = 1:(n-1)
            Y = W1 * X(:, t);
            
            phiY = zeros(size(Y));
            for i=1:size(Y, 1)
                phiY(i, 1) = KullbackLeiblerPhi(Y(i, 1));
            end
            
            W = W + niu * (eye(c) - phiY * (Y')) * ((W^-1)');
        end
    end
end

function phi = KullbackLeiblerPhi(y)
    phi = ((-1/2) * y^5) + ((-2/3) * y^7) + ((-15/2) * y^9) + ((-2/15) * y^11) + ((112/3) * y^13) + (-128 * y^15) + ((512/3) * y^17);
end

% Max entropy
function W = MaxEntropy(X, iterations)
    [c, n] = size(X);
    niu = 0.01;
    
    W = eye(c);
            
    for iteration=1:iterations
        for t=1:(n-1)
            Y = W * X(:, t);
            
            Z = zeros(c, 1);
            for i=1:c
                Z(i, 1) = MaxEntropyZ(Y(i, 1));
            end

            W = W + niu * (eye(c) + (1-2 * Z) * (Y')) * W;
        end
    end
end

function z = MaxEntropyZ(y)
    z = 1 / (1 + exp(-y));
end

% Performance index
function P = PerformanceIndex(Q)
    [~, n] = size(Q);
    t1 = 0.0;
    for i=1:n
        maxim = max(abs(Q(i, :)));
        sum = 0.0;
        for j=1:n
            sum = sum + abs(Q(i, j)) / maxim;
        end
        sum = sum - 1;
        t1 = t1 + sum;
    end
    
    t2 = 0.0;
    for j=1:n
        sum = 0.0;
        for i=1:n
            sum = sum + abs(Q(j, i)) / max(abs(Q(:, i)));
        end
        sum = sum - 1;
        t2 = t2 + sum;
    end
    
    P = t1 + t2;    
end