function [y, XS, tau] = avg_gtw(X, max_iter_num)

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Generalized time warping algorithm for DTW averaging
    % Inputs
    %     X: input time-series; it is an N-by-T matrix where N is the 
    %        number of time-series and T is the length of them. Note that 
    %        this algorithm assumes all time-series have the same length. 
    %        If your time-series have different lengths, use an interpolation
    %        technique (e.g., spline function) to make them equi-length.
    %     order: order of the DST used to model the warping functions. With
    %            higher order, we consider more details of the input signals 
    %            and we learn a more complex warping functions. However, a 
    %            high order value will caouse a more comples optimization
    %            landscape, which increases the likelihood of converging to
    %            a weak local optimum point.
    %    max_iter_num: maxumum number of iterations in our gradient based
    %                  optimization algorithm
    % 
    % Outputs
    %    y: final average signal. It is a 1-by-T vector.
    %    XS: synchronized time-series. y is equal to mean(XS, 1).
    %    tau: learned warping functions. It is an N-by-T matrix. tau(n, :)
    %         is the warping function used to obtain XS(n, :) from X(n, :).
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
    % Written by Soheil Khorram (khorram.soheil@gmail.com)
    % Please feel free to contact me if you have any question.
    %
    % @article{khorram2019trainable,
    %   title={Trainable Time Warping: Aligning Time-Series in the Continuous-Time Domain},
    %   author={Khorram, Soheil and McInnis, Melvin G and Provost, Emily Mower},
    %   journal={arXiv preprint arXiv:1903.09245},
    %   year={2019}
    % }
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    prm.K = 20;
    prm.iter_num = max_iter_num;
    [prm.N, prm.T] = size(X);
    [prm.min_tau, prm.max_tau] = create_max_min_tau(prm);
    f = [];
    for k = 1:1:7
        t = (0:(prm.T-1)) / (prm.T-1);
        t = (prm.T-1) * (t.^(2/(k+2))) + 1;
        f = [f; t];
    end
    for k = 1:1:7
        t = (0:(prm.T-1)) / (prm.T-1);
        t = (prm.T-1) * (t.^((k+2)/2)) + 1;
        f = [f; t];
    end
    for k = 1:1:5
        t = (0:(prm.T-1)) / (prm.T-1);
        t = (prm.T-1) * ((tanh(8*t - 7 + k) + 1)/2) + 1;
        f = [f; t];
    end
    t = (0:(prm.T-1)) / (prm.T-1);
    t = (prm.T-1) * (t) + 1;
    f = [f; t];
    prm.f = f;
    prm.K = size(prm.f, 1);
    alpha0 = ones(prm.N, prm.K) / prm.K;
%     alpha0 = alpha0 ./ (repmat(sum(alpha0, 2), 1, prm.K));
%     alpha0 = zeros(prm.N, prm.K);
%     alpha0(end, :) = 1;
    opt = optimset('GradObj','on', 'MaxIter', prm.iter_num);
    alpha_opt = fminunc(@(alpha)calc_mse_grad(alpha, X, prm),alpha0,opt);
    [~, ~, y, XS, tau] = forward_backward(alpha_opt, X, prm);
end

function [mse, d_alpha, y, XS, tau] = forward_backward(alpha, X, prm)
    alpha_abs = abs(alpha);
    alpha_abs(alpha_abs < 0.00001) = 0.00001;
%     alpha_abs_n = alpha_abs ./ (repmat(sum(alpha_abs, 2), 1, prm.K));
    alpha_abs_n = alpha_abs;
    tau = alpha_abs_n * prm.f;
    tau = max(tau, 1);
    tau = min(tau, prm.T);
    max_inds = (tau > prm.max_tau);
    min_inds = (tau < prm.min_tau);
    tau(max_inds) = prm.max_tau(max_inds);
    tau(min_inds) = prm.min_tau(min_inds);
    tau = correct_T(tau);
    tauF = floor(tau);
    tauC = ceil(tau);
    XF_index = min(max(tauF, 1), prm.T);
    XC_index = min(max(tauC, 1), prm.T);
    XF = X;
    XC = X;
    for n = 1:prm.N
        XF(n, :) = X(n, XF_index(n, :));
        XC(n, :) = X(n, XC_index(n, :));
    end
    XS = (tau-tauF).*XC + (tauC-tau).*XF;
    y = mean(XS, 1);
    e = repmat(y, prm.N, 1) - XS;
    mse = mean2(e.^2);
    d_tau = (-2/(prm.N * prm.T)) * (e .* (XC-XF));
    d_alpha_abs_n = d_tau * prm.f';
%     alpha_abs_sum = sum(alpha_abs, 2);
%     d_alpha_abs = ((d_alpha_abs_n .* repmat(alpha_abs_sum, 1, prm.K)) - repmat(sum((d_alpha_abs_n .* alpha_abs), 2), 1, prm.K)) ./ repmat(alpha_abs_sum.^2, 1, prm.K);
    d_alpha_abs = d_alpha_abs_n;
    d_alpha = d_alpha_abs .* sign(alpha);
%     disp(['mse: ' num2str(mse)]);
end

function [cost, grad] = calc_mse_grad(alpha, X, prm)
    [mse, d_alpha, ~, ~, ~] = forward_backward(alpha, X, prm);
    alpha_cost = alpha;
    alpha_cost(:, end) = 0;
    cost = mse + 0.1 * mean2(alpha_cost.^2);
    grad = d_alpha + 0.1 * 2 * alpha_cost;
end

function T = correct_T(T)
    TF = floor(T);
    TC = ceil(T);
    T = T + (TF == TC).*1/100000;
end

function [min_tau, max_tau] = create_max_min_tau(prm)
    pmin = 0.1;
    pmax = 0.9;
    T = prm.T;
    t = 1:T;
    t = (t-1)/(T-1);
    max_tau = zeros(1, T);
    max_tau(t <= pmin) = pmax * t(t <= pmin) / pmin;
    max_tau(t > pmin) = (t(t > pmin) - pmax * t(t > pmin) + pmax - pmin) / (1 - pmin);
    max_tau = max_tau * (T-1) + 1;
    min_tau = zeros(1, T);
    min_tau(t <= pmax) = pmin * t(t <= pmax) / pmax;
    min_tau(t > pmax) = (t(t > pmax) - pmin * t(t > pmax) + pmin - pmax) / (1 - pmax);
    min_tau = min_tau * (T-1) + 1; 
    max_tau = repmat(max_tau, prm.N, 1);
    min_tau = repmat(min_tau, prm.N, 1);
end
