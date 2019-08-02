function [y, XS, tau] = avg_fltw(X, order, max_iter_num)

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Fast learnable time warping algorithm for DTW averaging
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

    if nargin < 3
        max_iter_num = 500;
    end
    if nargin < 2
        order = 8;
    end
    prm_K = order;
    [prm_N, prm_T] = size(X);
    f = (1:(prm_K))' * (1:(prm_T-1)) * pi / (prm_T);
    f = sin(f);
    alpha0 = zeros(prm_N, prm_K);
    opt = optimset('GradObj','on', 'MaxIter', max_iter_num);
    alpha_opt = fminunc(@(alpha)calc_mse_grad(alpha, X, f, prm_T, prm_N, prm_K),alpha0,opt);
    [~, ~, y, XS, tau] = forward_backward(alpha_opt, X, f, prm_T, prm_N, prm_K);
end

function [mse, d_alpha, y, XS, tau] = forward_backward(alpha, X, f, prm_T, prm_N, prm_K)
    alphan = alpha - repmat(mean(alpha, 1), prm_N, 1);
    beta = alphan * f + 1;
    gamma = abs(beta);
    p = (prm_T-1) * gamma ./ repmat(sum(gamma, 2), 1, prm_T-1);
    p = [ones(prm_N, 1) p];
    tau = cumsum(p,2);
    tau = correct_T(tau);
    tauF = floor(tau);
    tauC = ceil(tau);
    XF_index = min(max(tauF, 1), prm_T);
    XC_index = min(max(tauC, 1), prm_T);
    XF = X;
    XC = X;
    for n = 1:prm_N
        XF(n, :) = X(n, XF_index(n, :));
        XC(n, :) = X(n, XC_index(n, :));
    end
    XS = (tau-tauF).*XC + (tauC-tau).*XF;
    y = mean(XS, 1);
    e = repmat(y, prm_N, 1) - XS;
    mse = mean2(e.^2);
    d_tau = (-2/prm_N) * (e .* (XC-XF));
    d_tau = d_tau(:, 2:end);
    d_p = cumsum(d_tau, 2, 'reverse');
    gamma_sum = sum(gamma, 2);
    d_gamma = ((prm_T-1) / prm_T) * ((d_p .* repmat(gamma_sum, 1, prm_T-1)) - repmat(sum((d_p .* gamma), 2), 1, prm_T-1)) ./ repmat(gamma_sum.^2, 1, prm_T-1);
    d_beta = d_gamma .* sign(beta);
    d_alphan = d_beta * f';
    d_alpha = d_alphan - repmat(mean(d_alphan, 1), prm_N, 1);
%     disp(['mse: ' num2str(mse)])
end

function [mse, grad] = calc_mse_grad(alpha, X, f, prm_T, prm_N, prm_K)
    [mse, grad, ~, ~, ~] = forward_backward(alpha, X, f, prm_T, prm_N, prm_K);
end

function T = correct_T(T)
    TF = floor(T);
    TC = ceil(T);
    T = T + (TF == TC) .* (1e-5);
end
