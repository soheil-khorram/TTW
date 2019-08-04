function [y, XS, tau] = avg_ttw(X, order, max_iter_num)

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Trainable time warping algorithm for DTW averaging
    % Inputs
    %     X: input time-series; it is an N-by-T matrix where N is the 
    %        number of time-series and T is the length of them. Note that 
    %        this algorithm assumes all time-series have the same length. 
    %        If your time-series have different lengths, use an interpolation
    %        technique (e.g., spline function) to make them equi-length.
    %     order: order of the DST used to model the warping functions. With
    %            higher order, we consider more details of the input signals 
    %            and we learn more complex warping functions. However, a 
    %            high order value will cause a more complex optimization
    %            landscape, which increases the likelihood of converging to
    %            a weak local optimum point.
    %    max_iter_num: maximum number of iterations in our gradient-based
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

    prm = struct;
    prm.iter_num = max_iter_num;
    prm.K = order;
    [prm.N, prm.T] = size(X);
    f = (1:prm.K)' * (0:(prm.T-1)) * pi / (prm.T-1);
    f = sin(f);
    prm.f = f;    
    alpha0 = zeros(prm.N, prm.K);
    opt = optimset('GradObj','on', 'MaxIter', prm.iter_num);
    alpha_opt = fminunc(@(alpha)calc_mse_grad(alpha, X, prm), alpha0, opt);
%     alpha_opt = fminunc_adam(@(alpha)calc_mse_grad(alpha, X, prm), alpha0);
    [~, ~, y, XS, tau] = forward_backward(alpha_opt, X, prm);
end

function [mse, d_alpha, y, XS, tau] = forward_backward(alpha, X, prm)
    alpha = alpha - repmat(mean(alpha, 1), prm.N, 1);
    tau = alpha * prm.f + ones(prm.N, 1) * (1:prm.T);
    tau = min(tau, prm.T);
    tau = max(tau, 1);
    tau = make_tau_increasing(tau);
    tauF = floor(tau);
    X_ = cell(41,1);
    for m = -40:40
        X_{m+41} = X;
    end
    for n = 1:prm.N
        for m = -40:40
            X_{m+41}(n, :) = X(n, min(max(tauF(n, :) + m, 1), prm.T));
        end
    end
    XS = zeros(prm.N, prm.T);
    for m = -40:40
        XS = XS + X_{m+41} .* sinc(m - tau + tauF);
    end
    y = mean(XS, 1);
    e = XS - repmat(y, prm.N, 1);
    mse = mean2(e.^2);
    d_e = (2/(prm.N*prm.T)) * e;
    d_tau = zeros(prm.N, prm.T);
    for m = -40:40
        d_tau = d_tau - X_{m+41} .* d_sinc(m - tau + tauF);
    end
    d_tau = d_tau .* d_e;
    d_tau(:, 1) = 0;
    d_tau(:, prm.T) = 0;
    d_alpha = d_tau * prm.f';
    d_alpha = d_alpha - repmat(mean(d_alpha, 1), prm.N, 1);
%     disp(['mse: ' num2str(mse)])
end

function [mse, grad] = calc_mse_grad(alpha, X, prm)
    [mse, grad, ~, ~, ~] = forward_backward(alpha, X, prm);
end

function best_teta = fminunc_adam(calc_cost_grad_func, teta0)
    teta = teta0;
    best_cost = 1e20;
    best_teta = teta0;
    patient = 0;
    alpha = 0.01;
    beta1 = 0.5;
    beta2 = 0.5;
    ep = 1e-8;
    m = 0;
    v = 0;
    for t = 1:100000
        if(rem(t, 1000) == 0)
            disp(['best_cost = ' num2str(best_cost)]);
        end
        [cost, grad] = calc_cost_grad_func(teta);
        if cost < best_cost
            patient = 0;
            best_cost = cost;
            best_teta = teta;
        else
            patient = patient + 1;
        end
        if patient > 100
            alpha = alpha / 2;
            patient = 0;
            teta = best_teta;            
        end
        if alpha < 0.0001
            break;
        end
        m = beta1 * m + (1-beta1) * grad;
        v = beta2 * v + (1-beta2) * (grad.^2);
        m = m / (1 - (beta1^t));
        v = v / (1 - (beta2^t));
        teta = teta - alpha * m ./ (sqrt(v) + ep);
    end
end

function tau = make_tau_increasing(tau)
    
    N = size(tau, 1);
    while 1==1
        incorrect = (tau(:, 2:end) - tau(:, 1:end-1)) < 0;
        if sum(sum(incorrect)) == 0
            break;
        end
        z_incorrect = [(zeros(N, 1) == 1) incorrect];
        incorrect_z = [incorrect (zeros(N, 1) == 1)];
        tau1 = tau;
%         tau1(z_incorrect) = (tau(incorrect_z) + tau(incorrect_z))/2;
%         tau1(incorrect_z) = (tau(incorrect_z) + tau(incorrect_z))/2;
        tau1(z_incorrect) = tau(incorrect_z);
        tau = tau1;
    end
end

function y = d_sinc(x)
    y = (cos(pi*x) - sinc(x)) ./ (x + 1e-20);
    y(abs(x) < 1e-6) = 0;
end
