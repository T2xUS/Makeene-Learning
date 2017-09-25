% Function used in conjunction with slider to update Gamma posterior plot
function update_plot_gamma(i,N,prec_vec,s,sigma_sq_ML_norm,a_o,b_o)
    % Update parameters
    a_N = a_o(s) + N(i)/2;
    b_N = b_o(s) + N(i)/2*sigma_sq_ML_norm(i);
    % DEBUG, check if ML and the gamma mean converges to 1
    %ML = sigma_sq_ML_norm(i)
    %gamma = b_N/a_N
    %mean_gamma = b_N/a_N
    % Update distribution
    % MATLAB EXPECTS 1/b instead of b AS A PARAMETER WTF
    p_post = pdf('gamma',prec_vec,a_N,1/b_N);
    % "Plot" graph
    plot(prec_vec,p_post)
    title(sprintf('Gamma Posterior Density (a_{0}=%.3f, b_{0}=%.3f)',a_o(s),b_o(s)))
    xlabel('Precision')
    ylabel('Density')
    axis([0 5 0 10])
    legend(['N = ' num2str(N(i))])
end