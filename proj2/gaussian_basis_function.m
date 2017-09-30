% Used to generate the 9 Gaussian basis functions for Figure 3.8
function y = gaussian_basis_function(x,u,s)
    % Use element-wise exponentiation in case x is a vector
    y = exp(-(x-u).^2/(2*s^2));
end