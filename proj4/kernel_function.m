% Kernel function used on P307, 6.63
% Argument theta is a 4 element vector
function k = kernel_function(xn,xm,theta)
	k = theta(1)*exp(-theta(2)/2*sum((xn-xm).^2)) + theta(3) + theta(4)*xn'*xm;
end