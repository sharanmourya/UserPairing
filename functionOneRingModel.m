function [H,U,A] = functionOneRingModel(N,angularSpread,theta)
%INPUT:
%N             = Number of antennas
%angularSpread = Angular spread around the main angle of arrival
%
%OUTPUT:
%R             = N x N channel covariance matrix


%Approximated angular spread
Delta = angularSpread*pi/180; 

%Half a wavelength distance
D = 1/2; 

%Angle of arrival (30 degrees)
% theta = pi/6; 

%The covariance matrix has the Toeplitz structure, so we only need to
%compute the first row.
firstRow = zeros(N,1);

%Go through all columns in the first row
for col = 1:N
    
    %Distance from the first antenna
    distance = col-1;
    
    %Define integrand
    F = @(alpha)exp(-1i*2*pi*D*distance*sin(alpha+theta))/(2*Delta);
    
    %Compute the integral
    firstRow(col) = integral(F,-Delta,Delta);
    
end

%Compute the covarince matrix by utilizing the Toeplitz structure
R = toeplitz(firstRow);
[U, A] = eig(R);
% rng('default')
% s = rng;
rnd1 = rand(N,1);
% rng(1)
% s1 = rng;
rnd2 = rand(N,1);
w = rnd1 + 1j*rnd2;
% w = load('C:\Users\ronal\OneDrive\Desktop\w.mat', 'w').w;
H = U*sqrt(A)*w;