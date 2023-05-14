M = 100; % antennas in ULA
K = 100; % no of users

H = zeros(M, K);
U = zeros(K, M, M);
A = zeros(K, M, M);
radius = 1;
d_min = 0;
d_max = 200;

for iter = 1:1000
    iter
    rnd1 = rand(1,K);
    d = (d_max-d_min)*rnd1 + d_min;

    theta_min = 0;
    theta_max = 2*pi;
    rnd2 = rand(1,K);
    rnd3 = rand(1,K);
    theta = rnd2;
    scale = floor(rnd3(1)*10)+10;
    
    for i = 1:K
        if rnd2(i)<=0.33
            theta(i) = 2*pi*rnd3(i);
        end
        if rnd2(i)<=0.66 && rnd2(i)>0.33
    %         theta(i) = pi/12*rnd3(i) + 2*pi/3-pi/24;
    %         theta(i) = pi/12*rnd3(i) + pi/8-pi/24;
            theta(i) = 2*pi*rnd3(i) + 2*pi/3;
        end
        if rnd2(i)<=1 && rnd2(i)>0.66
    %         theta(i) = pi/12*rnd3(i) + 4*pi/3-pi/24;
    %         theta(i) = pi/12*rnd3(i) - pi/8-pi/24;
            theta(i) = 2*pi*rnd3(i) - 2*pi/3;
        end
    end

    AS = 15;

    for i = 1:K
        [H(:,i), U(i,:,:), A(i,:,:)] = functionOneRingModel(M, AS, theta(i));
    end
    filename1 = sprintf('F:\\dataset_100\\H%d.mat', iter);
    filename2 = sprintf('F:\\dataset_100\\U%d.mat', iter);
    filename3 = sprintf('F:\\dataset_100\\A%d.mat', iter);
    save(filename1, 'H')
    save(filename2, 'U')
    save(filename3, 'A')
    x = d.*cos(theta);
    y = d.*sin(theta);
    filename4 = sprintf('F:\\dataset_100\\x%d.mat', iter);
    filename5 = sprintf('F:\\dataset_100\\y%d.mat', iter);
    save(filename4, 'x')
    save(filename5, 'y')
end

scatter(x,y)

figure;
grid on;
axis([min(x)-1, max(x)+1, min(y)-1, max(y)+1])
for k=1:length(x)
    text(x(k),y(k),num2str(k))
end

