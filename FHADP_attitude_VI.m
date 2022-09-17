function FHADP_attitude_VI()
% system matrix

% time step
T = 0.1;

% system model and parameter
w0 = 0.0011;
Jx = @(k) 1070-15.5*(k*T);
Jy = @(k) 2150-11*(k*T);
Jz = @(k) 1300-7.5*(k*T);
d = @(k) w0*(Jy(k)-Jx(k)-Jz(k))/Jx(k);
k1 = @(k) 4*w0^2*(Jy(k)-Jz(k))/Jx(k);
k2 = @(k) 3*w0^2*(Jx(k)-Jz(k))/Jy(k);
k3 = @(k) w0^2*(Jy(k)-Jx(k))/Jz(k);

A = @(k) eye(6)+...
    [zeros(3), eye(3);
    0, 0, -d(k), -k1(k), 0, 0;
    0, 0, 0, 0, -k2(k), 0;
    d(k), 0, 0, 0, 0, -k3(k);];
B = @(k) [zeros(3);
    diag([1/Jx(k),1/Jy(k),1/Jz(k)])];

% total length
N = 300;

% get dimensions
[n,m] = size(B(1));

% reward weighting matrices
Q = @(t) 10*eye(n);
% Q = @(t) [0.1*eye(3),zeros(3);zeros(3),10*eye(3)];
R = @(t) eye(m);
F = eye(n);
% Q = @(t) [0.04*t+2 0;0 0.04*t+2];
% R = @(t) 5-0.02*t;
% F = [1 0; 0 1]*10;
% Q = @(t) [exp(-0.4*t)+1 0;0 exp(-0.8*t)+1];
% R = @(t) 1-exp(-0.2*t);
% F = [10 0; 0 10];

% get true cost matrix
P = zeros(n,n,N);
K = zeros(m,n,N);
H1tr = zeros(n,n,N-1);
H2tr = zeros(m,n,N-1);
H3tr = zeros(m,m,N-1);
P(:,:,N) = F;
for i = 1:N-1
    t = N-i;
    P(:,:,t) = A(t)'*P(:,:,t+1)*A(t)+Q(t)-A(t)'*P(:,:,t+1)*B(t)*...
        inv(R(t)+B(t)'*P(:,:,t+1)*B(t))*B(t)'*...
        P(:,:,t+1)*A(t);
    K(:,:,t) = inv(R(t)+B(t)'*P(:,:,t+1)*B(t))*B(t)'*P(:,:,t+1)*A(t);
    H1tr(:,:,t) = A(t)'*P(:,:,t+1)*A(t);
    H2tr(:,:,t) = B(t)'*P(:,:,t+1)*A(t);
    H3tr(:,:,t) = B(t)'*P(:,:,t+1)*B(t);
end

% data collection rounds
l = (m*(m+1)/2+m*n+n*(n+1)/2);

% raw data
% Init = load('True.mat');
% L_init = Init.L(:,:,:,6);
% L_init = 0.3*ones(m,n,N);
L_init = zeros(m,n,N);
xtr = zeros(n,N,l);
utr = zeros(m,N,l);

% training data collection
for i=1:l
    % inital state
    xq = -1+ 2*rand(n/2,1);
    xqdot = -0.01+ 0.02*rand(n/2,1);
    xtr(:,1,i) = [xq;xqdot];
    % exploration noise params
    ww = (-500 + (500-(-500)).*rand(m,500));
    mm = 2;%(5 + (3-1).*rand(1,1));
    for j=1:N-1
        t = j;
        % exploration noise
        u_rand = mm*sum(sin(ww.*t),2);
        utr(:,j,i) = -L_init(:,:,j)*xtr(:,j,i)+u_rand;
        xtr(:,j+1,i) = A(t)*xtr(:,j,i)+B(t)*utr(:,j,i);
    end
end

xtr_tilt = zeros(n*(n+1)/2,N,l);
utr_tilt = zeros(m*(m+1)/2,N,l);
xutr = zeros(m*n,N,l);
xxtr = zeros(n^2,N,l);
for i=1:l
    for j=1:N
        xtr_tilt(:,j,i) = kronv(xtr(:,j,i));
        utr_tilt(:,j,i) = kronv(utr(:,j,i));
        xutr(:,j,i) = kron(xtr(:,j,i),utr(:,j,i));
        xxtr(:,j,i) = kron(xtr(:,j,i),xtr(:,j,i));
    end
end



% check the rank condition
RK = (n*(n+1)/2+m*n+m*(m+1)/2);
for j=1:N-1
    rkmat = [];
    for i=1:l
        rkmat = [rkmat;xtr_tilt(:,j,i)', xutr(:,j,i)',utr_tilt(:,j,i)'];
    end
    rk = rank(rkmat);
    if rk~=RK
        err = ['At time step ',num2str(j),' ',num2str(i),'th repeat',...
            ' rk=',num2str(rk),'<',num2str(RK),...
            ',try to increas l or change exploration noise'];
        disp(err);
        return;
    end
end


% Off policy learning
iter_max = 1000;
L = zeros(m,n,N,iter_max);
L(:,:,:,1) = L_init;
H1bar = ones(n,n,N-1,iter_max);
H2bar = ones(m,n,N-1,iter_max);
H3bar = ones(m,m,N-1,iter_max);
H1 = ones(n,n,N-1,iter_max);
H2 = ones(m,n,N-1,iter_max);
H3 = ones(m,m,N-1,iter_max);

% Compute the last Hbar
theta = [squeeze(xtr_tilt(:,N-1,:))',2*squeeze(xutr(:,N-1,:))',...
    squeeze(utr_tilt(:,N-1,:))'];
phi = squeeze(xxtr(:,N,:))'*reshape(F,[n^2,1]);
params = theta\phi;
tmpH1 = vec2sm(params(1:n*(n+1)/2),n);
tmpH2 = reshape(params(n*(n+1)/2+1:n*(n+1)/2+m*n),[m,n]);
tmpH3 = vec2sm(params(n*(n+1)/2+m*n+1:end),m);
for i=1:iter_max
    H1bar(:,:,N-1,i) = tmpH1;
    H2bar(:,:,N-1,i) = tmpH2;
    H3bar(:,:,N-1,i) = tmpH3;
    H1(:,:,N-1,i) = tmpH1;
    H2(:,:,N-1,i) = tmpH2;
    H3(:,:,N-1,i) = tmpH3;
end

% inital error
e = 10;

k = 1;
while e>1e-2
    
    if k+1>iter_max
        disp('maximum number of iterations achieved');
        break;
    end
    e_t = [];
    L(:,:,N-1,k+1) = (R(N-1)+H3(:,:,N-1,k))\H2(:,:,N-1,k);
    
    % construct the ls equation theta*params = phi
    for I=1:N-2
        i = N-I;
        theta = [];
        phi = [];
        for j=1:l
            Lxu = kronv(L(:,:,i-1,k)*xtr(:,i-1,j)+utr(:,i-1,j));
            theta = [theta;
                xtr_tilt(:,i-1,j)', ...
                2*xxtr(:,i-1,j)'*kron(eye(n),L(:,:,i-1,k)')+...
                2*xutr(:,i-1,j)',...
                Lxu'];
            tmp = Q(i)+H1(:,:,i,k)-H2(:,:,i,k)'*((R(i)+H3(:,:,i,k))\H2(:,:,i,k));
            phi = [phi;
                xxtr(:,i,j)'*reshape(tmp,[n^2,1])];
        end
        
        % rank check
        rk = rank(theta);
        if rk~=RK
            disp('rank deficient');
            return;
        end
        
        % ls estimation
        params = theta\phi;
        
        % extract matrices
        H1bar(:,:,i-1,k+1) = vec2sm(params(1:n*(n+1)/2),n);
        H2bar(:,:,i-1,k+1) = reshape(params(n*(n+1)/2+1:n*(n+1)/2+m*n),[m,n]);
        H3bar(:,:,i-1,k+1) = vec2sm(params(n*(n+1)/2+m*n+1:end),m);
        H3(:,:,i-1,k+1) = H3bar(:,:,i-1,k+1);
        H2(:,:,i-1,k+1) = H2bar(:,:,i-1,k+1)+H3(:,:,i-1,k+1)*L(:,:,i-1,k);
        H1(:,:,i-1,k+1) = H1bar(:,:,i-1,k+1)+H2(:,:,i-1,k+1)'*L(:,:,i-1,k)...
            +L(:,:,i-1,k)'*H2(:,:,i-1,k+1)...
            -L(:,:,i-1,k)'*H3(:,:,i-1,k+1)*L(:,:,i-1,k);
        L(:,:,i-1,k+1) = (R(i-1)+H3(:,:,i-1,k))\H2(:,:,i-1,k);
        
        % calculate error
        e_t = [e_t norm(H1(:,:,i-1,k+1)-H1(:,:,i-1,k))];
        
    end
    % display
    msg = ['Iteration',num2str(k)];
    disp(msg);
    
    % next iteration
    k = k+1;
    
    % update e
    e = max(e_t);
    
end

actual_iter = k-1;
% k=k-1;
% k0 = 1;
% kf = k-3;
% 
% figure();
% leg = {};
% plot((1-1:N-1)*T,squeeze(P(1,1,:)),'r*');hold on;
% leg{end+1} = 'Optimal Value';
% for i=k0:kf
%     plot((1-1:N-1)*T,squeeze(V(1,1,:,i)),'--');
%     hold on;
%     leg{end+1} = ['Iteration ' num2str(i)];
% end
% legend(leg);
% title('V_{t,i}^{1,1}');
% xlabel('Time (s)');


% save result to file for analysis
k = k-1;
save('Online.mat','H1','H2','H3','L','k','F','N');

errK = zeros(N-1,actual_iter);
errH1 = zeros(N-1,actual_iter);
errH2 = zeros(N-1,actual_iter);
errH3 = zeros(N-1,actual_iter);

for k=1:actual_iter
    for i=1:N-1
        errK(i,k) = norm(L(:,:,i,k)-K(:,:,i));
        errH1(i,k) = norm(H1(:,:,i,k)-H1tr(:,:,i));
        errH2(i,k) = norm(H2(:,:,i,k)-H2tr(:,:,i));
        errH3(i,k) = norm(H3(:,:,i,k)-H3tr(:,:,i));
    end
end

% figure();
% mesh(errK);
XX = 0:10:N;
XX(1) = 1;
XX(end) = N-1;
[xx,yy] = meshgrid(0:10:N,0:3:actual_iter-1);
figure();
surf(xx,yy,log(errH1(XX,1:3:actual_iter)'));
colormap(cool);
zlabel('$\ln[\Vert H_{k,1}^{(i)}-A_k^TP_k^*A_k \Vert]$','Interpreter','latex');
xlabel('Time step');
ylabel('Number of iteration');
figure();
surf(xx,yy,log(errH2(XX,1:3:actual_iter)'));
colormap(cool);
zlabel('$\ln[\Vert H_{k,2}^{(i)}-B_k^TP_k^*A_k \Vert]$','Interpreter','latex');
xlabel('Time step');
ylabel('Number of iteration');
figure();
surf(xx,yy,log(errH3(XX,1:3:actual_iter)'));
colormap(cool);
zlabel('$\ln[\Vert H_{k,3}^{(i)}-B_k^TP_k^*B_k \Vert]$','Interpreter','latex');
xlabel('Time step');
ylabel('Number of iteration');

% Implement the opt control to the system
L_opt = L(:,:,:,actual_iter);
X0 = [0.0175;0.0175;0.0175;0;0;0;];
X = zeros(n,N);
U = zeros(m,N);
X(:,1) = X0;
X_init = zeros(n,N);
X_init(:,1) = X0;
for i=1:N-1
    U(:,i) = -L_opt(:,:,i)*X(:,i);
    X(:,i+1) = A(i)*X(:,i)+B(i)*U(:,i);
    X_init(:,i+1) = A(i)*X_init(:,i);
end

figure();
for i=1:3
    plot((1-1:N-1)*T,X(i,:));hold on
end
for i=1:3
    plot((1-1:N-1)*T,X_init(i,:),'--');hold on
end
xlabel('Time (s)');
%ylabel(['x_' num2str(i)]);
legend({'$x_1$','$x_2$','$x_3$','$\hat{x}_1$','$\hat{x}_2$','$\hat{x}_3$'},...
    'Interpreter','latex');

figure();
for i=4:6
    plot((1-1:N-1)*T,X(i,:));hold on
end
for i=4:6
    plot((1-1:N-1)*T,X_init(i,:),'--');hold on
end
xlabel('Time (s)');
%ylabel(['x_' num2str(i)]);
legend({'$x_4$','$x_5$','$x_6$','$\hat{x}_4$','$\hat{x}_5$','$\hat{x}_6$'},...
    'Interpreter','latex');

figure();
for i=1:m
    plot((1-1:N-1-1)*T,U(i,1:N-1));hold on;
end
xlabel('Time (s)');
%    ylabel(['u_' num2str(i)]);
legend('u_1','u_2','u_3');

end

% unique kron vector
function X = kronv(x)
len = length(x);
X = [];
for i=1:len
    for j=i:len
        X(end+1) = x(i)*x(j);
    end
end
X = X';
end

function X = vec2sm(x,n)
X = zeros(n);
num = flip(1:n);
for i=1:n
    index = 0;
    for k=1:i-1
        index = index+num(k);
    end
    for j=0:n-i
        if j~=0
            X(i,i+j)=x(index+j+1)/2;
            X(j+i,i)=X(i,j+i);
        else
            X(i,j+i)=x(index+j+1);
        end
    end
end
end