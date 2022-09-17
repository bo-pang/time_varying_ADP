function FHADP_dis_recur_new()
% system matrix

% time step
T = 0.05;

A = @(k) eye(2)+[0 k*T;-2*cos(6*k*T) (k*T)^(0.5)*sin(10*k*T)]*T;
B = @(k) [1; (2*k*T+2)/(2*k*T+3)]*T;

% total length
N = 120+1;

% get dimensions
[n,m] = size(B(1));

% reward weighting matrices
% Q = @(t) 10*eye(n);
% R = @(t) eye(m);
% F = eye(n)*10;
Q = @(t) [0.04*t+2 0;0 0.04*t+2];
R = @(t) 5-0.02*t;
F = [1 0; 0 1]*50;
% Q = @(t) [exp(-0.4*t)+1 0;0 exp(-0.8*t)+1];
% R = @(t) 1-exp(-0.2*t);
% F = [10 0; 0 10];

% get true cost matrix
P = zeros(n,n,N);
P(:,:,N) = F;
for i = 1:N-1
    t = N-i;
    P(:,:,t)= A(t)'*P(:,:,t+1)*A(t)+Q(t)-A(t)'*P(:,:,t+1)*B(t)*...
        inv(R(t)+B(t)'*P(:,:,t+1)*B(t))*B(t)'*...
        P(:,:,t+1)*A(t);
end

% data collection rounds
l = m*(m+1)/2+m*n+n*(n+1)/2;

% raw data
% Init = load('True.mat');
% L_init = Init.L(:,:,:,6);
% L_init = ones(m,n,N);
L_init = zeros(m,n,N);
xtr = zeros(n,N,l);
utr = zeros(m,N,l);

% training data collection
for i=1:l
    % inital state
    xtr(:,1,i) = -1+ (1+1)*rand(n,1);
    % exploration noise params
    ww = (-500 + (500-(-500)).*rand(500,1));
    mm = 2;%(5 + (3-1).*rand(1,1));
    for j=1:N-1
        t = j;
        % exploration noise
        u_rand = mm*sum(sin(ww.*t));
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
V = zeros(n,n,N,iter_max);
BVA = zeros(m,n,N,iter_max);
BVB = zeros(m,m,N,iter_max);

% inital error
e = 10;

k = 1;
while e>1e-2
    
    if k+1>iter_max
        disp('maximum number of iterations achieved');
        break;
    end

    V(:,:,N,k) = F;
    e_t = [];
    
    % construct the ls equation theta*params = phi
    for I=1:N-1
        i = N-I;
        theta = [];
        phi = [];
        for j=1:l
            Lx = kronv(L(:,:,i,k)*xtr(:,i,j));
            LL = kron(L(:,:,i,k),L(:,:,i,k));
            theta = [theta;
                xtr_tilt(:,i,j)', ...
                2*xxtr(:,i,j)'*kron(eye(n),L(:,:,i,k)')+...
                2*xutr(:,i,j)',...
                utr_tilt(:,i,j)'-Lx'];
            phi = [phi;
                xxtr(:,i,j)'*(reshape(Q(i),[n^2,1])+...
                LL'*reshape(R(i),[m^2,1]))+...
                xxtr(:,i+1,j)'*reshape(V(:,:,i+1,k),[n^2,1])];
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
        V(:,:,i,k) = vec2sm(params(1:n*(n+1)/2),n);
        BVA(:,:,i+1,k) = reshape(params(n*(n+1)/2+1:n*(n+1)/2+m*n),[m,n]);
        BVB(:,:,i+1,k) = vec2sm(params(n*(n+1)/2+m*n+1:end),m);
        L(:,:,i,k+1) = (R(i)+BVB(:,:,i+1,k))\BVA(:,:,i+1,k);
        
        % calculate error
        if k>1
            e_t = [e_t norm(V(:,:,i,k)-V(:,:,i,k-1))];
        else
            e_t = [e_t norm(V(:,:,i,k)-zeros(n,n))];
        end
        
    end
    
    % display
    msg = ['Iteration',num2str(k)];
    disp(msg);
    
    % next iteration
    k = k+1;    
    
    % update e
    e = max(e_t);
    
end

k=k-1;
k0 = 1;
kf = k;

errV = zeros(N,k);

for i = 1:N
    for j = 1:k
        errV(i,j) = norm(V(:,:,i,j)-P(:,:,i));
    end
end

figure();
leg = {};
for i=1:4
    plot((1-1:N-1),squeeze(errV(:,i)),'*');
    hold on;
    leg{end+1} = ['Iteration ' num2str(i)];
end
legend(leg);
ylabel({'$\Vert V_{k,i}-P_k\Vert$'},'Interpreter','latex');
xlabel('Time Steps');

figure();
leg = {};
for i=4:8
    plot((1-1:N-1),squeeze(errV(:,i)),'*');
    hold on;
    leg{end+1} = ['Iteration ' num2str(i)];
end
legend(leg);
ylabel({'$\Vert V_{k,i}-P_k\Vert$'},'Interpreter','latex');
xlabel('Time Steps');

% save result to file for analysis
k = k-1;
save('Online.mat','V','L','k','F','N');

% Implement the opt control to the system
L_opt = L(:,:,:,k);
X0 = [-2;1];
X = zeros(n,N);
Xhat = zeros(n,N);
U = zeros(m,N);
Uhat = zeros(m,N);
X(:,1) = X0;
Xhat(:,1) = X0;
for i=1:N-1
    U(:,i) = -L_opt(:,:,i)*X(:,i);
    X(:,i+1) = A(i)*X(:,i)+B(i)*U(:,i);
    Uhat(:,i) = -L_init(:,:,i)*Xhat(:,i);
    Xhat(:,i+1) = A(i)*Xhat(:,i)+B(i)*Uhat(:,i);
end

figure();
for i=1:n
    stairs((1-1:N-1),X(i,:));hold on;
end
for i=1:n
    stairs((1-1:N-1),Xhat(i,:),'-.');hold on;
end
xlabel('Time Steps');
%ylabel(['x_' num2str(i)]);
legend({'$x_1$: proposed controller','$x_2$: proposed controller',...
    '$\hat{x}_1$: initial controller','$\hat{x}_2$: initial controller'}...
    ,'Interpreter','latex');

figure();
for i=1:m
    stairs((1-1:N-1-1),U(m,1:N-1));hold on;
end
xlabel('Time steps');
legend('u');

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