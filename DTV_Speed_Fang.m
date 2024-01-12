% The Program of DTV Algorithm For EPRI
% Author: Fang CY
% Updated on 2024/01/12

set(0,'DefaultFigureColormap',jet);

clear;
clc;

len_proj=3*sqrt(2); % fov

n_proj=208; % number of projection

bandwidth = 0.07; % filtercutoff

n_iter=150; % number of iteration

t_factor=0.5; % Try 0.2:0.1:0.6

N=64;         % the object is of size [N,N,N].

shots=100; % repetition time

n_polar=18;

[GX,GY,GZ]=GetAngleVector(n_polar,n_polar);

load('18x18_MSPS_idx.mat');% get a variable idx whose order is MSPS.

GX=GX(idx(1:n_proj));   % get the angle index for the projections
GY=GY(idx(1:n_proj));
GZ=GZ(idx(1:n_proj));




% Set up the introduced algorithm parameters
L1=compute_tau_3D_fun_POCS_MSPS(N,GX,GY,GZ);% the Largest Singular Value of the system matrix A
L2=compute_L_for_D_3D_fun(N); % the LSV of the matrix D, the discrete gradient transfrom of the object
v_A=L1/L2;  % This will be the coefficient of TV term

lamda=1;
v_factor=1;

v=v_A*v_factor;


% load and prepare the real projections

load('new_all_pro_10mL_1mM_100shots.mat');

projection=denoise_proj(my_projection,bandwidth); % denoise the projecitons

projection=projection(:,idx(1:n_proj)); % adjust the projection order to be MSPS

projection=imresize(projection,[N,n_proj]); % projection interpolation to let the number of projection-bin be N. 

projection=projection*((N/(len_proj))^2);  % normalize the projection values.



% set up the model parameter

load('object_FBP_Qiao_208_10mL_1mM_2000shots.mat');

object_truth=object_FBP;

TV_truth_x=D1f_3D(permute(flip(object_truth, 2), [3,2,1]));
TV_truth_x=sum(sum(sum(abs(TV_truth_x))));
tx=t_factor*TV_truth_x;

TV_truth_y=D2f_3D(permute(flip(object_truth, 2), [3,2,1]));
TV_truth_y=sum(sum(sum(abs(TV_truth_y))));
ty=t_factor*TV_truth_y;

TV_truth_z=D3f_3D(permute(flip(object_truth, 2), [3,2,1]));
TV_truth_z=sum(sum(sum(abs(TV_truth_z))));
tz=t_factor*TV_truth_z;


% set up the algorithm parameters
delta1=1/N/N;
delta2=1/2/v;
tau=1/(n_proj+6*v);

% initialize the variables in the iteration
p=zeros(N,n_proj);
q1=zeros(N,N,N);
q2=q1;
q3=q1;
object=q1;
object0=q1;
one_object=ones(N,N,N);

% prepare the iteration observation variable
error=0;
data_error=error;
TV_x=error;



figure;
tic;
%DTV algorithm
for ii=1:n_iter
    
    
  
   p=(p+delta1*(FP_3D_GPU(object0,GX,GY,GZ)-projection))/(1+delta1/lamda);  
 
   
   a1=q1+0.5*D1f_3D(object0);
   m=abs(a1)/delta2; 
   s=ProjectOntoL1Ball(m,v*tx);   
   q1(m~=0)=a1(m~=0).*(one_object(m~=0)-delta2*s(m~=0)./((m(m~=0)*delta2)));
   
   a2=q2+0.5*D2f_3D(object0);
   m=abs(a2)/delta2; 
   s=ProjectOntoL1Ball(m,v*ty);   
   q2(m~=0)=a2(m~=0).*(one_object(m~=0)-delta2*s(m~=0)./((m(m~=0)*delta2)));
   
   
   a3=q3+0.5*D3f_3D(object0);
   m=abs(a3)/delta2; 
   s=ProjectOntoL1Ball(m,v*tz);  
   q3(m~=0)=a3(m~=0).*(one_object(m~=0)-delta2*s(m~=0)./((m(m~=0)*delta2)));

  
 
   object1=object-tau*BP_3D_GPU_full(p,GX,GY,GZ)-tau*v*DTg_3D(q1,q2,q3);
  
   
   
   object_step=object1-object;
   object=object1;
   object0=object+object_step;
   
   
   b=FP_3D_GPU(object,GX,GY,GZ)-projection;
   data_error(ii)=norm(b(:))/sqrt(N*n_proj); % RMSE of data

   TV_x(ii)=sum(sum(sum(abs(D1f_3D(object))))); % TV at x direction.

   object_old=object;
   
   object = flip(object, 2); % flip y axis
   object = permute(object, [3,2,1]); % change x and z

   subplot(2,2,1) % show object_truth
   imagesc(flip(squeeze(object_truth(:,N/2,:)),1));
   title(num2str(shots));
   
   
   subplot(2,2,2) % show reconstructed object
   imagesc(flip(squeeze(object(:,N/2,:)),1));
   title(num2str(ii));

   subplot(2,2,3) % show profile
   plot(1:N,squeeze(object_truth(:,N/2,N/2)),'r',1:N,squeeze(object(:,N/2,N/2)),'b');
   

   subplot(2,2,4)
   semilogy(data_error);
   title('data RMSE')

   object=object_old;

   drawnow;
end

fprintf('the reconstruction used %f seconds',toc);

  object = flip(object, 2); % flip y axis
  object = permute(object, [3,2,1]); % change x and z

save ([num2str(N),'_',num2str(n_proj),'_',num2str(ii),'_','DTV','_',num2str(lamda),'_',num2str(v_factor),'_',num2str(t_factor),'_',num2str(shots),'.mat']);
saveas(gcf,[num2str(N),'_',num2str(n_proj),'_',num2str(ii),'_','DTV','_',num2str(lamda),'_',num2str(v_factor),'_',num2str(t_factor),'_',num2str(shots),'.fig']);






























