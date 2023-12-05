__global__ void backProj(float *object,float *projection,float *GX, float *GY,float *GZ,int N,int n_proj)
{


float t0,value1,value2;
int m,n,k,x,y,z,t1,t2,i;


 m=threadIdx.x;
 n=blockIdx.x;
 k=blockIdx.y;

 x=m+1-N/2;
 y=n+1-N/2;
 z=k+1-N/2;

for(i=0;i<n_proj;i++)
{
                
 t0=x*GX[i]+y*GY[i]+z*GZ[i];
 t1=floor(t0);
 t2=ceil(t0);

 t0=t0+(N/2);
 t1=t1+(N/2);
 t2=t2+(N/2); 

 if(t1>=0&&t2<=(N-1))
     {
      value1=projection[t1+i*N];
      value2=projection[t2+i*N];
      object[k*N*N+n*N+m]+=value1+(value2-value1)*(t0-t1);
     }

}
}