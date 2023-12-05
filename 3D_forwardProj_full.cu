__global__ void forwardProj3D(float *projection,float *object,int N,float *GX,float *GY, float *GZ)
{


float t0;

int m,n,k,s,x,y,z,t1,t2;



 m=threadIdx.x;
 n=blockIdx.x;
 k=blockIdx.y;
 s=blockIdx.z;

 x=m-N/2+1;
 y=n-N/2+1;
 z=k-N/2+1;



 t0=x*GX[s]+y*GY[s]+z*GZ[s];
 t1=floor(t0);
 t2=ceil(t0);

 t0=t0+(N/2);
 t1=t1+(N/2);
 t2=t2+(N/2); 

 if(t1>=0&&t2<=(N-1))
    {
      if(t1!=t2)
       { 
        atomicAdd(&projection[t1+N*s],object[k*N*N+n*N+m]*(t2-t0));
        atomicAdd(&projection[t2+N*s],object[k*N*N+n*N+m]*(t0-t1));

        }
                   
     else
       
       atomicAdd(&projection[t1+N*s],object[k*N*N+n*N+m]); 
   
}
}
