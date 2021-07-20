__global__ void sinwave_kernal(float4 *pos, unsigned int width, unsigned int hight, float time) {
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
	float u = x / (float)width;
	float v = y / (float)hight;
	u = u * 2.0f - 1.0f;
	v = v * 2.0f - 1.0f;
	float freq = 4.0f;
	float w = sinf(u * freq + time) * cosf(v * freq + time) * 0.5f;
	pos[y * width + x] = make_float4(u,w,v,1.0f);
}

void LaunchCUDAKernal(float4 *pPos, unsigned int mesh_width, unsigned int mesh_hight, float time) {
	dim3 block(8,8,1);
	//GPU threads block dimenation x: 8 , Y : 8 
	dim3 grid(mesh_width / block.x, mesh_hight / block.y,1);
	sinwave_kernal<<<grid,block>>>(pPos, mesh_width, mesh_hight, time);
}