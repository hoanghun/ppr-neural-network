__kernel void FeedForward(__global float* input, __global float* weights, __global float* layer_outputs, int previous_layer_size, int layer_size) {	
	int previous_layer_id = get_global_id(0);
	int layer_id = get_global_id(1);
	int neural_network_id = get_global_id(2);

	int neural_network_offset = neural_network_id * (previous_layer_size * layer_size);
	int offset = previous_layer_id * layer_size + layer_id;

	float weight = weights[neural_network_offset + offset];
	float input_val = input[neural_network_id * previous_layer_size + previous_layer_id];

	layer_outputs[neural_network_offset + offset] = weight * input_val;
}


__kernel void FeedForwardSum(__global float* input, __global float* layer_outputs, __global float* layer_biases, int layer_size, int previous_layer_size) {	
	int output_layer_id = get_global_id(0);
	int neural_network_id = get_global_id(1);

	int neural_network_offset = neural_network_id * (layer_size * previous_layer_size);
	float sum = 0;

	for (size_t i = output_layer_id; i < previous_layer_size * layer_size; i += layer_size) {
		sum += input[neural_network_offset + i];
	}
	
	layer_outputs[neural_network_id * layer_size + output_layer_id] = tanh(sum  + layer_biases[output_layer_id]); 
}


__kernel void AddTwoArrays(__global float *a, __global float* b, __global float* c) {
	size_t global_id = get_global_id(0);
	c[global_id] = a[global_id] + b[global_id];
}

float band_index_to_level(int index) {
	float Low_Threshold = 3.0;
	float High_Threshold = 13.0;
	size_t Internal_Bound_Count = 30;

	float Band_Size = (High_Threshold - Low_Threshold) / (float)Internal_Bound_Count;
	float Inv_Band_Size = 1.0 / Band_Size;
	float Half_Band_Size = 0.5 / Inv_Band_Size;
	size_t Band_Count = Internal_Bound_Count + 2;
	size_t Output_Layer_Count = 32;
	size_t Inner_Layer_Count = 8;
	if (index == 0) return Low_Threshold - Half_Band_Size;
	if (index >= Band_Count - 1) return High_Threshold + Half_Band_Size;

	return Low_Threshold + (float)(index - 1) * Band_Size + Half_Band_Size;
}

__kernel void CalculateErrors(__global float *outputs, __global float* errors, int layer_size, float measured_value) {
	int neural_network_id = get_global_id(0);
	int offset = neural_network_id * layer_size;

	int highest = 0;
	for (int i = 1; i < layer_size; i++) {
		if (outputs[highest + offset] < outputs[i + offset]) {
			highest = i;
		}
	}
	float x = band_index_to_level(highest);
	float error = fabs(x - measured_value) / measured_value;
	errors[neural_network_id] = error;
}
