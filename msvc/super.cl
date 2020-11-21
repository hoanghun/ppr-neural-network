__kernel void FeedForward(__global double* input, __global double* weights, __global double* layer_outputs, int previous_layer_size, int layer_size) {	
	int previous_layer_id = get_global_id(0);
	int layer_id = get_global_id(1);
	int neural_network_id = get_global_id(2);

	int neural_network_offset = neural_network_id * (previous_layer_size * layer_size);
	int offset = previous_layer_id * layer_size + layer_id;

	double weight = weights[neural_network_offset + offset];
	//printf("WEIGHt %f\n", weight);
	double input_val = input[neural_network_id * previous_layer_size + previous_layer_id];

	//printf("input %f\n", input_val);
	//printf("layer_outputs %f\n", layer_outputs[neural_network_offset + offset]);
	layer_outputs[neural_network_offset + offset] = weight * input_val;
}


__kernel void FeedForwardSum(__global double* input, __global double* layer_outputs, __global double* layer_biases, int layer_size, int previous_layer_size) {	
	int output_layer_id = get_global_id(0);
	int neural_network_id = get_global_id(1);

	int neural_network_offset = neural_network_id * (layer_size * previous_layer_size);
	double sum = 0;

	for (size_t i = output_layer_id; i < previous_layer_size * layer_size; i += layer_size) {
		sum += input[neural_network_offset + i];
	}
	
	layer_outputs[neural_network_id * layer_size + output_layer_id] = tanh(sum  + layer_biases[output_layer_id]); 
}


__kernel void AddTwoArrays(__global double *a, __global double* b, __global double* c) {
	size_t global_id = get_global_id(0);
	c[global_id] = a[global_id] + b[global_id];
}

double band_index_to_level(int index) {
	double Low_Threshold = 3.0;
	double High_Threshold = 13.0;
	size_t Internal_Bound_Count = 30;

	double Band_Size = (High_Threshold - Low_Threshold) / (double)Internal_Bound_Count;
	double Inv_Band_Size = 1.0 / Band_Size;
	double Half_Band_Size = 0.5 / Inv_Band_Size;
	size_t Band_Count = Internal_Bound_Count + 2;
	size_t Output_Layer_Count = 32;
	size_t Inner_Layer_Count = 8;
	if (index == 0) return Low_Threshold - Half_Band_Size;
	if (index >= Band_Count - 1) return High_Threshold + Half_Band_Size;

	return Low_Threshold + (double)(index - 1) * Band_Size + Half_Band_Size;
}

__kernel void CalculateErrors(__global double *outputs, __global double* errors, int layer_size, double measured_value) {
	int neural_network_id = get_global_id(0);
	int offset = neural_network_id * layer_size;

	int highest = 0;
	for (int i = 1; i < layer_size; i++) {
		if (outputs[highest + offset] < outputs[i + offset]) {
			highest = i;
		}
	}
	double x = band_index_to_level(highest);
	double error = fabs(x - measured_value) / measured_value;
	//printf("Neural network id: %d, Got %f, wanted %f, error is %f\n", neural_network_id, x, measured_value, error);
	errors[neural_network_id] = error;
}
