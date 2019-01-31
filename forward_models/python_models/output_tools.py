#in-house modules
import output

def write_paramnames(num_inputs, layer_sizes, num_outputs, m_trainable_arr = [], b_trainable_arr = [], file_root = 'test', write_np = True, write_tf = True, write_k = True, write_cpp = True):
	if len(m_trainable_arr) == 0:
			m_trainable_arr = [True] * (len(layer_sizes) + 1)
	if len(b_trainable_arr) == 0:
		b_trainable_arr = [True] * (len(layer_sizes) + 1)
	if write_np:
		base_dir = './np_chains/'
		output.make_param_names3(num_inputs, layer_sizes, num_outputs, m_trainable_arr, b_trainable_arr, base_dir, file_root)
	if write_tf:
		base_dir = './tf_chains/'
		output.make_param_names3(num_inputs, layer_sizes, num_outputs, m_trainable_arr, b_trainable_arr, base_dir, file_root)
	if write_k:
		base_dir = './keras_chains/'
		output.make_param_names3(num_inputs, layer_sizes, num_outputs, m_trainable_arr, b_trainable_arr, base_dir, file_root)
	if write_cpp:
		base_dir = '../cpp_models/cpp_chains/'
		output.make_param_names3(num_inputs, layer_sizes, num_outputs, m_trainable_arr, b_trainable_arr, base_dir, file_root)

def write_pred_paramnames(num_outputs, file_root = 'test', write_np = True, write_tf = True, write_k = True, write_cpp = True):
	if write_np:
		base_dir = './np_chains/'
		output.make_pred_param_names(num_outputs, base_dir, file_root)
	if write_tf:
		base_dir = './tf_chains/'
		output.make_pred_param_names(num_outputs, base_dir, file_root)
	if write_k:
		base_dir = './keras_chains/'
		output.make_pred_param_names(num_outputs, base_dir, file_root)
	if write_cpp:
		base_dir = '../cpp_models/cpp_chains/'
		output.make_pred_param_names(num_outputs, base_dir, file_root)
