#in-house modules
import input_tools

def forward_test_linear(models, num_weights, data_dir):
    weight_type = 'linear'
    weight_f = data_dir + weight_type + '_weights.txt' 
    w = input_tools.get_weight_data(weight_f, num_weights)
    for model in models:
    	model.test_output(w)

