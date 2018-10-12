//not tested. not useful since many members are declared constant
forward_prop::forward_prop() : 
    num_inputs(), 
    num_outputs(), 
    m(), 
    batch_size(), 
    layer_sizes(), 
    x_path(), 
    y_path(), 
    weight_shapes(), 
    x_tr_v(), 
    x_tr_m(0,0,0), 
    y_tr_v(), 
    y_tr_m(0,0,0), 
    LL_type(), 
    LL_var(), 
    LL_norm(), 
    num_complete_batches(), 
    num_batches(), 
    b_c(), 
    rand_m_ind() {}
 

//not tested. not sure it will work for one because of read only members
forward_prop::forward_prop(const forward_prop &fp2) :
    num_inputs(fp2.num_inputs), 
    num_outputs(fp2.num_outputs), 
    m(fp2.m), 
    batch_size(fp2.batch_size), 
    layer_sizes(fp2.layer_sizes), 
    x_path(fp2.x_path), 
    y_path(fp2.y_path), 
    weight_shapes(fp2.weight_shapes), 
    x_tr_v(fp2.x_tr_v), 
    x_tr_m(fp2.x_tr_m), 
    y_tr_v(fp2.y_tr_v), 
    y_tr_m(fp2.y_tr_m), 
    LL_type(fp2.LL_type), 
    LL_var(fp2.LL_var), 
    LL_norm(fp2.LL_norm), 
    num_complete_batches(fp2.num_complete_batches), 
    num_batches(fp2.num_batches), 
    b_c(fp2.b_c), 
    rand_m_ind(fp2.rand_m_ind) {}
