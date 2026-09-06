input X: tensor<17>;
let shared = X * 0.5;
return relu(shared * shared + shared);
