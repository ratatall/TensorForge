input activations: tensor<2,4>;
let positive = relu(activations);
return sum(positive, 1);
