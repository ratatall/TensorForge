input scale: f32;
input X: tensor<127>;
return relu(X * scale + -0.5);
