input A: tensor<1024>;
input B: tensor<1024>;
let scaled = A * 2.0;
let biased = scaled + B;
let activated = relu(biased);
return activated;
