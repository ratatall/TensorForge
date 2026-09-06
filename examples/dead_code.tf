input A: tensor<256>;
let unused = A * 5.0;
return relu(A);
