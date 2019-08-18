gray_code=[1 0 1 0 1 0 1 0];
binary_code=zeros(size(gray_code));
binary_code(1) = gray_code(1);
decimal =  binary_code(1) * 2^(length(binary_code)-1);
for i=2:length(gray_code)
    binary_code(i) = xor(gray_code(i),binary_code(i-1)) ;
    decimal = decimal + binary_code(i) * 2^(length(binary_code)-i);
end