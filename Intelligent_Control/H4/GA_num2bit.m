function bit = GA_num2bit(real, range, bit_length)

temp = (real-range(1));
resolution = (abs(range(1))+abs(range(2)))/(2^bit_length-1);
bit = fliplr(de2bi(floor(temp/resolution), bit_length));
