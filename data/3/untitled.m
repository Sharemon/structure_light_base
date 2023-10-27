clear
clc
Ik=imread('1.bmp');
sin_sum = 0.0;
cos_sum = 0.0;
N = 12;
for k = 0: N - 1
    pk = 2 * k * pi / N;
    %Ik = m_filter2d(Ik);
    sin_sum = sin_sum + double(Ik) * sin(pk);
    cos_sum = cos_sum + double(Ik) * cos(pk);
end
% 根据计算相位、调制度
pha = -atan2(sin_sum, cos_sum);
B = sqrt(sin_sum .^ 2 + cos_sum .^ 2) * 2 / N;
% 简单的过滤掉调制度低的区域
% 对于孔洞等边缘的相位，需要根据梯度进行滤波来滤除，较为复杂，这里暂且不处理
