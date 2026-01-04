# $S-RNN$ 算法原理
对于长度为 $T$ 的时间序列数据 $x=\left[x_1,x_2,\ldots,x_t,\ldots,x_T\right]$ ， $x_t$ 为时刻 $t$ 的输入向量。简单循环神经网络 $S-RNN$ 算法的结构单元如下：

$s_t=Uh_{t-1}+Wx_t+b$

$h_t=tanh\left(s_t\right)$

$z_t=Vh_t+c$

$\widehat{y_t}=softmax\left(z_t\right)$

其中， $h_{t-1}$ 代表 $t-1$ 时刻的隐状态， $x_t$ 为时刻 $t$ 的输入，时刻 $t$ 的净输入 $s_t$ 经过 $tanh(·)$ 激活函数转换为 $t$ 时刻的隐状态，时刻 $t$ 的净输入 $z_t$ 经过 $softmax(·)$ 转换为时刻 $t$ 的最终输出 $\widehat{y_t}$ ， $U$ 、 $W$ 、 $V$ 为神经网络的权重矩阵， $b$ 、 $c$ 为神经网络净输入的偏置向量。

$S-RNN$ 结构单元表明：在时刻 $t$ ，时刻 $t$ 的输入 $x_t$ 和上一个时刻的隐状态 $h_{t-1}$ 共同决定了时刻 $t$ 的隐状态 $h_t$ 以及输出 $\widehat{y_t}$ 。对于长度为 $T$ 的时间序列数据 $x$ ，循环使用该结构 $T$ 次可以得到 $S-RNN$ 算法在各个时刻的输出 $\hat{y}=\left[\widehat{y_1},\widehat{y_2},\ldots,\widehat{y_t},\ldots,\widehat{y_T}\right]$ 。上述关系用数学公式表示为：

$\widehat{y_t}=P(y_t|x_t,h_{t-1})=P(y_t|x_t,x_{t-1},h_{t-2})=P(y_t|x_t,x_{t-1},...,x_1)$

$S-RNN$ 算法的反向传播过程如下：

矩阵 $V$ 在每个时刻 $t$ 都会影响该时刻的损失 $L_t$ 进而影响总损失 $L$ ，对应前向链式传播路径： $V\rightarrow z_1,z_2,\ldots,z_t,\ldots,z_T\rightarrow L_1,L_2,\ldots,L_t,\ldots,L_T\rightarrow L$ ，可推：

$dL=tr\left(\sum_{t=1}^{T}{\left(\frac{\partial L}{\partial z_t}\right)^Tdz_t}\right)=tr\left(\sum_{t=1}^{T}{\left(\frac{\partial L}{\partial z_t}\right)^TdVh_t}\right)=tr\left(\sum_{t=1}^{T}{\left(\frac{\partial L}{\partial z_t}{h_t}^T\right)^TdV}\right)$

$\frac{\partial L}{\partial V}=\sum_{t=1}^{T}{\frac{\partial L}{\partial z_t}{h_t}^T}=\sum_{t=1}^{T}{\left(\widehat{y_t}-y_t\right){h_t}^T}$

$t$ 时刻的向量 $s_t$ 通过影响损失 $L_t$ 和 $t+1$ 时刻的向量 $s_{t+1}$ 进而影响总损失 $L$ ，对应前向链式传播路径： $s_t\rightarrow z_t\rightarrow L_t\rightarrow L$ ， $s_t\rightarrow s_{t+1}\rightarrow L$ ，可推：

$dL=tr\left(\left(\frac{\partial L}{\partial z_t}\right)^Tdz_t\right)+tr\left(\left(\frac{\partial L}{\partial s_{t+1}}\right)^Tds_{t+1}\right)=tr\left(\left(V^T\frac{\partial L}{\partial z_t}\right)^T\left(1-h_t\odot h_t\right)\odot d s_t\right)+tr\left(\left(U^T\frac{\partial L}{\partial s_{t+1}}\right)^T\left(1-h_t\odot h_t\right)\odot d s_t\right)=tr\left(\left(V^T\frac{\partial L}{\partial z_t}\odot\left(1-h_t\odot h_t\right)\right)^Tds_t+\left(U^T\frac{\partial L}{\partial s_{t+1}}\odot\left(1-h_t\odot h_t\right)\right)^Tds_t\right)=tr\left(\left(V^T\frac{\partial L}{\partial z_t}\odot\left(1-h_t\odot h_t\right)+U^T\frac{\partial L}{\partial s_{t+1}}\odot\left(1-h_t\odot h_t\right)\right)^Tds_t\right)$

$\frac{\partial L}{\partial s_t}=V^T\frac{\partial L}{\partial z_t}\odot\left(1-h_t\odot h_t\right)+U^T\frac{\partial L}{\partial s_{t+1}}\odot\left(1-h_t\odot h_t\right)$

$T$ 时刻的向量 $s_T$ 通过影响损失 $L_T$ 进而影响总损失 $L$ ，对应前向链式传播路径： $s_T\rightarrow L_T\rightarrow L$ ，可推：

$L=tr\left(dL_T\right)=tr\left(\left(\widehat{y_T}-y_T\right)^Tdz_T\right)=tr\left(\left(\widehat{y_T}-y_T\right)^TVdh_T\right)=tr\left(\left(V^T\left(\widehat{y_T}-y_T\right)\right)^T\left(1-h_T\odot h_T\right)\odot d s_T\right)=tr\left(\left(V^T\left(\widehat{y_T}-y_T\right)\odot\left(1-h_T\odot h_T\right)\right)^Tds_T\right)$

$\frac{\partial L}{\partial s_T}=V^T\left(\widehat{y_T}-y_T\right)\odot\left(1-h_T\odot h_T\right)$

矩阵 $U$ 在每个时刻 $t$ 都会影响该时刻的隐状态净输入 $s_t$ 进而影响总损失 $L$ ，对应前向链式传播路径： $U\rightarrow s_1,s_2,\ldots,s_t,\ldots,s_T\rightarrow L$ ，可推：

$dL=tr\left(\sum_{t=1}^{T}{\left(\frac{\partial L}{\partial s_t}\right)^Tds_t}\right)=tr\left(\sum_{t=1}^{T}{\left(\frac{\partial L}{\partial s_t}\right)^TdUh_{t-1}}\right)=tr\left(\sum_{t=1}^{T}{\left(\frac{\partial L}{\partial s_t}{h_{t-1}}^T\right)^TdU}\right)=tr\left(\left(\sum_{t=1}^{T}{\frac{\partial L}{\partial s_t}{h_{t-1}}^T}\right)^TdU\right)$

$\frac{\partial L}{\partial U}=\sum_{t=1}^{T}{\frac{\partial L}{\partial s_t}{h_{t-1}}^T}$

矩阵 $W$ 在每个时刻 $t$ 都会影响该时刻的隐状态净输入 $s_t$ 进而影响总损失 $L$ ，对应前向链式传播路径： $W\rightarrow s_1,s_2,\ldots,s_t,\ldots,s_T\rightarrow L$ ，可推：

$\frac{\partial L}{\partial W}=\sum_{t=1}^{T}{\frac{\partial L}{\partial s_t}{x_t}^T}$

向量 $b$ 在每个时刻 $t$ 都会影响该时刻的隐状态净输入 $s_t$ 进而影响总损失 $L$ ，对应前向链式传播路径： $b\rightarrow s_1,s_2,\ldots,s_t,\ldots,s_T\rightarrow L$ ，可推：

$\frac{\partial L}{\partial b}=\sum_{t=1}^{T}\frac{\partial L}{\partial s_t}$

向量 $c$ 在每个时刻 $t$ 都会影响该时刻的损失 $L_t$ 进而影响总损失 $L$ ，对应前向链式传播路径： $c\rightarrow z_1,z_2,\ldots,z_t,\ldots,z_T\rightarrow L_1,L_2,\ldots,L_t,...,L_T\rightarrow L$ ，可推：

$dL=tr\left(\sum_{t=1}^{T}{\left(\frac{\partial L}{\partial z_t}\right)^Tdz_t}\right)=tr\left(\sum_{t=1}^{T}{\left(\frac{\partial L}{\partial z_t}\right)^Tdc}\right)=tr\left(\left(\sum_{t=1}^{T}\frac{\partial L}{\partial z_t}\right)^Tdc\right)$

$\frac{\partial L}{\partial c}=\sum_{t=1}^{T}\frac{\partial L}{\partial z_t}=\left(\widehat{y_t}-y_t\right)$
