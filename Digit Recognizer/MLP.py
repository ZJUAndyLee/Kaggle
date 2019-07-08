MLPClassifier(activation='relu', alpha=1e-05, batch_size='auto',
              beta_1=0.9, beta_2=0.999, early_stopping=False,
              epsilon=1e-08, hidden_layer_sizes=(5, 2),
              learning_rate='constant', learning_rate_init=0.001,
              max_iter=200, momentum=0.9, n_iter_no_change=10,
              nesterovs_momentum=True, power_t=0.5, random_state=1,
              shuffle=True, solver='lbfgs', tol=0.0001,
              validation_fraction=0.1, verbose=False, warm_start=False)

'''
参数说明：
1. activation: 激活函数,{'identity','logistic','tanh','relu'},默认 relu
-identity: f(x)=x
-logistic: sigmoid
-tanh：f(x) = tanh(x)
-relu：f(x) = max(0, x)

2. hidden_layer_sizes :例如hidden_layer_sizes=(50, 50)，表示有两层隐藏层，第一层隐藏层有50个神经元，第二层也有50个神经元。

3. solver： {‘lbfgs’, ‘sgd’, ‘adam’}, 默认adam，用来优化权重 
- lbfgs：quasi-Newton方法的优化器 - sgd：随机梯度下降 
- adam： Kingma, Diederik, and Jimmy Ba提出的机遇随机梯度的优化器 
注意：默认solver ‘adam’在相对较大的数据集上效果比较好（几千个样本或者更多），对小数据集来说，lbfgs收敛更快效果也更好。

4. alpha :float,可选的，默认0.0001,正则化项参数 

5. batch_size : int , 可选的，默认’auto’,随机优化的minibatches的大小
batch_size=min(200,n_samples)，如果solver是’lbfgs’，分类器将不使用minibatch 

6. learning_rate :学习率,用于权重更新,只有当solver为’sgd’时使用，{‘constant’，’invscaling’, ‘adaptive’},默认constant 
- ‘constant’: 有’learning_rate_init’给定的恒定学习率 
- ‘incscaling’：随着时间t使用’power_t’的逆标度指数不断降低学习率learning_rate_ ，effective_learning_rate = learning_rate_init / pow(t, power_t)
- ‘adaptive’：只要训练损耗在下降，就保持学习率为’learning_rate_init’不变，当连续两次不能降低训练损耗或验证分数停止升高至少tol时，将当前学习率除以5.

7. power_t: double, 可选, default 0.5，只有solver=’sgd’时使用，是逆扩展学习率的指数.当learning_rate=’invscaling’，用来更新有效学习率。 

8. max_iter: int，可选，默认200，最大迭代次数。

9. random_state:int 或RandomState，可选，默认None，随机数生成器的状态或种子。 

10. shuffle: bool，可选，默认True,只有当solver=’sgd’或者‘adam’时使用，判断是否在每次迭代时对样本进行清洗。 

11. tol：float, 可选，默认1e-4，优化的容忍度 

12. learning_rate_int:double,可选，默认0.001，初始学习率，控制更新权重的补偿，只有当solver=’sgd’ 或’adam’时使用。 

14. verbose : bool, 可选, 默认False,是否将过程打印到stdout 

15. warm_start : bool, 可选, 默认False,当设置成True，使用之前的解决方法作为初始拟合，否则释放之前的解决方法。 

16. momentum : float, 默认 0.9,动量梯度下降更新，设置的范围应该0.0-1.0. 只有solver=’sgd’时使用. 17. nesterovs_momentum : boolean, 默认True, Whether to use Nesterov’s momentum. 只有solver=’sgd’并且momentum > 0使用.

18. early_stopping : bool, 默认False,只有solver=’sgd’或者’adam’时有效,判断当验证效果不再改善的时候是否终止训练，当为True时，自动选出10%的训练数据用于验证并在两步连续迭代改善，低于tol时终止训练。 

19. validation_fraction : float, 可选, 默认 0.1,用作早期停止验证的预留训练数据集的比例，早0-1之间，只当early_stopping=True有用

20. beta_1 : float, 可选, 默认0.9，只有solver=’adam’时使用，估计一阶矩向量的指数衰减速率，[0,1)之间 

21. beta_2 : float, 可选, 默认0.999,只有solver=’adam’时使用估计二阶矩向量的指数衰减速率[0,1)之间

22. epsilon : float, 可选, 默认1e-8,只有solver=’adam’时使用数值稳定值。
'''
