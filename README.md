# The instruction for MTOP benchmark CEC17-S-MTO

CEC17-S-MTO is an MTOP benchmark used for "Locality Sensitive Hashing-based Knowledge Transfer for Multitask Optimization" (Under Review).

## 1.Problem description

CEC17-S-MTO is derived from CEC2017 single objective real-parameter numerical optimization benchmark[^CEC17-S] (abbreviated as CEC17-S). In CEC17-S, 29 functions are involved, whose dimensionalities are selective among 10, 30, 50, and 100. The source code of CEC17-S[^CEC17-S-link] is also offered here. In CEC17-S-MTO, 20 of 29 different functions are randomly selected in CEC17-S to form 10 MTOPs. Meanwhile, each function is assigned a random dimension from the four alternatives (10, 30, 50, and 100). The details of CEC17-S-MTO benchmark are listed below.

|    Problem    | Task |       Function       | Dimension | Optimum |
| :-----------: | :--: | :------------------: | :-------: | :-----: |
| CEC17-S-MTO1  |  T1  |         Hf09         |    10     |  1800   |
|               |  T2  |         Hf03         |    100    |  1200   |
| CEC17-S-MTO2  |  T1  | Lunacek Bi-Rastrigin |    10     |   600   |
|               |  T2  |         Levy         |    10     |   800   |
| CEC17-S-MTO3  |  T1  |         Hf08         |    10     |  1700   |
|               |  T2  |         Cf02         |    10     |  2100   |
| CEC17-S-MTO4  |  T1  | Expanded Schaffer F6 |    50     |   500   |
|               |  T2  |      Bent Cigar      |    30     |   100   |
| CEC17-S-MTO5  |  T1  |       Schwefel       |    10     |   900   |
|               |  T2  |      Rosenbrock      |    30     |   300   |
| CEC17-S-MTO6  |  T1  |         Hf07         |    50     |  1600   |
|               |  T2  |         Cf06         |    100    |  2500   |
| CEC17-S-MTO7  |  T1  |         Hf01         |    100    |  1000   |
|               |  T2  |         Cf03         |    100    |  2200   |
| CEC17-S-MTO8  |  T1  |         Cf10         |    10     |  2900   |
|               |  T2  |         Hf04         |    50     |  1300   |
| CEC17-S-MTO9  |  T1  |      Rastrigin       |    100    |   400   |
|               |  T2  |         Cf05         |    100    |  2400   |
| CEC17-S-MTO10 |  T1  |       Zakharov       |    10     |   200   |
|               |  T2  |         Hf05         |    30     |  1400   |

The benchmark can be reorganized by generating a random **function list** and **dimension list**, which can be found in **'CEC17_S_MTO.py'** of the python version code and **'CEC17_MTSO_NEW.m'** of the matlab version code. Note that the official code initially forms 30 functions but the second function is deleted. Therefore, when forming the function list, number 2 should not appear in the list. The function numbers of 3 to 30 in the codes correspond to the 2rd to 29th functions in the document. We have also processed the returned values of these 28 functions ($-100$) to make them identical to the document.

~~~python
import numpy as np
# generate the function list
func_nums = np.random.choice(np.arange(1, 30), 20, replace=False)
func_nums[func_nums >= 2] += 1
# generate the dimension list
Dims = np.random.choice([10, 30, 50, 100], 20, replace=True)
~~~

## 2.Python version

Usage:

~~~python
import CEC17_S_MTO
# Get each problem
Prob = CEC17_S_MTO.Benchmark(i) # i = 1, 2, ..., 10
# Function Evaluation
Task1 = Prob[0]
f1 = Task1.function(x)
Task2 = Prob[1]
f2 = Task2.function(x)
~~~

## 3.Matlab version

The code of matlab version can be executed by the MTO Platform[^MTO]. You can put the whole "matlab" folder into any subdirectory of the "Problems" in MTO Platform. The platform will automatically recognize these newly added problems.

[^CEC17-S]: G. H. Wu, R. Mallipeddi, P. N. Suganthan, “Problem definitions and evaluation criteria for the CEC 2017 competition and special session on constrained single objective real-parameter optimization,” Nanyang Technol. Univ., Singapore, Tech. Rep, 2016: 1-18.
[^CEC17-S-link]: [https://github.com/P-N-Suganthan/CEC2017-BoundContrained](https://github.com/P-N-Suganthan/CEC2017-BoundContrained)
[^MTO]: [https://github.com/intLyc/MTO-Platform](https://github.com/intLyc/MTO-Platform)

