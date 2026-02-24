function Tasks = CEC17_MTSO_NEW(index)
%BENCHMARK function
%   Input
%   - index: the index number of problem set
%
%   Output:
%   - Tasks: benchmark problem set

% The function list and dimension list
func_list = [19, 13, 7, 9, 18, 22, 6, 1, 10, 4, 17, 26, 11, 23, 30, 14, 5, 25, 3, 15];
dim_list = [10, 100, 10, 10, 10, 10, 50, 30, 10, 30, 50, 100, 100, 100, 10, 50, 100, 100, 10, 30];

% 10 groups
group_size = 2;
num_groups = length(func_list) / group_size;

% check the index
if index < 1 || index > num_groups
    error('Index must be between 1 and %d', num_groups);
end

% func_num and dims
func_idx = (index-1)*group_size + 1;
func1 = func_list(func_idx);
func2 = func_list(func_idx + 1);
dim1 = dim_list(func_idx);
dim2 = dim_list(func_idx + 1);

% bounds
lb1 = -100;
lb2 = -100;
ub1 = 100;
ub2 = 100;

% Task 1
Tasks(1).Dim = dim1;
Tasks(1).Lb = lb1;
Tasks(1).Ub = ub1;
Tasks(1).Fnc = @(x)get_func(x, func1);

% Task 2
Tasks(2).Dim = dim2;
Tasks(2).Lb = lb2;
Tasks(2).Ub = ub2;
Tasks(2).Fnc = @(x)get_func(x, func2);

end

function [Obj, Con] = get_func(x, fnc)
% Call CEC2017
[Obj_temp] = cec17_func(x', fnc);
Obj = Obj_temp';
% Function 2 is deleted from original codes of CEC2017
if fnc > 2
    Obj = Obj - 100;
end
Con = zeros(size(x, 1), 1);
end