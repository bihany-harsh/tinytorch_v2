import tinytorch as tt

t1 = tt.tensor([[1, 2, 3, 4, 5], [6, 7, 0, 9, 10]], dtype=tt.Dtype.Float64)

print(t1)
print(t1.shape)
print(t1.size())
print(t1.size(1))