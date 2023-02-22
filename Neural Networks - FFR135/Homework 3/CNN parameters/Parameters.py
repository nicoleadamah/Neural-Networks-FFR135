
def calc_Conv(k,z, nr_k):
    parameters = (((k**2)*z)+1)*nr_k
    return parameters
def calc_fully(c,p):
    parameters = (c*p)+(1*c)
    return parameters

def calc_output():
    pass

#input
x1 = 25
y1 = 25
z1 = 3
#filter/kernel
k1=3
stride = 2
nmbr_k = 14
pad=1

#conv size
#13x13x14
conv_size = ((x1-k1+(2*pad))/stride)+1
#maxpoolsize
#12x12x14
p_size = ((13-2)/1)+1
# fully connected layer
FC = 15
# output
Out = 10
Fc_size = ((x1-k1+(2*pad))/stride)+1

conv1 = calc_Conv(k1,z1,nmbr_k)
print(conv1)
s = 12*14*14
s1 = 15
fully_c = calc_fully(s1, s)
print(fully_c)
s2 = 10
fully_c2 = calc_fully(s2, s1)
print(fully_c2)

