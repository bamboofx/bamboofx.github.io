import numpy as  np
def mprint(str):
    print("x"*30,str,"x"*30)
mprint("Cơ bản về về Numpy:")
mprint("Tạo mảng một chiều từ list")
a = np.array([1, 2, 3])
print(type(a)) #<class 'numpy.ndarray'>

print(a.shape) #(3,)
a[0] = 5
print(a) #[5 2 3]
b = np.array([[1, 2, 3], [4, 5, 6]])
print(b.size) #6
print(b)
# Tạo mảng từ khoảng cho trước
u = np.arange(3.0)
print(u) #[0. 1. 2.]
v = np.arange(3, 7)
print(v) #[3 4 5 6]

print(np.array([(1.5,2,3,4),(4,5,6,7)]),"Biến đổi một số tuple thành mảng 2 chiều")

print(np.array([[1,2],[3,4]],dtype=complex),"Định dạng khi tạo mảng")

# Thường thì các thành phần trong một array không xác định, nhưng size thì phải biết trước. Vì thế NumPy đưa cho chúng ta một vài hàm để tạo mảng
mprint(" Tạo mảng 2 chiều- ma trận")
# np.zeros tạo ma trận số 0 với shape(số cột,số hàng)
a = np.zeros((5, 3))
print(a)
#[[0. 0. 0.]
# [0. 0. 0.]
# [0. 0. 0.]
# [0. 0. 0.]
# [0. 0. 0.]]
#np.ones tạo mảng 2 chiều số 1 tương tự zeros
b = np.ones((5, 2))
# np.full() tạo mảng hằng số tham số tương tự như trên với shape gồm shape(số hàng,số cột), và tham số thứ 2 là hằng số cần tạo
c = np.full((3, 5), 3)
print(c)
# [[3 3 3 3 3]
# [3 3 3 3 3]
# [3 3 3 3 3]]
#np.eye(N,M,k) tạo ma trận chéo với N hàng và M cột,  nếu không gán giá trị M thì mặc định M sẽ = N
d = np.eye(5)
print(d)
#[[1. 0. 0. 0. 0.]
# [0. 1. 0. 0. 0.]
# [0. 0. 1. 0. 0.]
# [0. 0. 0. 1. 0.]
# [0. 0. 0. 0. 1.]]
print("-"*45)
# np.random.random((N.M)) Tạo ma trận ngẫu nhiên với N hàng M cột, với giá trị ngẫu nhiên từ 0->1
e=np.random.random((2,3))
print(e)
#[[0.67471134 0.96878567 0.13359926]
# [0.98475084 0.7250046  0.00370332]]
print(np.empty((2,3)),"Tạo mảng với np.empty cũng tương tự như với np.random.random")
# Để tạo một dãy số NumPy có một hàm tương tự với np.range là np.arange, nó trả về một array thay vì là một list như range
print(np.arange(10,30,5),"Tạo mảng từ 10 - tới 30 với số bước là 5")
print(np.arange(0,3,0.3)," np.arange cũng chấp nhận giá trị là số float")
# Khi dùng float làm tham số cho arange. Nó thường tạo ra một dãy số không thể xác định nên ta thường sử dụng thêm np.linspace để mặc định số phần tử được tạo ra thay vì số bước
print(np.linspace(0,2,9),"np.linspace(0,2,9) Tạo ra 9 phần tử là số float từ 0-2")
print(np.linspace(0,np.pi,180),"hoặc lấy số điểm trong 1 hình tròn")
print("-"*55)
# Slicing trong array 2 chiều, lấy giá trị của ma trận
a = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
print(a);
# a[N,M] với N là trỏ tới hàng và M là giá trị cần lấy
# vd a[2,1] là chọn hàng thứ 3 ( tính giá trị từ 0), và lấy giá trị thứ 2 của hàng này
# ta có một số cách slice giá trị như

print (a[1,1:3],"lấy giá trị từ 2->3 của hàng 2")
print(a[0,:]," lấy tất cả giá trị của hàng 1")

print(a[-1, :],"Lấy tất cả các giá trị của hàng cuối")
print(a[:, 1],"lấy tất cả giá trị ở cột 2 trên tất cả các hàng")

col_r=a[:, 0:2]
print(col_r,"Lấy tất cả giá trị ở cột từ 1->2 trên tất cả các hàng")
# a.shape trả về só hàng và cột của ma trận
print(col_r.shape,"Trả về số hàng và cột của ma trận bên trên");

print("-"*55)

mprint("Các phép toán trong numpy")

x = np.array([[1, 2], [3, 4]], dtype=np.float64)
y = np.array([[5, 6], [7, 8]], dtype=np.float64)
print("ma trận x:",x)
print("ma trận y:",y)
print(np.add(x, y) , "phép cộng")

print(np.subtract(x, y),"phép trừ")
print(np.divide(x, 2),"phép chia mảng 1 cho 2 ")
print(np.divide(x, y),"phép chia 2 mảng")
print(np.multiply(x, y),"phép nhân 2 mảng")
print(np.sqrt(x),"phép khai căn trên 1 mảng ")
print(np.sqrt((x,y)),"phép khai căn trên 2 mảng ")

mprint(" Phép toán trên hàng và cột ")
print(np.sum(x)," Tổng các giá trị trong ma trận x")
print(x.T,"phép chuyển vị ma trận x")
mprint("Phép toán nhân ma trận: Tích trong (inner Product)")
x = np.array([[1, 2], [3, 4]])
y = np.array([[5, 6], [7, 8]])
v = np.array([9, 10])
w = np.array([11, 12])
d = np.array([3, 4])
print("Ma trận x:\n",x)
print("Ma trận y:\n",y)
print("Ma trận v:\n",v)
print("Ma trận w:\n",w)
print("Ma trận d:\n",d)

print(np.dot(v, w),"= 9*11+10*12")
print(np.matmul(x, v),"= [9*1+10*2,9*3+10*4]")
print(np.dot(v, d)," = 9*3+10*4")
print(np.matmul(x, y),"= [[5*1+7*2,6*1+8*2],[5*3+7*4,6*3+8*4]]")
print(np.multiply(x, y),"= [[1*5,2*6],[3*7+8*4]]")

a = np.ones((2,3), dtype=int)
b = np.random.random((2,3))
a *= 3
print(a)
#[[3, 3, 3],
#       [3, 3, 3]]
b += a
print(b)
#[[ 3.417022  ,  3.72032449,  3.00011437],
#       [ 3.30233257,  3.14675589,  3.09233859]]
try:
    a+=b
except TypeError:
    print("ta không thể dùng thế này a += b, vì b là số float không tự động chuyển thành số nguyên nên sẽ báo lỗi TypeError")
#TypeError: Cannot cast ufunc add output from dtype('float64') to dtype('int64') with casting rule 'same_kind'
# Khi ta thực hiên phép toán trên các array với các định dạng khác nhau thì định dạng ở kết quả sẽ tự động lấy giá trị là các định dạng có tính chi tiết hơn
a = np.ones(3, dtype=np.int32)
b = np.linspace(0,np.pi,3)
print(b.dtype.name)
#'float64'
c = a+b
print(c)
#[ 1.        ,  2.57079633,  4.14159265])
print(c.dtype.name)
#'float64'
d = np.exp(c*1j)
print(d)
#[ 0.54030231+0.84147098j, -0.84147098+0.54030231j,
#       -0.54030231-0.84147098j]
print(d.dtype.name)
# complex128
#Một vài phép tính thực hiện trong các phần tử của array như sum(), hay lấy min,max
a = np.random.random((2,3))
print(a.sum())
#2.5718191614547998
print(a.min())
#0.1862602113776709
print(a.max())
#0.6852195003967595

#Mặc định là những phép toán trên sẽ tính toán tất cả các phần tử của array không cần biết định dạng của nó là gì. Nhưng chúng ta cũng có thể chỉ định
# nó tính toán theo hàng hoặc cột của array
b=np.arange(12).reshape(3,4)
print(b.sum(axis=0),"= 1 array tính tổng trên mỗi cột")
print(b.min(axis=1),"= 1 array số nhỏ nhất thuộc mỗi hàng ")
print(b.cumsum(axis=1),"= 1 array giữ nguyên định dạng với mỗi hàng được tính lũy tiến")

a=np.arange(10)**3
print(a,"Mảng a được tạo với 10 phần tử từ 0->10 và toán tử ** lấy số mũ mỗi phần tử")
# INDEXED
print(a[2]," vị trí thứ 3 index tính từ vị trí 0")
print(a[2:5]," lấy giá trị từ vị trí 3->vị trí 4")
# Gán
a[:6:2]=-1000
# Dấu : tượng trưng cho bắt đầu từ vị trí 0 tới vị trí số 6 cách mỗi 2 bước thì gán giá trị đó =-1000
print(a,"Dấu : tượng trưng cho bắt đầu từ vị trí 0 tới vị trí số 6 cách mỗi 2 bước thì gán giá trị đó =-1000")
a=a[: : -1]
print(a,"Đảo các phần tử từ cuối lên đầu ndarray")
# Duyệt qua tất cả các phần tử trong mảng với for và thực hiện phép tính với phần tử
for i in enumerate(a):
    d=i[-1]**(1/3.)
    print(str(round(d,2)),str(i[-1]),"lũy thừa ",str(round(1/3.,2))," của phần tử thứ : "+str(i[0]))
a=[1,2]
print("list a:\n",a)
print("type of a",type(a))
a=np.asarray(a)
print("type of a",type(a))
a=np.arange(6);print(a," Thay đổi array a giữ nguyên dữ liệu")
b=a.reshape(3,2);print(b," Chuyển đổi thành array 2 chiều với 3 hàng và 2 cột")
print(b.size," Tổng số phần tử của array 2 chiều a.size")
c=np.reshape(b,b.size);print(c,"Chuyển đổi từ array 2 chiều thành 1 chiều với reshape")
d=b.flatten();print(d,"Chuyển đổi từ array 2 chiều thành 1 chiều với flatten")
b=a.reshape(3,2)
c=a.reshape(3,2)
print(b)
print(np.insert(b,0,5),"sử dụng hàm np.insert(array,cột,giá trị) thêm giá trị vào cột 1 với giá trị bằng 5 và làm phẳng array")
print(np.insert(c,1,3,axis=1),"sử dụng hàm np.insert(array,cột,giá trị,axis=x) chèn thêm giá trị vào cột 2 với giá trị bằng 3,không thay đổi cấu trúc array với tham số axis=1")
print(np.insert(c,0,3,axis=0),"Sử dụng hàm np.insert(array,cột,giá trị,axis=x) chèn vào hàng 1 một hàng có giá trị bằng 3,không thay đổi cấu trúc array với tham số axis=0")
