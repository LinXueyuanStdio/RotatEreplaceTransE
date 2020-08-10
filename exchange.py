import pickle

lang = 'fr_en'
result = []
count = 0
RTElines = pickle.load(open('data/'+lang+'/RTentsembed.pkl','rb'),encoding='utf-8')
for i in range(RTElines.shape[0]):
    tmp = RTElines[i]
    count+=1
    for j in tmp:
        result.append(j)
print(count)

f = open('data/'+lang+'/transe_init.txt', 'w')
for i in result:
    f.write(str(i)+'\n')
f.close()

f = open('data/'+lang+'/transe_init.txt', 'r')
line = f.readline()
while line:
    print(line)
    line = f.readline()
f.close()