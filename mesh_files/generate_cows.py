class vertex():
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

with open("cow.geo", "r") as fp:
    # with open("multiplecows.geo", "rw") as newF:
    line = fp.readline()
    cnt = 1
    num = 1
    lfloat = []
    num2 = 0
    loc = []
    vertices = []
    while line:
        if(cnt == 1):
            num = int(line)
        if(cnt == 2):
            l = line.split()
            for x in l:
                x = float(x)
                lfloat.append(x)
        if(cnt == 3):
            num2 = int(line)
        if(cnt == 4):
            loc = line
        line = fp.readline()
        cnt += 1
lfloatstr = []
for i in range (0,len(lfloat)):
    # if(i%3 == 2):
    #     lfloat[i] -= 10
    if(i%3 == 0):
        lfloat[i] -= 6
for i in range (len(lfloat)):
    lfloatstr.append(str(lfloat[i]))

with open("cow5.geo", 'w') as fp:
    fp.write(str(num))
    fp.write("\n")
    fp.write(" ".join(lfloatstr))
    fp.write("\n")

    fp.write(str(num2))
    fp.write("\n")

    fp.write(loc)
