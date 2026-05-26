class MyCallist():
    def __init__(self,arg1,arg2):
        self.arg1=arg1
        self.arg2=arg2
        #print(f'{arg1},{arg2}')
    def Sumoflist1(self):
        result=[]
        for count in range(0,len(self.arg1)):
            result.append(self.arg1[count]+self.arg2[count])
        print(result)
    def Sumoflist2(self):
        result=[]
        for count in self.arg1:
            if count not in self.arg2:
                result.append(count)
        print(result)


data=MyCallist([5,6,7,9], [8,9,5,10])

data.Sumoflist1() #[13,15,12,17]
data.Sumoflist2()# [6,7]