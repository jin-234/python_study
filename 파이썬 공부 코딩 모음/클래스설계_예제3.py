class MyCalData():
    def __init__(self,arg):
        self.arg=arg
      #  print(self.arg[0][0])
    def Avgdisplay(self):
        total=0
        for p in self.arg:
            total+=p[1]
        total=total/len(self.arg)
        print(f'{total:.3f}')
data=MyCalData([("kim",100), ("Park",90), ("Hong",70)])
data.Avgdisplay() #(100+90+70)3 에 대한 평균을 출력
