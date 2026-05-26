class MyDataControl(): #클래스 정의
    def __init__(self,*args):
        self.arg =args
        # print(f'객체가 생성 완료!!{len(args)},{len(self.arg)}')


    def SumData(self):
        total=0
        for plus in self.arg:
            total+=plus
        print('total :',total)#50~90의 합을 출력

#객체생성 필요    #객체가 생성될때 '객체가 생성 완료!!'출력
value=MyDataControl(50,60,70,80,90) #객체를 메모리에 생성했으면 변수로 참조해야함
value.SumData()


