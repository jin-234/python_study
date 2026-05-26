class PersonInfo():
    def __init__(self,*args):
        self.args=args
        # print(self.args)
    def Display(self):
        self.args=list(self.args)
        print(f'이름 : {self.args[0]},나이 : {self.args[1]},지역 : {self.args[2]}')

per1=PersonInfo('Hong',30,"Seoul")
per2=PersonInfo('Kim',50,'Daejeon')
per3=PersonInfo('Park',40,'Busan')


perlist=[per1,per2,per3]
for item in perlist:
    item.Display()   #"이름 : Hong,나이 : 30, 지역 : Seoul"
    print('='*30)