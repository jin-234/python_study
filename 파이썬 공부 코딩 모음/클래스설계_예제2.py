class PersonInfo():
    def __init__(self,name,age,local):
        self.name=name
        self.age=age
        self.local=local
        # print(self.args)
    def Display(self):
        print(f'이름 : {self.name},나이 : {self.age},지역 : {self.local}')

per1=PersonInfo('Hong',30,"Seoul")
per2=PersonInfo('Kim',50,'Daejeon')
per3=PersonInfo('Park',40,'Busan')


perlist=[per1,per2,per3]
for item in perlist:
    item.Display()   #"이름 : Hong,나이 : 30, 지역 : Seoul"
    print('='*30)
