class MyComInfo():
    def __init__(self,arg='Python Academy'): #디폴트 파라미터
        self.arg=arg
    def DisplayName(self):
        print(self.arg)
com1=MyComInfo("Ai Academy")
com1.DisplayName() #Ai Academy
com2=MyComInfo()
com2.DisplayName()#Python Academy