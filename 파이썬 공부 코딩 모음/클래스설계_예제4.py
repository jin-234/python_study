class MyComInfo():
    def __init__(self,arg='Python Academy'): #디폴트 파라미터
        self.name=arg
    def DisplayName(self):
        print(self.name)
    def SettingName(self,rename):
        self.name=rename
com1=MyComInfo("Ai Academy")
com1.DisplayName() #Ai Academy
com2=MyComInfo()
com2.DisplayName()#Python Academy

com2.SettingName("Agent Academy")
com2.DisplayName()#Agent Academyac
