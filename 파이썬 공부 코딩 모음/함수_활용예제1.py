def EncryptFunc(msg):
    for alph in msg:
        if alph in EncBook:
            msg=msg.replace(alph,EncBook[alph])
    return msg
def DecryptFunc(msg):
    DncBook={}
    for key in EncBook:
        DncBook[EncBook[key]]=key
    for alph in msg:
        if alph in DncBook:
            msg=msg.replace(alph,DncBook[alph])
    return msg
stringdata='I love ai python programming'
EncBook={'1':'#','p':'@','o':'7','g':'$','I':'%','a':'8','t':'*','r':'3','n':'6'}
encoding=EncryptFunc(stringdata)#전달된 문자열을 암호화 시켜서 반환하는 함수
print(encoding)
#위 암호화된 문자열 다시 복원시키는 함수 구현 
decmsg=DecryptFunc(encoding)
print(decmsg)
