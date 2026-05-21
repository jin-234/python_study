def EncryptFunc(msg):
    for alph in msg:
        if alph in EncBook:
            msg=msg.replace(alph,EncBook[alph])
    return msg


stringdata='I love ai python programming'
EncBook={'1':'#','p':'#','o':'7','g':'$','I':'%','a':'8','t':'*','r':'3','n':'6'}
encoding=EncryptFunc(stringdata)#전달된 문자열을 암호화 시켜서 반환하는 함수
print(encoding)