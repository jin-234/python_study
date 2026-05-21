def AlphaFindFunc(a):#영어 대소문자만 추출 출력 하는 함수
    result=''
    for alph in a:
        if 'a'<=alph<='z':
            result+=alph
        elif 'A'<= alph<='Z':
            result+=alph
    return result
stddata='# Ai % 3 pro&*graM'
result=AlphaFindFunc(stddata)#함수 호출
print('result : ',result)#AiprogeaM



def WordCountFunc(len):#같은 단어끼리 모아서 개수 새기
    dic={}
    for asg in len:
        if asg in dic:
            dic[asg]=dic[asg]+1
        else:
            dic[asg]=1
    return dic
listdata=['python','ai','study','good','ai','python','ai']

result=WordCountFunc(listdata)
print(result)#{'python':2,'ai':3,'study':1,'good':1}


def InforCombine(key,value):#리스트 두개를 합쳐서 dict형식으로 바꿔주는 함수
    dic={}
    for k in range(0,len(key)):
        dic[key[k]]=value[k]
    return dic
key_list = ['name','age','address']
value_list=['Hong',50,'seoul']

result=InforCombine(key_list,value_list)
print('result :',result)#{'name':'Hong','age':50,'address:'seoul']
