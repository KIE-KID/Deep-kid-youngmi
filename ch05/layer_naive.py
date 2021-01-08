#단순한 계층 구현
#곱셈 계층과 덧셈 계층의 구현

#곱셉 노드
class MulLayer:
    def __init__(self):
        #필요한 입력없이 초기화
        self.x = None
        self.y = None

    #순전파
    def forward(self,x,y):
        self.x = x
        self.y = y
        out = x * y

        return out

    # 역전파
    def backward(self, dout):
        #미분에 x y를 바꿔서 곱해준다.
        dx = dout * self.y
        dy = dout * self.x

        return dx, dy

#덧셈 노드
class AddLayer:
    def __init__(self):
        self.x = None
        self.y = None

    #순전파
    def forward(self,x, y):
        self.x = x
        self.y = y
        out = x + y

        return out

    #역전파
    def backward(self, dout):
        #상류에서 내려온 미분을 그대로 흘려보낸다.
        dx = dout * 1
        dy = dout * 1

        return dx, dy



