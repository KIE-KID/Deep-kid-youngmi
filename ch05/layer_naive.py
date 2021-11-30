#1. 곱셈 계층과 덧셈 계층의 구현한 코드
#장에서는 곱셈 노드와 덧셈 노드를 ‘계층’ 단위로 구현
class MulLayer: # 계산 그래프의 곱셉 노드
    # 인스턴스 변수인 x와 y를 초기화. x, y는 순전파 시의 입력값을 유지함.
    def __init__(self):
        self.x = None
        self.y = None

    #순전파, x와 y를 인수로 받고 두 값을 곱해서 반환.
    def forward(self,x,y):
        self.x = x
        self.y = y
        out = x * y

        return out

    # 역전파, 상류에서 넘어온 미분(dout)에 순전파 때의 값을 ‘서로 바꿔’ 곱한 후 하류로 흘림.
    # 앞 절에서 계산 그래프의 순전파와 역전파를 써서 계산.
    def backward(self, dout):
        #미분에 x y를 바꿔서 곱해준다.
        dx = dout * self.y
        dy = dout * self.x

        return dx, dy

class AddLayer: # 계산 그래프의 덧셈 노드
    # 덧셈 계층에서는 초기화가 필요 없으므로, pass로 아무 일도 안함.
    def __init__(self):
        self.x = None
        self.y = None

    #순전파, 입력 받은 두 인수 x, y를 더해서 반환.
    def forward(self,x, y):
        self.x = x
        self.y = y
        out = x + y

        return out

    #역전파, 상류에서 내려온 미분(dout)을 그대로 하류로 흘림
    def backward(self, dout):
        dx = dout * 1
        dy = dout * 1

        return dx, dy



