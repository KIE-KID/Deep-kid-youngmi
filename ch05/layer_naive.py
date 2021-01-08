#단순한 계층 구현
#곱셈 계층과 덧셈 계층의 구현

#곱셉 노드
class MulLayer:
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self,x,y):
        self.x = x
        self.y = y
        out = x * y

        return out

    def backward(self, dout):
        #x y를 바꿔서 곱해준다.
        dx = dout * self.y
        dy = dout * self.x

        return dx, dy
#덧셈 노드
