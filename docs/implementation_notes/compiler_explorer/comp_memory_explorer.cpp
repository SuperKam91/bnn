int a;
const int b = 5;
int c = 10;
int d[5];
int e[5] = {1,2,3,4,5};
int * g = 0;
int * h = new int;
int * i = new int[c];
int * j = new int[11];
int * cc = new int[c]{1,2,3,4,5,6};

class Rectangle {
  public: 
    int t;
    const int u = 9; // in c++11 this initialises as if it were in constructor
    int v = 10;
    int w[5];
    int x[5] = {1,2,3,4,5};
    int * y = 0;
    int * z = new int;
    int * aa = new int[v];
    int * bb = new int[11];
    int * dd = new int[v]{1,2,3,4,5,6,7,8};
    void nowt();
    Rectangle();
};

void Rectangle::nowt () {
}

Rectangle::Rectangle() {
  t = 22;
}

void func();

int main(){
    int k;
    const int l = 99;
    int m = 74;
    int n[9];
    int o[7] = {33,44,55,66,77,88,99};
    int * p = new int;
    int * q = new int[m];
    int * r = new int[10];
    int * ee = new int[6]{69,69,69};
    k = 6;
    return 0;
    }

void func()
{
int s = 6;
}


