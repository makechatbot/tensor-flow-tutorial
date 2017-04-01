# TensorFlow 공부


개인적으로 학습을 위한 소스입니다.

진행 중 TensorFlow가 1.0으로 업데이트 되어 적용하였습니다.

Python 3.6을 사용합니다.

## 요구사항

- Mac
- TensorFlow 1.0.0
- Python 3.6.0
    - numpy 1.12.0
    - matplotlib 2.0.0

## 간단한 설명

### [공부한 내용 정리]
- [tf.transpose 설명](./08-StudyForMySelf/transpose_example.py)
- transPose1 = tf.transpose(x, perm=perm);
- perm은 permutation의 약자로 치환, 순열, 바꾸기를 사전적인 의미를 가진다. 이 곳에서는 dimensions에 관한 변환이다.

    - perm[0, 1] 일 때
        - 차원 0 를 차원 0으로 변환
        - 차원 1 를 차원 1으로 변환
        - 그러므로 변화되는 것이 없는 것이다.
    - perm 이 =  [1, 0]  일 때
        - 차원 0 를 차원 1으로 변환
        - 차원 1 를 차원 0으로 변환
        - 뒤 바뀐다.

    - perm=[1,2,0] 일 때
        - 차원 1 를 차원 0으로 변환
        - 차원 2 를 차원 1으로 변환
        - 차원 0 를 차원 2으로 변환

        - x[ 4 ][ 2 ][ 3 ]
        - y[ 2 ][ 3 ][ 4 ]
        - y[x[1]{=2}][x[2]{=3}][x[0]{=4}]
