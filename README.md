Details about this assignment can be found [on the course webpage](http://cs231n.github.io/), under Assignment #1 of Spring 2020`.
"# DCGAN" 
"# DCGAN" 
## reference
> https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
## data set
"상표 이미지 데이타셋 - AIhub" 
jpg 이미지 파일만 대량으로 다운
> https://aihub.or.kr/aidata/30757
## Theoretical background
![image](https://user-images.githubusercontent.com/73246476/153514353-9f9bd710-317f-4968-a861-6589dfd7eefd.png)

  ▲ DCGAN의 기본적인 원리
## **DCGAN(Deep Convolutional Generative Adversarial Networks)**
:Unsupervised Learning의 한 종류의 대표적인 모델로서, generator network와 discriminator network가 서로를 견제하며 입력 데이터의 분포와 가장 비슷한 분포를 가진 가중치 (W)를 출력하는 프로그램
## **GAN의 목적**
: generator가 최대한 진짜 같은 가짜 이미지를 생성하게 하는 것.
## **장점**
: input을 학습한 model이 직접 input의 분포를 따르는 비슷한 이미지를 ‘생성’해내고, input에 정답 레이블 등 추가적인 정보가 크게 필요하지 않아 input의 제한적인 부분이 별로 없음.
## Algorithm 요약
![image](https://user-images.githubusercontent.com/73246476/153514974-f2f52a0f-b9d2-4358-be73-9ead54103eb0.png)


