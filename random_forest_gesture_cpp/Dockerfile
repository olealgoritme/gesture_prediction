FROM davisking/dlib:latest

WORKDIR /usr/src/app

COPY . .

RUN apt-get update && apt-get install -y cmake g++ libopencv-dev

RUN mkdir build
WORKDIR /usr/src/app/build
RUN cmake ..
RUN make

CMD ["./generate_data && ./train && ./run"]
