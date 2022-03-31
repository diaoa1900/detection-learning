import time
import paddle.nn

pre_anchor = [[(85, 66), (115, 146), (275, 240)], [(22, 45), (46, 33), (43, 88)], [(7, 10), (12, 22), (24, 17)]]


class Convolutional(paddle.nn.Layer):
    def __init__(self, in_channel, out_channel, k_size, stride, pad):
        super().__init__()
        self.conv = paddle.nn.Conv2D(in_channel, out_channel, k_size, stride, pad)
        self.bn = paddle.nn.BatchNorm2D(num_features=out_channel)
        self.leak_relu = paddle.nn.LeakyReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        output = self.leak_relu(x)
        return output


class ConvResidual(paddle.nn.Layer):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.convolutional1 = Convolutional(in_channel, out_channel, 1, 1, 0)
        self.convolutional2 = Convolutional(out_channel, out_channel * 2, 3, 1, 1)

    def forward(self, x):
        x1 = self.convolutional1(x)
        x2 = self.convolutional2(x1)
        return x + x2


class DarkNet53(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.output = []
        stage = [1, 2, 8, 8, 4]
        self.feature = [Convolutional(3, 32, 3, 1, 1)]
        for index, val in enumerate(stage):
            feature = paddle.nn.Sequential(('down', Convolutional(2 ** (index + 5), 2 ** (index + 6), 3, 2, 1)))  # 向下取整， 尺寸下降一半 (h-3+2)/2+1
            for v in range(val):
                feature.add_sublayer('cr'+str(v), ConvResidual(2 ** (index + 6), 2 ** (index + 5)))  # append再转Sequential不知道是否可行
            self.feature.append(feature)

    def forward(self, x):
        # print(self.feature)
        for f in range(len(self.feature)):
            x = self.feature[f](x)
            self.output.append(x)
        self.output.reverse()
        return self.output


class ConvolutionalFive(paddle.nn.Layer):
    def __init__(self, channel, out_chan):
        super().__init__()
        self.convolutional1 = Convolutional(channel, out_chan, 1, 1, 0)
        self.convolutional2 = Convolutional(out_chan, out_chan*2, 3, 1, 1)
        self.convolutional3 = Convolutional(out_chan*2, out_chan, 1, 1, 0)
        self.convolutional4 = Convolutional(out_chan, out_chan*2, 3, 1, 1)
        self.convolutional5 = Convolutional(out_chan*2, out_chan, 1, 1, 0)

    def forward(self, x):
        x = self.convolutional5(self.convolutional4(self.convolutional3(self.convolutional2(self.convolutional1(x)))))
        return x


class ConvolutionalConv(paddle.nn.Layer):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.convolutional1 = Convolutional(in_channel, in_channel*2, 3, 1, 1)
        self.conv2 = paddle.nn.Conv2D(in_channel*2, out_channel, 1, 1, 0)

    def forward(self, x):
        return self.conv2(self.convolutional1(x))


class YOLOv3(paddle.nn.Layer):
    def __init__(self, class_num, anchors):
        super().__init__()
        self.darknet53 = DarkNet53()
        self.output = []
        self.class_num = class_num
        self.anchors = anchors
        self.ConvFive = []
        self.ConvolutionalConv = []

    def forward(self, x):
        darknet_output = self.darknet53(x)
        ratio = 2
        for i in range(3):
            if i > 0:
                ratio = 3
                up_sample = (paddle.nn.Sequential(
                    Convolutional(darknet_output[i].shape[1], int(darknet_output[i].shape[1]/2), 1, 1, 0),
                    paddle.nn.Upsample(size=[darknet_output[i].shape[2], darknet_output[i].shape[3]])
                ))
                darknet_output[i] = paddle.concat([darknet_output[i], up_sample(self.ConvFive[i-1](darknet_output[i-1]))], axis=1)
            self.ConvFive.append(ConvolutionalFive(darknet_output[i].shape[1], darknet_output[i].shape[1] // ratio))  # append, =不行 //，要防止为浮点数
            self.ConvolutionalConv.append(ConvolutionalConv(int(darknet_output[i].shape[1] // ratio), 3 * (self.class_num + 5)))
            self.output.append(self.ConvolutionalConv[i](self.ConvFive[i](darknet_output[i])))
        return self.output


if __name__ == '__main__':
    t1 = time.time()
    yolo3 = YOLOv3(80, None)
    print(yolo3)
    fake_input = paddle.randn([1, 3, 416, 416], dtype='float32')
    out = yolo3(fake_input)
    for o in out:
        print(paddle.shape(o))
    print(time.time()-t1)
    # print(paddle.summary(yolo3, (1, 3, 416, 416), dtypes='float32'))
