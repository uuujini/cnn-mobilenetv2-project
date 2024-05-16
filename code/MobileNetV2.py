from keras.layers import Input, Conv2D, DepthwiseConv2D, BatchNormalization, ReLU, GlobalAveragePooling2D, Dense
from keras.models import Model

def MobileNetV2(input_shape=(299, 299, 3), num_classes=101):
    input_tensor = Input(shape=input_shape)
    
    # 첫 번째 레이어
    x = Conv2D(32, (3, 3), strides=(2, 2), padding='same', use_bias=False)(input_tensor)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    
    # Depthwise Separable Convolution 블록
    def _depthwise_conv_block(x, filters, strides=(1, 1)):
        # Depthwise Convolution
        x = DepthwiseConv2D((3, 3), padding='same', strides=strides, use_bias=False)(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        
        # Pointwise Convolution
        x = Conv2D(filters, (1, 1), padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        return x
    
    # 중간 레이어들
    x = _depthwise_conv_block(x, 64, strides=(1, 1))
    x = _depthwise_conv_block(x, 128, strides=(2, 2))
    x = _depthwise_conv_block(x, 128, strides=(1, 1))
    x = _depthwise_conv_block(x, 256, strides=(2, 2))
    x = _depthwise_conv_block(x, 256, strides=(1, 1))
    x = _depthwise_conv_block(x, 512, strides=(2, 2))
    
    for _ in range(5):
        x = _depthwise_conv_block(x, 512, strides=(1, 1))
    
    x = _depthwise_conv_block(x, 1024, strides=(2, 2))
    x = _depthwise_conv_block(x, 1024, strides=(1, 1))
    
    # 최종 레이어
    x = GlobalAveragePooling2D()(x)
    output_tensor = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=input_tensor, outputs=output_tensor)
    return model

# 모델 생성
model = MobileNetV2()
model.summary()