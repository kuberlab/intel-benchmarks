import grpc

from ml_serving import predict_pb2
from ml_serving import predict_pb2_grpc
from ml_serving import tensor_pb2
from ml_serving.utils import dtypes

port = 9000
host = '10.5.2.13'

filename = 'Screenshot_20180319_181842.png'
MAX_LENGTH = 67108864  # 64 MB
opts = [
    ('grpc.max_send_message_length', MAX_LENGTH),
    ('grpc.max_receive_message_length', MAX_LENGTH)
]


if __name__ == '__main__':
    server = '%s:%s' % (host, port)
    channel = grpc.insecure_channel(server, options=opts)

    stub = predict_pb2_grpc.PredictServiceStub(channel)

    # test = np.load('2.npy').reshape(1, 1, 28, 28).astype(np.float64)
    # shape = im.shape
    tensor_proto = tensor_pb2.TensorProto(
        dtype=dtypes.string.as_datatype_enum,
        #tensor_shape=tensor_shape.as_shape(shape).as_proto()
    )
    tensor_proto.string_val.append(open(filename, 'rb').read())
    inputs = {'images': tensor_proto}
    response = stub.Predict(predict_pb2.PredictRequest(inputs=inputs))

    res = list(response.outputs.values())[0].string_val[0]

    open('result.png', 'wb').write(res)
    # print('Received from serving:\n%s' % response)
    print('Saved to result.png.')
