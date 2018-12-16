from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

import numpy as np
import os
import pickle
import sys
import tensorflow as tf
from tqdm import tqdm


def load_graph(model_file):
    graph = tf.Graph()
    graph_def = tf.GraphDef()

    with open(model_file, "rb") as f:
        graph_def.ParseFromString(f.read())
    with graph.as_default():
        tf.import_graph_def(graph_def)

    return graph


def read_tensor_from_image_file(frames, input_height=299, input_width=299, input_mean=0, input_std=255):
    input_name = "file_reader"
    frames = [(tf.read_file(frame, input_name), frame) for frame in frames]
    #tensor of type string
    #frmaes = tensor 들의 집합
    decoded_frames = []
    for frame in frames:
        file_name = frame[1] #~~~.jpg
        file_reader = frame[0] #"file_reader"
        if file_name.endswith(".png"):
            image_reader = tf.image.decode_png(file_reader, channels=3, name="png_reader")
        elif file_name.endswith(".gif"):
            image_reader = tf.squeeze(tf.image.decode_gif(file_reader, name="gif_reader"))
        elif file_name.endswith(".bmp"):
            image_reader = tf.image.decode_bmp(file_reader, name="bmp_reader")
        else:
            image_reader = tf.image.decode_jpeg(file_reader, channels=3, name="jpeg_reader")
        decoded_frames.append(image_reader)#imagereader들의 집합
    float_caster = [tf.cast(image_reader, tf.float32) for image_reader in decoded_frames]#imagereader의 배열 분류값을 float32로 설정
    float_caster = tf.stack(float_caster) #설정한 imagereader들을 쌓음
    resized = tf.image.resize_bilinear(float_caster, [input_height, input_width]) #해당 imagereader를 통해 이미지의 input 사이즈를 조정
    #Tensor("ResizeBilinear:0", shape=(stack_size, 299, 299, 3), dtype=float32)
    normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std]) #tensor(shape, type, rank) 정규화
    sess = tf.Session()
    result = sess.run(normalized)
    return result #tensor 들을 반환

def load_labels(label_file):
    label = []
    proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
    for l in proto_as_ascii_lines:
        label.append(l.rstrip())
    return label


def predict(graph, image_tensor, input_layer, output_layer):
    input_name = "import/" + input_layer
    #import/Placeholder
    output_name = "import/" + output_layer
    #import/final_result or "import/module_apply_default/InceptionV3/Logits/GlobalPool"
    input_operation = graph.get_operation_by_name(input_name) #placeholder이름인 input작업 반환
    output_operation = graph.get_operation_by_name(output_name) #globalPool이름인 ouput 작업 반환
    with tf.Session(graph=graph) as sess:
        results = sess.run(
            output_operation.outputs[0],
            {input_operation.outputs[0]: image_tensor}
        )
    results = np.squeeze(results)
    return results


def predict_on_frames(frames_folder, model_file, input_layer, output_layer, batch_size):
    input_height = 299
    input_width = 299
    input_mean = 0
    input_std = 255
    batch_size = batch_size
    graph = load_graph(model_file)

    labels_in_dir = os.listdir(frames_folder) #frames_folder안의 파일명 리스트 반환 -> [ki, ni, di, li]
    frames = [each for each in os.walk(frames_folder) if os.path.basename(each[0]) in labels_in_dir]
    #os.walk : 시작 디렉터리부터 하위의 모든 디렉토리를 차례대로 방문해주는 함수
    #each[0] =train_frames, train_frames\di1, train_frames\ki1, train_frames\li1, train_frames\ni1
    #frames = [('train_frames\\di1', [], ['20181201_212904_frame_0.jpeg'...]), ('train_frames\\ki1', ...)

    predictions = []
    for each in frames:
        label = each[0]
        print("Predicting on frame of %s\n" % (label))
        for i in tqdm(range(0, len(each[2]), batch_size), ascii=True):
            #batch_size=100, len(each[2]) = 6040
            batch = each[2][i:i + batch_size]
            try:
                framespath = [os.path.join(label, frame) for frame in batch]
                #framespath = 프레임 들의 경로 리스트(0~100->101~200...)
                frames_tensors = read_tensor_from_image_file(framespath, input_height=input_height, input_width=input_width, input_mean=input_mean, input_std=input_std)
                #frame의 tensor 집합 shpae(stack_size, 299, 299, 3)
                pred = predict(graph, frames_tensors, input_layer, output_layer)
                pred = [[each.tolist(), os.path.basename(label)] for each in pred]
                #결과 리스트, 'di' 인 리스트
                predictions.extend(pred)

            except KeyboardInterrupt:
                print("You quit with ctrl+c")
                sys.exit()

            except Exception as e:
                print("Error making prediction: %s" % (e))
                x = input("\nDo You Want to continue on other samples: y/n")
                if x.lower() == 'y':
                    continue
                else:
                    sys.exit()
    return predictions


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("graph", help="graph/model to be executed")
    parser.add_argument("frames_folder", help="'Path to folder containing folders of frames of different gestures.'")
    parser.add_argument("--input_layer", help="name of input layer", default='Placeholder')
    parser.add_argument("--output_layer", help="name of output layer", default='final_result')
    parser.add_argument('--test', action='store_true', help='passed if frames_folder belongs to test_data')
    parser.add_argument("--batch_size", help="batch Size", default=10)
    args = parser.parse_args()

    model_file = args.graph
    frames_folder = args.frames_folder
    input_layer = args.input_layer
    output_layer = args.output_layer
    batch_size = int(args.batch_size)

    if args.test:
        train_or_test = "test"
    else:
        train_or_test = "train"

    # reduce tf verbosity
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    predictions = predict_on_frames(frames_folder, model_file, input_layer, output_layer, batch_size)

    out_file = 'predicted-frames-%s-%s.pkl' % (output_layer.split("/")[-1], train_or_test)
    print("Dumping predictions to: %s" % (out_file))
    with open(out_file, 'wb') as fout:
        pickle.dump(predictions, fout)

    print("Done.")
