from imageai.Prediction import ImagePrediction
import numpy as np
import h5py
import os


def convert_file(input_dir, filename, output_dir):
    filepath = input_dir + '/' + filename
    fin = open(filepath, 'rb')
    binary_data = fin.read()
    new_filepath = output_dir + '/' + filename[:-4] + '.hdf5'
    f = h5py.File(new_filepath)
    dt = h5py.special_dtype(vlen=np.dtype('uint8'))
    data_set = f.create_dataset('binary_data', (100, ), dtype=dt)
    data_set[0] = np.fromstring(binary_data, dtype='uint8')


execution_path = os.getcwd()
prediction = ImagePrediction()
prediction.setModelTypeAsResNet()
prediction.setModelPath(execution_path + "\\resnet50_weights_tf_dim_ordering_tf_kernels.h5")
prediction.loadModel()


predictions, percentage_probabilities = prediction.predictImage(execution_path + "\\car.jpeg", result_count=5)
for index in range(len(predictions)):
    print(predictions[index], " : ", percentage_probabilities[index])

