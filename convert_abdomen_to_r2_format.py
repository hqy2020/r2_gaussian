import pickle
import numpy as np

# 原始NAF路径
src_path = '/home/qyhu/Documents/SAX-NeRF-master/data/abdomen_50.pickle'
# 保存为r2格式的路径
dst_path = '/home/qyhu/Documents/r2_gaussian/data_new/abdomen_r2_format.pickle'

with open(src_path, 'rb') as f:
    naf_data = pickle.load(f)

r2_data = {
    'mode': naf_data.get('mode', 'cone'),
    'DSD': naf_data.get('DSD', 1500.0),
    'DSO': naf_data.get('DSO', 1000.0),
    'sDetector': naf_data.get('nDetector', [1024, 1024]),
    'dDetector': naf_data.get('dDetector', [1.0, 1.0]),
    'nVoxel': naf_data.get('nVoxel', [512, 512, 463]),
    'sVoxel': np.multiply(naf_data.get('nVoxel', [512, 512, 463]), naf_data.get('dVoxel', [0.625, 0.625, 1.0])).tolist(),
    'offOrigin': naf_data.get('offOrigin', [0, 0, 0]),
    'offDetector': naf_data.get('offDetector', [0, 0]),
    'accuracy': naf_data.get('accuracy', 0.5),
    'totalAngle': naf_data.get('totalAngle', 180),
    'startAngle': naf_data.get('startAngle', 0.0),
    'numTrain': naf_data['numTrain'],
    'numVal': naf_data['numVal'],
    'train': {
        'projections': naf_data['train']['projections'],
        'angles': naf_data['train']['angles']
    },
    'val': {
        'projections': naf_data['val']['projections'],
        'angles': naf_data['val']['angles']
    }
}

with open(dst_path, 'wb') as f:
    pickle.dump(r2_data, f)
