import h5py


def save_h5(data, path):
    f = h5py.File(path, 'w')
    a = data.data.cpu().numpy()
    print(a.shape)
    f.create_dataset('data', data=a)
    f.close()


def save_obj(point, path):
    n = point.shape[0]
    with open(path, 'w') as f:
        for i in range(n):
            f.write("v {0} {1} {2}\n".format(point[i][0], point[i][1], point[i][2]))
    f.close()
