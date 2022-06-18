import argparse, os, csv, glob, random

def write_list(data_list, path, ):
    with open(path, 'w') as f:
        writer = csv.writer(f, delimiter=',')
        for row in data_list:
            print(row)
            if row: writer.writerow(row)
    print('split saved to %s' % path)

parser = argparse.ArgumentParser(description = "WriteCSV");
parser.add_argument('--out_dir',  type=str, default='', help='Output direcotry');
opt = parser.parse_args();

setattr(opt,'real_dir_test',os.path.join(opt.out_dir,'test','pytmp','real'))
setattr(opt,'fake_dir_test',os.path.join(opt.out_dir,'test','pytmp','fake'))
setattr(opt,'real_dir_train',os.path.join(opt.out_dir,'train','pytmp','real'))
setattr(opt,'fake_dir_train',os.path.join(opt.out_dir,'train','pytmp','fake'))
setattr(opt,'csv_root',os.path.join(opt.out_dir))

realListTest = []
fakeListTest = []
realListTrain = []
fakeListTrain = []
train_set = []
test_set = []

for directory in os.listdir(opt.real_dir_train):
        realListTrain.append(directory)

for directory in os.listdir(opt.fake_dir_train):
        fakeListTrain.append(directory)

for directory in os.listdir(opt.real_dir_test):
        realListTest.append(directory)

for directory in os.listdir(opt.fake_dir_test):
        fakeListTest.append(directory)

random.shuffle(realListTrain)
random.shuffle(fakeListTrain)
random.shuffle(realListTest)
random.shuffle(fakeListTest)

#for i in range(len(fakeListTrain)):
for file in os.listdir(opt.fake_dir_train):
  if os.path.isdir(os.path.join(opt.fake_dir_train,file)):
    #audio_file = file + '.wav'
    if len(os.listdir(os.path.join(opt.fake_dir_train,file)))==20:
      train_set.append([os.path.join(opt.fake_dir_train,file),'fake'])


for file in os.listdir(opt.real_dir_train):
  if os.path.isdir(os.path.join(opt.real_dir_train,file)):
    #audio_file = file + '.wav'
    if len(os.listdir(os.path.join(opt.real_dir_train,file)))==20:
      train_set.append([os.path.join(opt.real_dir_train,file),'real'])


for file in os.listdir(opt.fake_dir_test):
  if os.path.isdir(os.path.join(opt.fake_dir_test,file)):
    #audio_file = file + '.wav'
    if len(os.listdir(os.path.join(opt.fake_dir_test,file)))==20:
      test_set.append([os.path.join(opt.fake_dir_test,file),'fake'])

for file in os.listdir(opt.real_dir_test):
  if os.path.isdir(os.path.join(opt.real_dir_test,file)):
    #audio_file = file + '.wav'
    if len(os.listdir(os.path.join(opt.real_dir_test,file)))==20:
      test_set.append([os.path.join(opt.real_dir_test,file),'real'])

write_list(train_set, os.path.join(opt.csv_root, 'train_split_lstm.csv'))
write_list(test_set, os.path.join(opt.csv_root, 'test_split_lstm.csv'))