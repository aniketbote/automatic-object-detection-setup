import subprocess
from git import Repo
import os
import requests, zipfile, io
import fileinput
import tarfile
import pandas as pd
from shutil import copyfile
import sys
import re

try:
    import pycocotools
except ImportError as e:
    #pycocotools installation
    print('Installing Pycocotools')
    pip_cmd = 'pip install {}'.format(pycocotools_url)
    pip_data = execute(pip_cmd, ret = True)
    if re.search('Successfully installed pycocotools',str(pip_data)) or re.search('Successfully built pycocotools',str(pip_data)):
        print('Successfully installed pycocotool')
    else:
        print('ERROR in installing pycocotools')
        execute(pip_cmd, out = True)



#variables
pycocotools_url = "git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI"
git_url = 'https://github.com/tensorflow/models.git'
repo_path = os.getcwd()
tensorflow_path = os.path.join(repo_path,'TensorFlow')
protobuf_url = 'https://github.com/protocolbuffers/protobuf/releases/download/v3.11.1/protoc-3.11.1-win64.zip'
label_img = 'https://www.dropbox.com/s/kqoxr10l3rkstqd/windows_v1.8.0.zip?dl=1'
flag = True

def download_file(url,file_type,destination = '-1'):
    if file_type == 'zip':
        r = requests.get(url)
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extractall()
    elif file_type == 'git':
        Repo.clone_from(url,destination)
    elif file_type == 'tar':
        target_path = url.split('/')[-1]
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(target_path, 'wb') as f:
                f.write(response.raw.read())
        if (target_path.endswith("tar.gz")):
            tar = tarfile.open(target_path, "r:gz")
            tar.extractall()
            tar.close()


def execute(cmd,out = False, ret = False):
    global flag
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    file_name = os.path.join(repo_path,'temp','logs.txt')
    data = p.stdout.readlines()
    if out == True:
        for line in data:
            print(line)
    if flag:
        f= open(file_name,'w+')
        for line in data:
            f.write(line.decode(sys.stdout.encoding))
        f.close()
        flag = False

    else:
        f= open(file_name,'a')
        f.write('\n************************************************************************\n')
        for line in data:
            f.write(line.decode(sys.stdout.encoding))
        f.close()
    if ret == True:
        return data


if 'temp' not in os.listdir(repo_path):
    os.mkdir('temp')

#step 1
#Tensorflow downloading.
print('Downloading TensorFlow')
if 'TensorFlow' not in os.listdir(repo_path):
    download_file(git_url,'git',tensorflow_path)
print('Downloading TensorFlow completed')

#Protobuf
print('Downloading protobuf')
if 'protobuf' not in os.listdir(repo_path):
    os.mkdir('protobuf')
    os.chdir(os.path.join(repo_path,'protobuf'))
    download_file(protobuf_url,'zip')
    os.chdir(repo_path)
print('Downloading protobuf complete')

#downloading labelimg
print('Downloading labelimg')
if 'labelimg' not in os.listdir(repo_path):
    os.mkdir('labelimg')
    os.chdir(os.path.join(repo_path,'labelimg'))
    download_file(label_img,'zip')
    os.chdir(repo_path)
print('Downloading labelimg completed')

#creating workspace
print('Creating workspace')
if 'workspace' not in os.listdir(repo_path):
    os.mkdir('workspace')
    os.chdir(os.path.join(repo_path,'workspace'))
    for directory in ['annotations','images','pre-trained-model','training']:
        os.mkdir(directory)
        if directory == 'images':
            os.mkdir('images/train')
            os.mkdir('images/test')
    os.chdir(repo_path)
print('Creating workspace complete')


#Compiling portos
print('Compiling protos')
protoc_path = os.path.join(os.getcwd(),'protobuf','bin')
cmd1 = "for /f %i in ('dir /b "
cmd2 = "object_detection\protos\*.proto') do "
cmd3 = os.path.join(protoc_path,'protoc ')
cmd4 = "object_detection\protos\%i --python_out=."
protoc_cmd = cmd1 + cmd2 + cmd3 +cmd4
os.chdir(os.path.join(tensorflow_path,'research'))
execute(protoc_cmd)
os.chdir(repo_path)
print('Compiling protos complete')

#environment
print('Building object-detection')
research_path = os.path.join(tensorflow_path,'research')
os.chdir(research_path)
execute('python setup.py build')
execute('python setup.py install')
os.chdir(repo_path)
print('Building object detection complete')



# #opening label img
# print('Opening label img')
# label_path = os.path.join(repo_path,'labelimg','windows_v1.8.0')
# os.chdir(label_path)
# execute('labelImg.exe')
# os.chdir(repo_path)



###creating label_map
print('*******************************************\n\nOnly carry on if u have placed your data in train and test directory and completed labelling of images using label img software.\nElse close the python shell label the data and rerun this file\n\n************************************************\n\n')
no_objects = int(input("Enter the number of object to be detected : "))
objects = []
for i in range(no_objects):
    temp = input('Object name of {} :'.format(i+1))
    objects.append(temp)

# no_objects = 2
# objects = ['cat','dog']

f = open(os.path.join(repo_path,'workspace','annotations','label_map.pbtxt'),'wb+')
for i in range(no_objects):
    f.write('item {\n'.encode('utf-8'))
    f.write('   name: "{}"\n'.format(objects[i]).encode('utf-8'))
    f.write('   id: {}\n'.format(i+1).encode('utf-8'))
    f.write('}\n'.encode('utf-8'))
    f.write('\n'.encode('utf-8'))
f.close()


#converting xml - csv
print('Converting xml to csv')
cmd1 = 'python xml_to_csv.py -i '
cmd2 = os.path.join(repo_path,'workspace','images','train -o ')
cmd3 = os.path.join(repo_path,'workspace','annotations','train_labels.csv')
cmd4 = os.path.join(repo_path,'workspace','images','test -o ')
cmd5 = os.path.join(repo_path,'workspace','annotations','test_labels.csv')

train_csv = cmd1 + cmd2 + cmd3
test_csv = cmd1 + cmd4 + cmd5

dtrain = execute(train_csv, ret = True)
if re.search('##Successfully converted xml to csv.##',str(dtrain)):
    print('Converting train xml to csv completed Successfully')
else:
    print('ERROR in Converting XML \n1. Might be a path problem \n2. check for spaces in path')
    print('command = {}'.format(train_csv))
    print()
    execute(train_csv, out = True)
    exit()

dtest = execute(test_csv, ret = True)
if re.search('##Successfully converted xml to csv.##',str(dtest)):
    print('Converting test xml to csv completed Successfully')
else:
    print('ERROR in Converting XML \n1. Might be a path problem \n2. check for spaces in path')
    print('command = {}'.format(test_csv))
    print()
    execute(test_csv, out = True)
    exit()

#creating generate_tfrecords
print('Generating Tfrecords')
if 'temp' not in os.listdir(repo_path):
    os.mkdir('temp')
f = open(os.path.join(repo_path,'temp','temp.txt'),'w+')
f.write('def class_text_to_int(row_label):\n')
f.write("   if row_label == '{}':\n".format(objects[0]))
f.write('         return {}\n'.format(1))
for i in range(1,no_objects):
    f.write("   elif row_label == '{}':\n".format(objects[i]))
    f.write('         return {}\n'.format(i+1))
f.write('   else:\n')
f.write('         None')
f.close()

f = open('generate_tfrecord.py','r')
filedata = f.read()
f.close()
f = open('temp/temp.txt','r')
data = f.read()
f.close()
newdata = filedata.replace("##REPLACE_HERE_9999##",data)
f = open('generate_tfrecord1.py','w+')
f.write(newdata)
f.close()

annot_path = os.path.join(repo_path,'workspace','annotations')
img_path = os.path.join(repo_path,'workspace','images')
tf_records_train = "python generate_tfrecord1.py --csv_input={}/train_labels.csv --img_path={}/train  --output_path={}/train.record".format(annot_path,img_path,annot_path)
tf_records_test = "python generate_tfrecord1.py --csv_input={}/test_labels.csv --img_path={}/test  --output_path={}/test.record".format(annot_path,img_path,annot_path)



dtrain_rec = execute(tf_records_train, ret = True)
if re.search('#*#Successfully created the TFRecords#*#',str(dtrain_rec)):
    print('Converting train csv to record completed Successfully')
else:
    print('ERROR in Converting train csv to record \n1. Might be a path problem \n2. check for spaces in path')
    print('command = {}'.format(tf_records_train))
    print()
    execute(tf_records_train,out = True)
    exit()

print()

dtest_rec = execute(tf_records_test, ret = True)
if re.search('#*#Successfully created the TFRecords#*#',str(dtest_rec)):
    print('Converting test csv to record completed Successfully')
else:
    print('ERROR in Converting test csv to record \n1. Might be a path problem \n2. check for spaces in path')
    print('command = {}'.format(tf_records_test))
    print()
    execute(tf_records_test,out = True)
    exit()



#downloading model
data = pd.read_csv('model_zoo1.csv')
df = pd.DataFrame(data)
display = pd.DataFrame([df.name,df.speed,df.mAP]).transpose()
print(display)
model_no = int(input('Enter the model no : '))
pre_path = os.path.join(repo_path,'workspace','pre-trained-model')
os.chdir(pre_path)
model_url = df['link'][model_no]
if df['name'][model_no] not in os.listdir(os.getcwd()):
    print('download model {}'.format(df['name'][model_no]))
    download_file(model_url,'tar')
    print('model downloaded')
os.chdir(repo_path)


#generating config
print('Generating config')

training_path = os.path.join(repo_path,'workspace','training',df['name'][model_no])
if not os.path.isdir(training_path):
    os.mkdir(training_path)
model_path = os.path.join(pre_path,df['name'][model_no])

config_url = df['config'][model_no]
r = requests.get(config_url)
f = open(training_path+r'\new.config','wb+')
for line in r:
    f.write(line)
f.close()

f = open(training_path+r'\new.config','r')
filedata = f.read()

#numclasses
filedata = filedata.replace(r"num_classes: 90",r'num_classes: {}'.format(no_objects))

#checkpoint and from_detection_checkpoint
r1 = r'fine_tune_checkpoint: "{}\model.ckpt"'.format(model_path)
r1 = r1.replace('\\','/')
r2 = '\n  from_detection_checkpoint: true'
if re.search('from_detection_checkpoint: true',filedata):
    filedata = filedata.replace(r'fine_tune_checkpoint: "PATH_TO_BE_CONFIGURED/model.ckpt"',r1)
if re.search('from_detection_checkpoint: false',filedata):
    filedata = filedata.replace(r'fine_tune_checkpoint: "PATH_TO_BE_CONFIGURED/model.ckpt"',r1)
    filedata = filedata.replace(r'from_detection_checkpoint: false"',r2)
if ((not re.search('from_detection_checkpoint: false',filedata)) or (not re.search('from_detection_checkpoint: true',filedata))):
    r3 = r1 + r2
    filedata = filedata.replace(r'fine_tune_checkpoint: "PATH_TO_BE_CONFIGURED/model.ckpt"',r3)



#eval number of examples
n_test = len(set(pd.read_csv(os.path.join(annot_path,'test_labels.csv'))['filename']))
r1 = r'num_examples: {}'.format(n_test)
filedata = filedata.replace(r'num_examples: 8000',r1)

#train_record
annot_path = os.path.join(repo_path,'workspace','annotations')
r1 = r'input_path: "{}\train.record"'.format(annot_path)
r1 = r1.replace('\\','/')
filedata = filedata.replace(r'input_path: "PATH_TO_BE_CONFIGURED/mscoco_train.record"',r1)
filedata = filedata.replace(r'input_path: "PATH_TO_BE_CONFIGURED/mscoco_train.record-00000-of-00100"',r1)
filedata = filedata.replace(r'input_path: "PATH_TO_BE_CONFIGURED/mscoco_train.record-?????-of-00100"',r1)

#label_map
r1 = r'label_map_path: "{}\label_map.pbtxt"'.format(annot_path)
r1 = r1.replace('\\','/')
filedata = filedata.replace(r'label_map_path: "PATH_TO_BE_CONFIGURED/mscoco_label_map.pbtxt"',r1)

#test record
r1 = r'input_path: "{}\test.record"'.format(annot_path)
r1 = r1.replace('\\','/')
filedata = filedata.replace(r'input_path: "PATH_TO_BE_CONFIGURED/mscoco_val.record"',r1)
filedata = filedata.replace(r'input_path: "PATH_TO_BE_CONFIGURED/mscoco_val.record-00000-of-00010"',r1)
filedata = filedata.replace(r'input_path: "PATH_TO_BE_CONFIGURED/mscoco_val.record-?????-of-00010"',r1)



filedata = filedata.encode('utf-8')
f = open(r'{}\new.config'.format(training_path),'wb+')
f.write(filedata)
f.close()
print('Config generated')


#training
train_file_path = os.path.join(research_path,'slim')
copyfile(os.path.join(research_path,'object_detection','legacy','eval.py'),os.path.join(train_file_path,'eval.py'))
copyfile(os.path.join(research_path,'object_detection','legacy','train.py'),os.path.join(train_file_path,'train.py'))


train_cmd = r'python {}\train.py --logtostderr --train_dir={}\ --pipeline_config_path={}\new.config'.format(train_file_path,training_path,training_path)

eval_cmd = r'python {}\eval.py --logtostderr --checkpoint_dir={}\ --pipeline_config_path={}\new.config --eval_dir=workspace\eval_{}'.format(train_file_path,training_path,training_path,df['name'][model_no])

tensorboard_training = r'tensorboard --logdir {}'.format(training_path)

tensorboard_evaluation = r'tensorboard --logdir {}'.format(os.path.join(repo_path,'workspace','eval_{}'.format(df['name'][model_no])))

print('\n\nEXECUTE THIS COMMAND IN CMP PROMPT\n\n')
print('Training command')
print(train_cmd)

print('\n')
print('Evaluation command')
print(eval_cmd)

print('\n')
print('Tensorboard Training command')
print(tensorboard_training)

print('\n')
print('Tensorboard Evaluation command')
print(tensorboard_evaluation)


f = open('training_command.txt','w+')
f.write('###Training Command###\n')
f.write(train_cmd)
f.write('\n\n###Evaluation Command###\n')
f.write(eval_cmd)
f.write('\n\n###Tensorboard Training Command###\n')
f.write(tensorboard_training)
f.write('\n\n###Tensorboard Evaluation Command###\n')
f.write(tensorboard_evaluation)


print('\nFinish')
