import subprocess
from git import Repo
import os
import requests, zipfile, io
import fileinput
import tarfile
import pandas as pd
from shutil import copyfile
import sys

#variables
git_url = 'https://github.com/tensorflow/models.git'
repo_path = os.getcwd()
tensorflow_path = os.path.join(repo_path,'TensorFlow')
protobuf_url = 'https://github.com/protocolbuffers/protobuf/releases/download/v3.9.1/protoc-3.9.1-win64.zip'
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

        
def execute(cmd,out = False):
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
if 'temp' not in os.listdir(repo_path):
    os.mkdir('temp')
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



#opening label img
print('Opening label img')
label_path = os.path.join(repo_path,'labelimg','windows_v1.8.0')
os.chdir(label_path)
execute('labelImg.exe')
os.chdir(repo_path)



###creating label_map
print('*******************************************\n\nOnly carry on if u have placed your data in train and test directory and completed labelling of images using label img software.\nElse close the python shell label the data and rerun this file\n\n************************************************\n\n')
no_objects = int(input("Enter the number of object to be detected : "))
objects = []
for i in range(no_objects):
    temp = input('Object name of {} :'.format(i+1))
    objects.append(temp)

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

execute(train_csv)
execute(test_csv)
print('Conerting xml to csv completed')

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
execute(tf_records_train)
print('\n' * 3)
execute(tf_records_test)
print('Tfrecords Generated')

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
config_path = os.path.join(pre_path,df['name'][model_no],'pipeline.config')
f = open(config_path,'r')
filedata = f.read()
f.close()

#step 0 prob
f = open('{}/changes.txt'.format(repo_path),'r')
change_data = f.read()
f.close()
filedata = filedata.replace(change_data,' ')

#numclasses
annot_path = os.path.join(repo_path,'workspace','annotations')
filedata = filedata.replace(r"num_classes: 90",r'num_classes: {}'.format(no_objects))

training_path = os.path.join(repo_path,'workspace','training',df['name'][model_no])
if not os.path.isdir(training_path):
    os.mkdir(training_path)
model_path = os.path.join(pre_path,df['name'][model_no])


#checkpoint
r1 = r'fine_tune_checkpoint: "{}\model.ckpt"'.format(model_path)
r1 = r1.replace('\\','/')
r2 = '{}\n  from_detection_checkpoint : true'
if df['name'][model_no] == 'ssd_mobilenet_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03':
    filedata = filedata.replace(r'fine_tune_checkpoint: "PATH_TO_BE_CONFIGURED/model.ckpt"',r2)
else:
    filedata = filedata.replace(r'fine_tune_checkpoint: "PATH_TO_BE_CONFIGURED/model.ckpt"',r1)
    

#label_map
r1 = r'label_map_path: "{}\label_map.pbtxt"'.format(annot_path)
r1 = r1.replace('\\','/')
filedata = filedata.replace(r'label_map_path: "PATH_TO_BE_CONFIGURED/mscoco_label_map.pbtxt"',r1)

#train_record
r1 = r'input_path: "{}\train.record"'.format(annot_path)
r1 = r1.replace('\\','/')
filedata = filedata.replace(r'input_path: "PATH_TO_BE_CONFIGURED/mscoco_train.record"',r1)
filedata = filedata.replace(r'input_path: "PATH_TO_BE_CONFIGURED/mscoco_train.record-00000-of-00100"',r1)


#test record
r1 = r'input_path: "{}\test.record"'.format(annot_path)
r1 = r1.replace('\\','/')
filedata = filedata.replace(r'input_path: "PATH_TO_BE_CONFIGURED/mscoco_val.record"',r1)
filedata = filedata.replace(r'input_path: "PATH_TO_BE_CONFIGURED/mscoco_val.record-00000-of-00010"',r1)


filedata = filedata.encode('utf-8')
f = open(r'{}\new.config'.format(training_path),'wb+')
f.write(filedata)
f.close()
print('Config generated')


#training
train_file_path = os.path.join(research_path,'slim')
copyfile(os.path.join(research_path,'object_detection','legacy','train.py'),os.path.join(train_file_path,'train.py'))

train_cmd = r'python {}\train.py --logtostderr --train_dir={}\ --pipeline_config_path={}\new.config'.format(train_file_path,training_path,training_path)

print('\n\nEXECUTE THIS COMMAND IN CMP PROMPT\n\n')
print(train_cmd)


print('\nFinish')



    
