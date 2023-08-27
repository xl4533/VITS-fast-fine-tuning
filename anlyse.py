import zipfile
import os
import shutil

# src_dir = "D:\\360MoveData\\Users\\DELL\\Desktop\\中文 - Chinese\\labels\\"
# target_dir = 'D:\\360MoveData\\Users\\DELL\\Desktop\\中文 - Chinese\\中文 - Chinese\\'
# for root,dir1,filename in os.walk(src_dir,True):
#     for index in range(len(filename)):
#         print(index)
#         if(os.path.splitext(filename[index])[1]=='.txt'):
#                 zip_filename =filename[index].split('.')[0]

#                 for root1,dir2,filename1 in os.walk(target_dir + zip_filename,True):
#                     for index1 in range(len(filename1)):
#                         if(os.path.splitext(filename1[index1])[1]=='.lab'):
#                             shutil.copy(target_dir + zip_filename +'\\'+ filename1[index1], 'C:\\Users\\xiaolu\\Desktop\\VITS-fast-fine-tuning-main\\data\\' + zip_filename +'\\'+ filename1[index1])


# src_dir = "C:\\Users\\xiaolu\\Desktop\\VITS-fast-fine-tuning-main\\data\\"
# for root,dir1,filename in os.walk(src_dir,True):
#     for index in range(len(filename)):
        
#         if(os.path.splitext(filename[index])[1]=='.lab'):
#             print(index)
#             #shutil.move(root + '\\' + filename[index],'D:\\360MoveData\\Users\\DELL\\Desktop\\中文 - Chinese\\label\\' + filename[index])
#             name = root.split('\\')[-1]
#             with open(root + '\\' + filename[index],"r",encoding='utf-8') as f:
#                 text = f.read()
#             with open("C:\\Users\\xiaolu\\Desktop\\VITS-fast-fine-tuning-main\\sample.txt","a+",encoding='utf-8') as file:
#                 file.write('./sampled_audio4ft/'+name+'/'+filename[index].replace('lab','wav')+'|'+name+'|'+'[ZH]'+text+'\n')

src_dir = "C:\\Users\\xiaolu\\Desktop\\VITS-fast-fine-tuning-main\\custom_character_voice\\"
for root,dir1,filename in os.walk(src_dir,True):
    for index in range(len(filename)):
        
        if(filename[index].startswith("processed_")):
            #print(index)
            os.remove(root + '\\' + filename[index])