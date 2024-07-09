import os
import pandas as pd

berc = '/Users/mahsa/Mahsa_Research/Synthetic_Project/Other_sources/dataset/BERC_IRIS_FAKE'
berc_files = []
for subfolders, folders, files in os.walk(berc):
    for fname in files:
        berc_files.append(os.path.join(subfolders, fname))
print("len berc_files: ", len(berc_files))


iiitd = '/Users/mahsa/Mahsa_Research/Synthetic_Project/Other_sources/dataset/IIITD_Contact_Lens_Iris_DB'
iiitd_files = []
for subfolders, folders, files in os.walk(iiitd):
    for fname in files:
        iiitd_files.append(os.path.join(subfolders, fname))
print("len iiitd_files: ", len(iiitd_files))

clark = '/Users/mahsa/Mahsa_Research/Synthetic_Project/Other_sources/dataset/LivDet-Iris-2015-Clarkson'
clark_files = []
for subfolders, folders, files in os.walk(clark):
    for fname in files:
        clark_files.append(os.path.join(subfolders, fname))
print("len clark: ", len(clark_files))


clark_2017 = '/Users/mahsa/Mahsa_Research/Synthetic_Project/Other_sources/dataset/LivDet-Iris-2017-Clarkson'
clark_2017_files = []
for subfolders, folders, files in os.walk(clark_2017):
    for fname in files:
        clark_2017_files.append(os.path.join(subfolders, fname))
print("len iiitd_2017_files: ", len(clark_2017_files))


wvu = '/Users/mahsa/Mahsa_Research/Synthetic_Project/Other_sources/dataset/LivDet17_IIIT_WVU'
wvu_files = []  
for subfolders, folders, files in os.walk(wvu):
    for fname in files:
        wvu_files.append(os.path.join(subfolders, fname))
print("len wvu_files: ", len(wvu_files))

print("Total number of files: ", len(berc_files) + len(iiitd_files) + len(clark_files) + len(clark_2017_files) + len(wvu_files))

# Create a dataframe to store the file names and their corresponding labels
berc_df = pd.DataFrame(berc_files, columns=['file_name'])
berc_df['ds_name'] = 'berc'

iiitd_df = pd.DataFrame(iiitd_files, columns=['file_name'])
iiitd_df['ds_name'] = 'iiitd'

clark_df = pd.DataFrame(clark_files, columns=['file_name'])
clark_df['ds_name'] = 'clark_2015'

clark_2017_df = pd.DataFrame(clark_2017_files, columns=['file_name'])
clark_2017_df['ds_name'] = 'clark_2017'

wvu_df = pd.DataFrame(wvu_files, columns=['file_name'])
wvu_df['ds_name'] = 'wvu'


df = pd.concat([berc_df, iiitd_df, clark_df, clark_2017_df, wvu_df], ignore_index=True)

# save file_name column into a text file
with open('/Users/mahsa/Mahsa_Research/Synthetic_Project/Other_sources/meta/other_resources.txt', 'w') as f:
    for item in df['file_name']:
        f.write("%s\n" % item)

