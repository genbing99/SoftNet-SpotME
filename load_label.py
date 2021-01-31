import numpy as np
import pandas as pd

def load_excel(dataset_name):
    if(dataset_name == 'CASME_sq'):
        xl = pd.ExcelFile(dataset_name + '/code_final.xlsx') #Specify directory of excel file
    
        colsName = ['subject', 'video', 'onset', 'apex', 'offset', 'au', 'emotion', 'type', 'selfReport']
        codeFinal = xl.parse(xl.sheet_names[0], header=None, names=colsName) #Get data
    
        videoNames = []
        for videoName in codeFinal.iloc[:,1]:
            videoNames.append(videoName.split('_')[0])
        codeFinal['videoName'] = videoNames
    
        naming1 = xl.parse(xl.sheet_names[2], header=None, converters={0: str})
        dictVideoName = dict(zip(naming1.iloc[:,1], naming1.iloc[:,0]))
        codeFinal['videoCode'] = [dictVideoName[i] for i in codeFinal['videoName']]
    
        naming2 = xl.parse(xl.sheet_names[1], header=None)
        dictSubject = dict(zip(naming2.iloc[:,2], naming2.iloc[:,1]))
        codeFinal['subjectCode'] = [dictSubject[i] for i in codeFinal['subject']]
        
    elif(dataset_name=='SAMMLV'):
        xl = pd.ExcelFile(dataset_name + '/SAMM_LongVideos_V2_Release.xlsx')
    
        colsName = ['Subject', 'Filename', 'Inducement Code', 'Onset', 'Apex', 'Offset', 'Duration', 'Type', 'Action Units', 'Notes']
        codeFinal = xl.parse(xl.sheet_names[0], header=None, names=colsName, skiprows=[0,1,2,3,4,5,6,7,8,9])
    
        videoNames = []
        subjectName = []
        for videoName in codeFinal.iloc[:,1]:
            videoNames.append(str(videoName).split('_')[0] + '_' + str(videoName).split('_')[1])
            subjectName.append(str(videoName).split('_')[0])
        codeFinal['videoCode'] = videoNames
        codeFinal['subjectCode'] = subjectName
        #Synchronize the columns name with CAS(ME)^2
        codeFinal.rename(columns={'Type':'type', 'Onset':'onset', 'Offset':'offset', 'Apex':'apex'}, inplace=True) 
        print('Data Columns:', codeFinal.columns) #Final data column
    return codeFinal
    
def load_gt(dataset_name, expression_type, images, subjectsVideos, subjects, codeFinal):
    dataset_expression_type = expression_type
    if(dataset_name == 'SAMMLV' and expression_type=='micro-expression'):
        dataset_expression_type = 'Micro - 1/2'
    elif(dataset_name == 'SAMMLV' and expression_type=='macro-expression'):
        dataset_expression_type = 'Macro'
        
    vid_need = []
    vid_count = 0
    ground_truth = []
    for sub_video_each_index, sub_vid_each in enumerate(subjectsVideos):
        ground_truth.append([])
        for videoIndex, videoCode in enumerate(sub_vid_each):
            on_off = []
            for i, row in codeFinal.iterrows():
                if (row['subjectCode']==subjects[sub_video_each_index]): #S15, S16... for CAS(ME)^2, 001, 002... for SAMMLV
                    if (row['videoCode']==videoCode):
                        if (row['type']==dataset_expression_type): #Micro-expression or macro-expression
                            if (row['offset']==0): #Take apex if offset is 0
                                on_off.append([int(row['onset']-1), int(row['apex']-1)])
                            else:
                                if(dataset_expression_type!='Macro' or int(row['onset'])!=0): #Ignore the samples that is extremely long in SAMMLV
                                    on_off.append([int(row['onset']-1), int(row['offset']-1)])
            if(len(on_off)>0):
                vid_need.append(vid_count) #To get the video that is needed
            ground_truth[-1].append(on_off) 
            vid_count+=1
    
    #Remove unused video
    final_samples = []
    final_videos = []
    final_subjects = []
    count = 0
    for subjectIndex, subject in enumerate(ground_truth):
        final_samples.append([])
        final_videos.append([])
        for samplesIndex, samples in enumerate(subject):
            if (count in vid_need):
                final_samples[-1].append(samples)
                final_videos[-1].append(subjectsVideos[subjectIndex][samplesIndex])
                final_subjects.append(subjects[subjectIndex])
            count += 1
    
    #Remove the empty data in array
    final_subjects = np.unique(final_subjects)
    final_videos = [ele for ele in final_videos if ele != []]
    final_samples = [ele for ele in final_samples if ele != []]
    final_images = [images[i] for i in vid_need]
    print('Total Videos:', len(final_images))
    return final_images, final_videos, final_subjects, final_samples
        
def cal_k(dataset_name, expression_type, final_samples):
    samples = [samples for subjects in final_samples for videos in subjects for samples in videos]
    total_duration = 0
    for sample in samples:
        total_duration += sample[1]-sample[0]
    N=total_duration/len(samples)
    k=int((N+1)/2)
    print('k (Half of average length of expression) =', k)
    return k
