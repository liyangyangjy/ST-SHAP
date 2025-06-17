import argparse
import numpy as np
import functools
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
import os
import torch.nn.functional as F

import torch

from get_data1 import getdata
from swin_transformer import SwinTransformer

from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
import os

parser = argparse.ArgumentParser(description='MDAT')

parser.add_argument('--print_acc', default='1')
parser.add_argument('--save_models', default='1')
parser.add_argument('--print_detail', default='1')
args = parser.parse_args()

def train_XAI(file0_1,file0_2,file1_1,file1_2,best_model_path,save_path,shap_values_path,shap_summary_path,data_path):

    if_prt = args.print_acc
    if_save = args.save_models
    if_dtl = args.print_detail

    acc = 0
    topacc = 0

    # Use GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    batch_size = 16

    # Get data
    train_loader, test_loader, y_all, X_all, X_test, y_test, X_train, y_train = getdata(file0_1, file1_1, file1_2, batch_size)
    

    # Training times 
    epochs = 10
    # Learning rate
    learning_rate = 0.0001
    # Cross-entropy loss
    criterion = nn.CrossEntropyLoss()
    model = SwinTransformer()

    # Load the pre-trained model
    checkpoint = torch.load("swin_tiny_patch4_window7_224.pth")
    if checkpoint['model']['head.weight'].shape[0] == 1000:
        checkpoint['model']['head.weight'] = torch.nn.Parameter(torch.nn.init.xavier_uniform(torch.empty(2, 768)))
        checkpoint['model']['head.bias'] = torch.nn.Parameter(torch.randn(2))
    ms = model.load_state_dict(checkpoint['model'], strict=False)


    # Load the model onto the GPU
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=0, last_epoch=-1)

    num_sample = 10  # 160/batch_size
    y_true = []
    y_scores = []
    y_pred = []
    for epoch in range(epochs):
        # Set the model to training mode
        model.train()
        i = 0
        for images, labels in train_loader:
            optimizer.zero_grad()

            images = images.permute(0,3,1,2).to(device)
            images = F.interpolate(images, size=(224, 224), mode='bilinear')
            labels = labels.to(device).long()
            outputs = model(images)

            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            i = i+1

            # Print the loss
            print(f'Epoch [{epoch + 1}/{epochs}], Step [{i}], Loss: {loss.item():.4f}')

        lr_scheduler.step()

        # Switch to eval mode (testing per epoch)
        # Save the highest accuracy and its corresponding model
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in test_loader:
                images = images.permute(0,3,1,2).to(device)
                images = F.interpolate(images, size=(224, 224), mode='bilinear')
                labels = labels.to(device).long()
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                probs = torch.softmax(outputs, dim=1)
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())

                # Check if the current accuracy is the highest
            if acc < correct / total:
                topacc = correct / total
                save_state = {'model': model.state_dict()}
                # Save the highest-accuracy model (in .pth format, loaded in the same way as the pre-trained model above)
                torch.save(save_state, 'best_model_3')
            print(f'Test Accuracy: {(100 * correct / total):.2f}%')

    save_state = {'model': model.state_dict()}
    torch.save(save_state, 'best_model_path')
    #torch.save(save_state, 'best_model.pth')

    # 打印出最高精度
    print('----------------')
    print(topacc)
    print(acc)
    recall, precision, fscore, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    print(f'Recall: {recall:.2f}')
    print(f'Precision: {precision:.2f}')
    print(f'F1 Score: {fscore:.2f}')
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    plt.plot(recall, precision, marker='.')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.savefig(save_path)
    plt.show()

    #=============================================================================================lyy
    import shap

    
    image=X_test[-100]
    image=image.permute(2,0,1).to(device)
    image_size=image.shape
    image= image.view(-1, image_size[-3], image_size[-2], image_size[-1])
    image = F.interpolate(image,size=(224, 224), mode='bilinear')
    images = X_train[:16]
    images = images.permute(0,3,1,2).to(device)
    images = F.interpolate(images, size=(224, 224), mode='bilinear')
    explainer = shap.DeepExplainer(model, images)
    # Calculate SHAP values
    shap_values = explainer.shap_values(image)
    
    
    # Save SHAP values to file
    import pickle
    #with open('shap_values_2.pkl', 'wb') as file:
    with open(shap_values_path, 'wb') as file:
        pickle.dump(shap_values, file)
    
    # Visualize SHAP values if needed
    shap_numpy = [np.swapaxes(np.swapaxes(x, 1, -1), 1, 2) for x in shap_values]
    image_numpy = np.swapaxes(np.swapaxes(image.view(-1, 3, 224, 224).cpu().numpy(), 1, -1), 1, 2)
    shap.image_plot(shap_numpy, image_numpy, show=False, labels= [1,0])
    #plt.savefig('shap_summary.png')
    plt.savefig(shap_summary_path)
    
    #Retrieve the corresponding atomic weights
    #=======================================================================================================================================
    import xlsxwriter
    import xlrd
    results_list=[]
    for ii in range(2):
        shap_values_tensor=torch.tensor(shap_values[ii])
        shap_values_narrow=F.interpolate(shap_values_tensor, size=(image_size[-2], image_size[-1]), mode='bilinear')
        shap_numpy_narrow = np.swapaxes(np.swapaxes(shap_values_narrow, 1, -1), 1, 2)
        shap_val=shap_numpy_narrow[0].sum(-1)
        importance_pic = np.array(shap_val)
        #1.Generate atomic weight file
        workbook = xlsxwriter.Workbook(data_path+'/atom_train_shap_'+str(ii)+'.xlsx')
        worksheet = workbook.add_worksheet()
        size = len(importance_pic)
        for i in range(size):
            for j in range(size):
                worksheet.write(i * size + j, 0, i * size + j + 1)
                worksheet.write(i * size + j, 1, importance_pic[i][j])
        workbook.close()
        #2.Obtain atom-to-amino acid mappings
        f = open(file0_2, "r")
        data = f.readlines()
        f.close()
        p = len(data) - 1
        data.pop(p)
        data.pop(p - 1)
        bond = []
        for line in data:
            bond_line = line.split(' ')
            while '' in bond_line:
                bond_line.remove('')
            try:
                bond.append([int(bond_line[1]), int(bond_line[5])])
            except Exception as e:
                print(f"Error occurred：{e}")
        #3.Read the atomic weight file
        rbook = xlrd.open_workbook(data_path+'/atom_train_shap_'+str(ii)+'.xlsx')
        rbook.sheets()
        rsheet = rbook.sheet_by_index(0)
        for row in rsheet.get_rows():
            if (int(row[0].value) - 1) < len(bond):
                bond[int(row[0].value) - 1].append(row[1].value)
            else:
                break
        #4.Handle atomic residue mappings
        res = []
        for i in range(len(bond)):
            res.append([bond[i][1], bond[i][2]])
        #5.Calculate corresponding amino acid weights and sort them
        freq = []
        s = 0
        score = 0
        n = 0
        f = 0
        for i in range(len(res) - 1):
            if res[i][0] == res[i + 1][0]:
                score += res[i][1]
                n += 1
                if i+2==len(res):
                    score += res[i + 1][1]
                    n += 1
                    freq.append([res[i+1][0], score / n])
            else:
                score += res[i][1]
                n += 1
                freq.append([res[i][0], score / n])
                score = 0
                s = 0
                n = 0
        freq_sort = sorted(freq, key=lambda x: (x[1]))
        freq_sort.reverse()
        #6.Write the calculated amino acid weight ranking results to an xlsx file
        workbook = xlsxwriter.Workbook(data_path+'/res_score_train_shap_'+str(ii)+'.xlsx')
        worksheet = workbook.add_worksheet()
        for i in range(len(freq_sort)):
            worksheet.write(i, 0, freq_sort[i][0])
            worksheet.write(i, 1, freq_sort[i][1])
        workbook.close()
        results_list.append(freq_sort)
    return results_list,topacc,acc


def write_file(file_path, content):
    with open(file_path, 'a') as file:
        file.write(content) 

if __name__ == '__main__':
    '''
    name_list = ['1g79_58451','1vie_57451','2acy_59918','2dhn_17001','2hw4_64837',
    '3cla_17698','5fit_58529','1ast_73393','1b66_58462','1bob_28920',
    '1cqq_73848','1hfs_90799','1im5_17154','1lba_28920','1mj9_29969',
    '1o08_57684','1pau_46761','1q91_58043','1qib_73907','1qv0_16531',
    '1uch_73604','206l_87004','2ime_59350','2ixj_57649','3mak_16130']
    '''
    #test
    name_list = ['1ast_73393']
    path='../Data/'
    '''
    pdb_res=[['1g79',[178]],
    ['1vie',[14,49,50,51]],
    ['2acy',[23,41]],
    ['2dhn',[22,100]],
    ['2hw4',[47,48,88,90,72,23]],
    ['3cla',[189,13,193,168]],
    ['5fit',[97,82,95]],
    ['1ast',[92,96,102,149,93]],
    ['1b66',[36,127,42,44,17,83,82]],
    ['1bob',[241,206]],
    ['1cqq',[147,40,71,145]],
    ['1hfs',[114,118,124,115]],
    ['1im5',[51,53,70,59,100,93,132,128,9]],
    ['1lba',[42,13,126,118,124]],
    ['1mj9',[143,177]],
    ['1o08',[8,10,9,115,145,114,16,45,169,170]],
    ['1pau',[24,25,125,84,83]],
    ['1q91',[8,143,10]],
    ['1qib',[124,114,118]],
    ['1qv0',[16,86,128,165,180]],
    ['1uch',[91,85,145,160]],
    ['206l',[20,11]],
    ['2ime',[11,179,182,181,168]],
    ['2ixj',[65,74,134,171]],
    ['3mak',[9]]]
    '''
    #test
    pdb_res=[['1ast',[92,96,102,149,93]]]
    result_file_path='../Result/results.csv'
    noper_0=[]
    noper_1=[]
    np_score=[]
    for i in range(len(name_list)):
        name = name_list[i]
        names = name.split('_')

        file0_1 = path+name+'/'+names[0]+'_'+names[1]+'_c.nc'
        file0_2 = path+name+'/out.pdb'
        file1_1 = path+name+'/'+names[0]+'_c.nc'
        file1_2 = path+name+'/out.pdb'
        np_score_10=[]
        noper_10_0=[]
        noper_10_1=[]
        for time in range(100):
            folder_path=path+'results_swin-shap_100_test/results_'+str(time)+'/'+name
            if not os.path.exists(folder_path):
                # Create the folder
                os.makedirs(folder_path)
                print(f"Folder has been created：{folder_path}")
            else:
                print(f"Folder already exists：{folder_path}")
            best_model_path = folder_path+'/best_model.pth'
            save_path = folder_path+'/precision_recall_curve.png'
            shap_values_path = folder_path+'/shap_values.pkl'
            shap_summary_path = folder_path+'/shap_summary.png'
            data_path = folder_path
            results_list,topacc,acc = train_XAI(file0_1,file0_2,file1_1,file1_2,best_model_path,save_path,shap_values_path,shap_summary_path,data_path)
            #Experimental data processing: Summation and averaging
            kk=0
            np_score_10_2=[]
            for freq_sort in results_list:
                result_str = []
                first_rank = []
                for k in range(len(freq_sort)):
                    rank = k+1
                    res_id = freq_sort[k][0]
                    res_score = freq_sort[k][1]
                    if res_id in pdb_res[i][1]:
                        if len(first_rank)==0:
                            first_rank=[rank,res_id]
                        rank_res=[rank,res_id]
                        result_str.append(rank_res)
                noper=first_rank[0]/len(freq_sort)
                np_score_10_2.append(noper)
                if kk==0:
                    noper_0.append(noper)
                    noper_10_0.append(noper)
                if kk==1:
                    noper_1.append(noper)
                    noper_10_1.append(noper)
                content=names[0]+','+names[1]+','+str(pdb_res[i][1])+','+folder_path+','+str(noper)+','+str(first_rank)+','+str(result_str)+','+str(len(freq_sort))+','+'2号'+','+'shap-'+str(kk)+','+str(topacc)+','+str(acc)+'\n'
                write_file(result_file_path, content)
                kk = kk+1
            np_score_10.append(np_score_10_2)
        np_score.append([names[0]+'_'+names[1],np_score_10])
        avg_10_0=sum(noper_10_0) / len(noper_10_0)
        avg_10_1=sum(noper_10_1) / len(noper_10_1)
        content='avg_10_0:'+str(avg_10_0)+','+'avg_10_1:'+str(avg_10_1)+'\n'
        write_file(result_file_path, content)
    avg_0=sum(noper_0) / len(noper_0)
    avg_1=sum(noper_1) / len(noper_1)
    content='avg_0:'+str(avg_0)+','+'avg_1:'+str(avg_1)+'\n'
    write_file(result_file_path, content)
    for p in np_score:
        content=str(p)+'\n'
        write_file(result_file_path, content)

