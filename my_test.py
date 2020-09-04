from deepface import DeepFace
import pandas as pd
import numpy as np

##Script for collecting results from given dataset with deepface

#labeled data
labeled_datas = pd.read_csv("formatted-data.csv")

labels = labeled_datas["label"]
labels = np.array(labels)

all_pairs = []
# a = 0
## verify method takes images as pairs
for index, rows in labeled_datas.iterrows(): 
    # a += 1
    # Create list for the current row 
    my_list =[rows.image1, rows.image2] 
      
    # append the list to the final list 
    all_pairs.append(my_list) 
    

del labeled_datas

pred = []
lenght = len(all_pairs)

# for i in range(0,41):
#     if i % 5 == 0:
#         print("{}/{}".format(i, lenght))
#         with open("output.txt", "a+") as txt_file:
#             for line in pred:
#                 txt_file.write(str(line) + "\n") # works with any number of elements in a line
#             pred = []
#     pair = all_pairs[i]
#     print(len(pair))
#     """result = DeepFace.verify([pair], enforce_detection = False)
#     pred.append(result["pair_1"]["verified"])"""
#     try:
#         result = DeepFace.verify([pair])
#         pred.append(result["pair_1"]["verified"])

#     except ValueError:
#         pred.append(2)

### first line runs deepface with default vgg model and second line runs it with Facenet model models will be automaticly downloaded
# result = DeepFace.verify(all_pairs, enforce_detection= False)
result = DeepFace.verify(all_pairs, enforce_detection= False, model_name="Facenet")

#print(result)

### write results to txt file
with open ("facenet_model_output.txt", "w") as file:
    for i in range(0,4000):
    # for i in range(0,len(all_pairs[0:10])):
        file.write(str(result["pair_" + str(i+1)]["verified"]) + "\n")
    # i = 1
    # for res in result:
    #     print(res)
    #     file.write(str(res["pair_" + str(i)]["verified"]) + "\n")
    #     i += 1


""""try:
    # print(pair)
    result = DeepFace.verify([pair])
    pred.append([0, 1][result["pair_1"]["verified"]])

except ValueError:
    pred.append(2)"""

