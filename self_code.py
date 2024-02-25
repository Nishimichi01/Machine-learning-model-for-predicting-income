import math
import random as rand
import numpy as np

#splitting raw data into training and testing data
def data_prepare(data,percent):
    data = data.tolist()
    processed_data = []
    train_data = []
    test_data = []

    #convert array raw data to list type
    row = 0
    while row < len(data):
        temp_processed_data = []
        col = 0
        while col < len(data[row]):
            num = float(data[row][col])
            temp_processed_data.append(num)
            col += 1
        processed_data.append(temp_processed_data)
        row += 1

    #downsampling so that value 0 in class attribute is approximately similar to value 1
    ds = [] #downsampled data list
    for i in processed_data:
        if i[len(data[0]) - 1] == 0:
            x = rand.random()
            if x < 0.4:
                ds.append(i)
        else:
            ds.append(i)

    #produce train set and test set
    for i in ds:
        x = rand.random()
        if x <= percent:
            train_data.append(i)
        elif x > percent:
            test_data.append(i)

    #hot encode output
    #determining training target
    train_target = []
    for i in train_data:
        if i[len(ds[0])-1] == 0:
            train_target.append([1,0])
        elif i[len(ds[0])-1] == 1:
            train_target.append([0,1])
    #determining testing target
    test_target = []
    for i in test_data:
        if i[len(ds[0])-1] == 0:
            test_target.append([1,0])
        elif i[len(ds[0])-1] == 1:
            test_target.append([0,1])

    separated_data = {'train' : train_data, 'test' : test_data, 'train target' : train_target, 'test target' : test_target}

    return separated_data

def network_w(n_input,n_hidden,n_output):
    input_weight = []
    hidden_weight = []
    input_list = []
    hidden_list = []

    count = 0
    counter = 0
    while count < n_hidden :
        while counter < n_input :
            input_weight.append(rand.random())
            counter += 1
        input_list.append(input_weight)
        input_weight = []
        count += 1
        counter = 0
    
    count = 0
    counter = 0
    while count < n_output :
        while counter < n_hidden :
            hidden_weight.append(rand.random())
            counter += 1
        hidden_list.append(hidden_weight)
        hidden_weight = [] #1 list store hidden weights to a Y node
        count += 1
        counter = 0

    #new arrangement for weights for new weight calculation
    count = 0
    counter = 0
    input_list_2 = []
    temp_input_list_2 = []
    hidden_list_2 = []
    temp_hidden_list_2 = []
    while count < n_input:
        while counter < n_hidden:
            temp_input_list_2.append(input_list[counter][count])
            counter += 1
        counter = 0
        input_list_2.append(temp_input_list_2)
        temp_input_list_2 = []
        count += 1
    
    count = 0
    counter = 0
    while count < n_hidden:
        while counter < n_output:
            temp_hidden_list_2.append(hidden_list[counter][count])
            counter += 1
        counter = 0
        hidden_list_2.append(temp_hidden_list_2)
        temp_hidden_list_2 = []
        count += 1

    weight = {'input weight 1' : input_list, 'hidden weight 1' : hidden_list, 'input weight 2' : input_list_2, 'hidden weight 2' : hidden_list_2}
    return weight

#step 4 and step 5
def feedforward(data,weight,row) :
    z_in = []
    z = []
    y_in = []
    y = []
    
    #calculate z-in
    for i in weight['input weight 1']:
        col = 0
        sum = 0
        for a in i :
            sum += data[row][col] * a
            col += 1
        z_in.append(sum)

    #calculate z
    for i in z_in:
        z.append(1/(1 + math.exp(-i)))

    #calculate y-in
    sum = 0
    for w in weight['hidden weight 1']:
        for w1 in w:
            sum += z[w.index(w1)] * w1
        y_in.append(sum)

    #calculate y
    for i in y_in:
        y.append(1/(1+math.exp(-i)))

    ff = {'z-in':z_in,'z':z,'y-in':y,'y':y}

    return ff

def error (lr,ff,data,weight,row):
    wc = [] #hold value for wc (weight change for hidden layer)
    delta_k = [] #hold value for each delta-k
    delta_z = [] #hold value for each delta z
    delta_zin = []
    vc = [] #values for input weights
    
    #delta k calculation
    y_pos = 0
    for y in ff['y']:
        delta_k.append((data['train target'][row][y_pos] - y) * y * (1 - y))
        y_pos += 1

    #wc calculation
    for dk in delta_k:
        temp_wc = []
        for z in ff['z']:
            temp_wc.append(lr * dk * z)
        wc.append(temp_wc)

    #delta z-in calculation
    for w in weight['hidden weight 2']:
        dz_in = 0
        for w1 in w:
            dz_in += w1 * delta_k[w.index(w1)]
        delta_zin.append(dz_in)


    #delta z calculation
    for dz_in in delta_zin:
        for z in ff['z']:
            if ff['z'].index(z) == delta_zin.index(dz_in):
                delta_z.append(dz_in * z * (1 - z))

    #vc calculation
    for dz in delta_z:
        vc_z = []
        count = 0
        for r in data['train'][row]:
            #for input in r:
            if count < len( data['train'][row]) - 1:
                vc_z.append(lr * dz * r)
                count += 1
        vc.append(vc_z)

    delta_weight = {'input' : vc, 'hidden' : wc, 'error' : delta_k}
    return delta_weight

#update weight
def update_weight(weight,delta_weight):
    new_input_weight = []
    new_hidden_weight = []
    
    #update input weight
    for i in weight['input weight 1']:
        temp_input_list_1 = []
        for old_input in i:
            temp_input_list_1.append(old_input + delta_weight['input'][weight['input weight 1'].index(i)][i.index(old_input)])
        new_input_weight.append(temp_input_list_1)
        
    weight['input weight 1'] = new_input_weight
    
    #update hidden weight
    for h in weight['hidden weight 1']:
        temp_hidden_list_1 = []
        for old_hidden in h:
            temp_hidden_list_1.append(old_hidden + delta_weight['hidden'][weight['hidden weight 1'].index(h)][h.index(old_hidden)])
        new_hidden_weight.append(temp_hidden_list_1)
        
    weight['hidden weight 1'] = new_hidden_weight

    #update (input weight 2) and (hidden weight 2)
    count = 0
    counter = 0
    input_list_2 = []
    temp_input_list_2 = []
    hidden_list_2 = []
    temp_hidden_list_2 = []
    while count < len(weight['input weight 1'][0]):
        while counter < len(weight['input weight 1']):
            temp_input_list_2.append(weight['input weight 1'][counter][count])
            counter += 1
        counter = 0
        input_list_2.append(temp_input_list_2)
        temp_input_list_2 = []
        count += 1
    weight['input weight 2'] = input_list_2
    
    count = 0
    counter = 0
    while count < len(weight['hidden weight 1'][0]):
        while counter < len(weight['hidden weight 1']):
            temp_hidden_list_2.append(weight['hidden weight 1'][counter][count])
            counter += 1
        counter = 0
        hidden_list_2.append(temp_hidden_list_2)
        temp_hidden_list_2 = []
        count += 1
    weight['hidden weight 2'] = hidden_list_2

    return weight

#testing process
def testing(data,weight,epoch,delta_weight):
    length = len(data['test'])

    #calculate error
    sum_error1 = 0
    sum_error2 = 0
    correct_prediction = 0
    for row in range(length):
        #do feedforward to see
        ff = feedforward(data['test'],weight,row)

        #calculate accuracy
        target = []
        if ff['y'][0] > ff['y'][1]:
            target = [1,0]
        elif ff['y'][0] < ff['y'][1]:
            target = [0,1]
        if target == data['test target'][row]:
            correct_prediction += 1

        #calculate error rate
        for e in delta_weight['error']:
            if(delta_weight['error'].index(e) == 0):
                sum_error1 += math.sqrt(e * e)
            if(delta_weight['error'].index(e) == 1):
                sum_error2 += math.sqrt(e * e)

    accuracy = (correct_prediction/len(data['test'])) * 100
    #calculate mean error
    sum_error1 = sum_error1/length
    sum_error2 = sum_error2/length
    total_error = ((sum_error1 + sum_error2)/2) * 100

    string2 = "Epoch:{}; Accuracy:{:.3f}%; Error rate:{:.3f}%".format(epoch+1,accuracy,total_error)
    print(string2)

#Training process
def training(data,lr,epoch,weight):
    length = len(data['train'])
    i = 0
    while i < epoch:
        for row in range(length):
            ff = feedforward(data['train'],weight,row)
            delta_weight = error(lr,ff,data,weight,row)
            weight = update_weight(weight,delta_weight)

        #Do testing for every epoch
        testing(data,weight,i,delta_weight)
        i += 1
    
#Training process
data = np.genfromtxt('Income_after change nominal to numeric_and after replace missing value_and after normalization.csv',delimiter=',',skip_header = 1)
data = data_prepare(data,.60)
lr = .5
epoch = 50
weight = network_w(5,10,2)
training(data,lr,epoch,weight)



