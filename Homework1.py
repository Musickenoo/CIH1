import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt1
#อ่านไฟส์ data.txt
def read_file(filename):
    matrix = np.loadtxt(filename, dtype=float)
    return matrix

def split_and_shuffle_data(data, train_ratio=0.8):
    np.random.shuffle(data)
    total_samples = len(data)
    train_size = int(total_samples * train_ratio)
    train_data = data[:train_size]
    test_data = data[train_size:]
    return train_data, test_data

#data =[CA , CL , TA , TS , NCL , TL , ET , EBI , CFO , NP , SHE , CGR ]
def input_data(data):
    input_array = np.array([])  # สร้าง numpy array ว่างสำหรับเก็บข้อมูล input
    
    for row in data:
        CA, CL, TA, TS, NCL, TL, TE, EB, CFO, NP, SE, CGR, _, _ = row  # ใช้ 15 ค่าแทน 12 ค่า
        
        CACL = CA / CL
        WCTA = (CA - CL) / TA
        CATA = CA / TA
        CFOCL = CFO / CL
        TITA = TS / TA
        NCLTA = NCL / TA
        NCLTL = NCL / TL
        TLTA = TL / TA
        TLTE = TL / TE
        TETA = TE / TA
        EBTL = EB / TL
        CFOTL = CFO / TL
        ROA = NP / TA
        ROE = NP / SE
        CGR = CGR/1000
        row_input = np.array([CACL, WCTA, CATA, CFOCL, TITA, NCLTA, NCLTL, TLTA, TLTE, TETA, EBTL, CFOTL, ROA, ROE, CGR])
        
        if input_array.size == 0:
            input_array = row_input  # ถ้า numpy array ว่างเริ่มใหม่ด้วยข้อมูลแถวแรก
        else:
            input_array = np.vstack([input_array, row_input])  # นำข้อมูลแถวใหม่มาเชื่อมต่อกับ numpy array ที่มีอยู่แล้ว
    
    return input_array

# ฟังก์ชันคำนวณ sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# ฟังก์ชันคำนวณ sigmoid derivative
def sigmoid_derivative(x):
    return x * (1 - x)

#อัพเดทค่าออกจาก hidden layer และ output layer
def forward_propagation(input_data):
    hidden = sigmoid(np.dot(w_input_to_hidden, input_data.T) + b_hidden)
    output = sigmoid(np.dot(w_hidden_to_output, hidden) + b_output)

    return hidden, output

#อัพเดทค่า weight ระหว่าง input เข้า hidden note และ bias เข้า hidden note
def update_input_hidden_layer_weights(input_data, hidden_gradient, learning_rate, momentum_rate):
    global w_input_to_hidden, b_hidden, v_w_input_hidden, v_b_hidden
    v_w_input_hidden = (momentum_rate * v_w_input_hidden) + (learning_rate * np.dot(hidden_gradient, input_data) / len(input_data))
    w_input_to_hidden += v_w_input_hidden

    v_b_hidden = (momentum_rate * v_b_hidden) + (learning_rate * np.mean(hidden_gradient, axis=1, keepdims=True))
    b_hidden += v_b_hidden
    
#อัพเดทค่า weight ระหว่าง hidden note เข้า output note และ bias เข้า output note
def update_hidden_output_layer_weights(hidden, output_gradient, learning_rate, momentum_rate):
    global w_hidden_to_output, b_output, v_w_hidden_output, v_b_output
    v_w_hidden_output = (momentum_rate * v_w_hidden_output) + (learning_rate * np.dot(output_gradient, hidden.T) / len(hidden))
    w_hidden_to_output += v_w_hidden_output

    v_b_output = (momentum_rate * v_b_output) + (learning_rate * np.mean(output_gradient, axis=1, keepdims=True))
    b_output += v_b_output
    
#นำชุดข้อมูลมาเทรน 
def train_custom_neural_network(inputdata, outputdata, Target_Epochs, Mean_Squared_Error, learning_rate, momentum_rate):
    error_array = []

    for epochs in range(Target_Epochs):
        hidden, output = forward_propagation(inputdata)

        output_error = outputdata - output.T
        output_gradient = output_error.T * sigmoid_derivative(output)
        update_hidden_output_layer_weights(hidden, output_gradient, learning_rate, momentum_rate)

        hidden_error = np.dot(w_hidden_to_output.T, output_gradient)
        hidden_gradient = hidden_error * sigmoid_derivative(hidden)
        update_input_hidden_layer_weights(inputdata, hidden_gradient, learning_rate, momentum_rate)

        error = np.mean(output_error**2, axis=0)
        if epochs % 10000 == 0:
            print(f"Epoch loop: {epochs + 10000}, Error: {error}")

        if np.all(error <= Mean_Squared_Error):
            break

        error_array.append(error)

    error_array = np.array(error_array)  # แปลงเป็น NumPy array
    return error_array

def calculate_accuracy(TP, TN, FP, FN):
    total_predictions = TP + TN + FP + FN
    if total_predictions == 0:
        return 0.0
    accuracy_percentage = ((TP + TN) / total_predictions) * 100
    return accuracy_percentage
        
file = "financial_data.txt"
data = read_file(file)  # อ่านข้อมูลจากไฟล์
train, test = split_and_shuffle_data(data)


# กำหนดขนาด Input layer, Hidden layer , Output layer จากชุดข้อมูลที่กำหนดให้
input_size = 15
hidden_size = 4 
output_size = 2

#initialize weight แตกต่างกัน โดย สร้างตัวแปร array สุ่มค่า weight และ bias ปัจจุบัน รวมถึง สร้างตัวแปร array สุ่มค่า weight และ bias ก่อนหน้า
#weight ระหว่าง input note เข้า hidden note
w_input_to_hidden = np.random.randn(hidden_size, input_size)
v_w_input_hidden = np.random.randn(hidden_size, input_size)

#weight ระหว่าง hidden note เข้า output note
w_hidden_to_output = np.random.randn(output_size, hidden_size)
v_w_hidden_output = np.random.randn(output_size, hidden_size)
    
#bias เข้า hidden note 
b_hidden = np.random.randn(hidden_size, 1)
v_b_hidden = np.random.randn(hidden_size, 1)
    
#bias เข้า  note 
b_output = np.random.randn(output_size, 1)
v_b_output = np.random.randn(output_size, 1)

# ปรับ learning_rates และ momentum_rates ตามที่ต้องการ
learning_rates = [0.01]
momentum_rates = [0.1]

for lr in learning_rates:
    for momentum in momentum_rates:
        print(f"Training with learning rate = {lr} and momentum = {momentum}")
        
        #แปลงข้อมูล train ในอยู่ในช่วง 0-1 โดยการใช้ normalize
        
        train_input = input_data(train)  
        train_output = train[:, 12:14]
        
        #นำข้อมูล train มาฝึกโดยสามารถกำหนด จำนวน epoch และ ค่าคลาดเคลื่อนเฉลี่ย MSE ที่ต้องการได้ 
        error_array=train_custom_neural_network(train_input,train_output,100000, 0.0001, lr, momentum)
        
        #แปลงข้อมูล test ในอยู่ในช่วง 0-1 โดยการใช้ normalize
        test_input = input_data(test)  
        
        x,Predict = forward_propagation(test_input)
        Predict = np.transpose(Predict) 
        Actual = test[:, 12:14]
        
    ########################Plotting
    plt.plot(range(1, len(error_array) + 1), error_array[:, 0], marker='o', label='Error type I')
    plt.plot(range(1, len(error_array) + 1), error_array[:, 1], marker='x', label='Error type II')

    plt.xscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Squared Error')
    plt.title("Train data")
    plt.grid(True)
    plt.legend()
    plt.show()
    
    # สร้าง Confusion Matrix
    confusion = confusion_matrix(np.argmax(Actual, axis=1), np.argmax(Predict, axis=1))
    TP = confusion[1, 1]  # True Positives
    TN = confusion[0, 0]  # True Negatives
    FP = confusion[0, 1]  # False Positives
    FN = confusion[1, 0]  # False Negatives
    Accuracy = calculate_accuracy(TP, TN, FP, FN)

    print("True Positive (TP):", TP)
    print("True Negative (TN):", TN)
    print("False Positive (FP):", FP)
    print("False Negative (FN):", FN)
    print(f"************Accuracy = {Accuracy} % **************")
    
    # แปลงค่าทำนายเป็น binary (0 หรือ 1) โดยกำหนด threshold เป็น 0.5
    threshold = 0.5
    predicted_binary = (Predict > threshold).astype(int)

    # สร้าง Confusion Matrix
    confusion_matrix = np.zeros((2, 2))
    for i in range(len(Actual)):
        confusion_matrix[Actual[i].argmax()][predicted_binary[i].argmax()] += 1
    # สร้าง Heatmap
    plt1.imshow(confusion_matrix, cmap="Blues", interpolation="nearest")
    plt1.colorbar()
    plt1.xticks([0, 1], ["Risk", "Non-Risk"])
    plt1.yticks([0, 1], ["Risk", "Non-Risk"])
    plt1.xlabel("Predicted")
    plt1.ylabel("Actual")
    plt1.title("Confusion Matrix")
    for i in range(2):
        for j in range(2):
            plt1.text(j, i, int(confusion_matrix[i, j]), ha="center", va="center", color="red", fontsize=18)
    plt1.show()