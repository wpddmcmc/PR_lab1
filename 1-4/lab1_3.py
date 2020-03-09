import numpy as np
import matplotlib.pyplot as plt
import csv

def SDLA():
    test_x = np.array([[0,0],[1,0],[2,1],[0,1],[1,2]])
    test_class = np.array([1,1,1,0,0])
    w = np.array([-1.5,5,-1])
    lr = 1
    i = 1
    while(i<11):
        print("Epoch%d:"%(i))
        for n in range(len(test_class)):
            x_k = np.append(np.array([1]),test_x[n])
            wx_k = np.dot(w,x_k)
            if wx_k<=0:
                H = 0
            else:
                H = 1
            w = w + lr*(test_class[n]-H)*x_k
            print(w)

            for n in range(len(test_class)):
                if test_class[n] == 1:
                    plt.plot(test_x[n][0],test_x[n][1],'bx')
                if test_class[n] == 0:
                    plt.plot(test_x[n][0],test_x[n][1],'ro')
            temp_x = np.arange(-3.1, 3.1, 0.1)
            plot_x = []
            plot_y = []
            for x in temp_x:
                temp_y = -(w[1]/w[2])*x-(w[0]/w[2])
                if temp_y<2.1 and temp_y>-2.1:
                    plot_x.append(x)
                    plot_y.append(temp_y)
            plt.plot(plot_x, plot_y)
            plt.draw()
            plt.pause(1)
            plt.close()
        i = i+1

def lab1_3():
    test_x = np.array([[0,2],[1,2],[2,1],[-3,1],[-2,-1],[-3,-2]])
    test_class = np.array([1,1,1,0,0,0])
    KingsID = [1,9,0,0,7,7,4,0]
    w = np.array([-1*KingsID[2],-1*KingsID[3],KingsID[4]])
    lr = 1
    i = 1
    iteration = 1
    with open('data3.csv','w',newline='') as f_csv:
        while(i<11):
            print("Epoch%d:"%(i))
            for n in range(len(test_class)):
                print("iteration%d:"%(iteration))
                x_k = np.append(np.array([1]),test_x[n])
                wx_k = np.dot(w,x_k)
                if wx_k<0:
                    H = 0
                elif wx_k==0:
                    H=0.5
                else:
                    H = 1
                w_old = w
                w = w + lr*(test_class[n]-H)*x_k
                print(w)
                f_writer = csv.writer(f_csv)
                f_writer.writerow([iteration,w_old,x_k,wx_k,H,test_class[n],w])
                iteration = iteration+1

                
                for n in range(len(test_class)):
                    if test_class[n] == 1:
                        dot1, =  plt.plot(test_x[n][0],test_x[n][1],'bx')
                    if test_class[n] == 0:
                        dot2, =plt.plot(test_x[n][0],test_x[n][1],'ro')
                    plt.text(test_x[n][0]+0.05,test_x[n][1]+0.05,(test_x[n][0],test_x[n][1]), ha='center', va='bottom', fontsize=8) 
                temp_x = np.arange(-3.1, 3.1, 0.1)
                dot1.set_label('class 1')
                dot2.set_label('class -1')
                plt.legend(loc='upper left')
                plot_x = []
                plot_y = []
                for x in temp_x:
                    if w[2] ==0:
                        temp_y = x
                        x = -(w[0]/w[1])
                    elif w[1] ==0:
                        temp_y = -(w[0]/w[2])
                    else:
                        temp_y = -(w[1]/w[2])*x-(w[0]/w[2])
                    if temp_y<2.5 and temp_y>-2.5:
                        plot_x.append(x)
                        plot_y.append(temp_y)
                plt.plot(plot_x, plot_y,'g')
                plt.show()
                #plt.draw()
                #plt.pause(1)
                #plt.close()
                
            i = i+1

def lab1_3_2():
    from sklearn import datasets,neighbors
    iris = datasets.load_iris()

    test_x = iris.data
    test_class = []
    for label in iris.target:
        if label == 0:
            test_class.append(0)
        else:
            test_class.append(1)
    kingsID = [1,9,0,0,7,7,4,0]
    Stest = np.array([[kingsID[0],kingsID[1],kingsID[2],kingsID[3],kingsID[4],kingsID[5],kingsID[6]],
    [kingsID[1],kingsID[2],kingsID[3],kingsID[4],kingsID[5],kingsID[6],kingsID[0]],
    [kingsID[2],kingsID[3],kingsID[4],kingsID[5],kingsID[6],kingsID[0],kingsID[1]],
    [kingsID[3],kingsID[4],kingsID[5],kingsID[6],kingsID[0],kingsID[1],kingsID[2]]])
    Stest = Stest/np.array([2.3,4,1.5,4]).reshape(-1,1)
    Xtest = Stest+np.array([4,2,1,0]).reshape(-1,1)
    Xtest = np.transpose(Xtest)
    print(Xtest.shape)
    print(iris.data.shape)
    w = np.array([-1*kingsID[2],-1*kingsID[3],kingsID[4],-1*kingsID[5],kingsID[6]])
    lr = 0.1
    i = 1
    while(i<3):
        print("Epoch%d:"%(i))
        for n in range(len(test_class)):
            x_k = np.append(np.array([1]),test_x[n])
            wx_k = np.dot(w,x_k)
            if wx_k<0:
                H = 0
            elif wx_k==0:
                H=0.5
            else:
                H = 1
            w = w + lr*(test_class[n]-H)*x_k
            print(w)
        i = i+1
    for test in Xtest:
        result = linearTU(test,w)
        print(result)
        
def linearTU(input,w):
    y=0
    for a in range(len(input)):
        y = y + input[a]*w[a+1]
    if (y-w[0])<=0:
        return 0
    else:
        return 1

#lab1_3()
lab1_3_2()