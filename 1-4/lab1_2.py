# LMS
import numpy as np
import matplotlib.pyplot as plt
import csv

test_x = np.array([[0,2],[1,2],[2,1],[-3,1],[-2,1],[-3,-2]])
test_class = np.array([1,1,1,-1,-1,-1])
test_b = np.array([1,9,0,0,7,7,4,0])
a = np.array([1,0,0])
lr = 0.1
i=1

iteration = 1

with open('data2.csv','w',newline='') as f_csv:
    while(i<21):
        print("Epoch%d:"%(i))
        for n in range(len(test_class)):
            print("iteration%d:"%(iteration))
            if test_class[n] == 1:
                y_k = np.append(np.array([1]),test_x[n])
            if test_class[n] ==-1:
                y_k = -1*np.append(np.array([1]),test_x[n])
            g_x = np.dot(a,y_k)
            print(g_x)
            a_old = a
            a = a + lr*(test_b[n]-np.dot(a,y_k))*y_k
            print(a)
            f_writer = csv.writer(f_csv)
            f_writer.writerow([iteration,a_old,y_k,g_x,a])
            iteration = iteration+1
            
            for n in range(len(test_class)):
                if test_class[n] == 1:
                    dot1, =  plt.plot(test_x[n][0],test_x[n][1],'bx')
                if test_class[n] == -1:
                    dot2, =plt.plot(test_x[n][0],test_x[n][1],'ro')
                plt.text(test_x[n][0]+0.05,test_x[n][1]+0.05,(test_x[n][0],test_x[n][1]), ha='center', va='bottom', fontsize=8) 
            dot1.set_label('class 1')
            dot2.set_label('class -1')
            plt.legend(loc='upper left')
            temp_x = np.arange(-3.1, 3.1, 0.1)
            plot_x = []
            plot_y = []
            for x in temp_x:
                if a[2] ==0:
                    temp_y = x
                    x = -(a[0]/a[1])
                elif a[1] ==0:
                    temp_y = -(a[0]/a[2])
                else:
                    temp_y = -(a[1]/a[2])*x-(a[0]/a[2])
                if temp_y<2.5 and temp_y>-2.5:
                    plot_x.append(x)
                    plot_y.append(temp_y)
            plt.plot(plot_x, plot_y,'g')
            plt.show()
            
        n = 0
        i=i+1