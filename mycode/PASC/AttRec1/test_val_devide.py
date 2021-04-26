# -*- coding: utf-8 -*-
#数据集分割
def data_divide():
    test_file = open("test.txt", "r", encoding="utf8")
    test_data = test_file.readlines()
    divide = int(0.5*len(test_data))
    val_data = test_data[:divide]
    test_data2 = test_data[divide:]
    val_data2 = ""
    test_data3 = ""
    for i in range(len(val_data)):
        val_data2 = val_data2 +val_data[i]
    for i in range(len(test_data2)):
        test_data3 = test_data3+test_data2[i]  
    test_file2 = open("test2.txt","w", encoding="utf8")
    val_file = open("val2.txt","w", encoding="utf8")
    test_file2.write(test_data3)
    val_file.write(val_data2)
    test_file2.close()
    val_file.close()
    test_file.close()

#作图
import matplotlib.pyplot as plt
def movie_len():
    #Cache Replace Rate图像
    x = [100,200,300,400,500,600,700,800,900,1000]
    y1 = [0.9982290562036069, 0.9983934252386015, 0.9985683987274668, 0.9987301166489803, 0.9988907741251424, 0.9989802050194423, 0.9990577185274839, 0.999121155885458, 0.9991669612348404, 0.9992163308589735]
    y2 = [0.997667020148464, 0.9980010604453886, 0.9982856132909169, 0.9984676564156796, 0.9986617179215392, 0.99882467302934, 0.9989410695349054, 0.9990402969246933, 0.9991068693295744, 0.9991770943796527]
    plt.figure(figsize=(10,8))
    plt.plot(range(len(y1)),y1,"rv-", label = "Caser")
    plt.plot(range(len(y1)),y2,"b^--", label = "AttRec")
    plt.xlabel("Cache Size", fontsize=14)
    plt.ylabel("Cache Replace Rate", fontsize=14)
    plt.xticks(range(len(y1)), x, fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=14)
    plt.show()
    
    #Network Load Rate图像
    x = [100,200,300,400,500,600,700,800,900,1000]
    y1 = [0.8229056203605515, 0.6786850477200425, 0.5705196182396607, 0.49204665959703076, 0.44538706256627786, 0.38812301166489926, 0.3404029692470838, 0.29692470837751855, 0.25026511134676566, 0.2163308589607635]
    y2 = [0.7667020148462355, 0.6002120890774125, 0.48568398727465534, 0.3870625662778367, 0.33085896076352067, 0.2948038176033934, 0.25874867444326616, 0.23223753976670203, 0.19618239660657477, 0.17709437963944857]
    plt.figure(figsize=(10,8))
    plt.plot(range(len(y1)),y1,"rv-", label = "Caser")
    plt.plot(range(len(y1)),y2,"b^--", label = "AttRec")
    plt.xlabel("Cache Size", fontsize=14)
    plt.ylabel("Network Load Rate", fontsize=14)
    plt.xticks(range(len(y1)), x, fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=14)
    plt.show()

def our_dataset():
    #Network Load Rate图像
    x=[0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    y1=[1, 0.9730190825224314, 0.9719449007961583, 0.9718817136357892, 0.970870719069885, 0.9699860988247189, 0.9692278529002907, 0.9696701630228738, 0.9676481738910654, 0.9677113610514343, 0.9656261847592569]
    y2=[1, 0.9453431062808038,  0.9330847971692152, 0.9210792366991027, 0.912359408568179, 0.9112852268419057, 0.908062681663086, 0.902881334512827, 0.902754960192089, 0.8986477947681031, 0.8948565651459623]
    y3=[1, 0.9359914065461898, 0.9204473650954126, 0.9099582964741565, 0.9012384683432326, 0.9002906609376975, 0.8960571211929735, 0.8934032604574751, 0.8904334639201315, 0.8887274105901681, 0.8870213572602047]
    
    y4=[1, 0.9994313155566789, 0.9989258182737267, 0.9982939466700367, 0.9895741185391128, 0.9814861620118792, 0.9778844938708454, 0.9742196385694427, 0.9654366232781498, 0.9584860356375584, 0.9509035763932769 ]
    
    y5=[1, 0.9282825729811702, 0.9121066599267029, 0.9059143182105396, 0.9002274737773285, 0.8995956021736383, 0.8981422974851511, 0.8962466826740806, 0.8946038165044863, 0.8927713888537849, 0.8923290787312018]
    y6=[1, 0.9357386579047138, 0.9202578036143055, 0.9109692910400606, 0.9040818905598382, 0.9016175913054467, 0.8993428535321623, 0.8958675597118666, 0.8940351320611651, 0.8915708328067736, 0.8896120308353342]
    y7=[1, 0.9724049824337272, 0.9635899073778346, 0.9579048227403385, 0.9540083040562121, 0.9507505589268604, 0.9469179175982114, 0.9432769083359949, 0.9396997764292558, 0.9354838709677419, 0.9269243053337591]
    plt.figure(figsize=(10,8))
    plt.plot(range(len(y1)),y1,"rv-", label = "Caser")
    plt.plot(range(len(y1)),y2,"b^--", label = "AttRec")
    plt.plot(range(len(y1)),y1,'b-o', label = "LFU")
    plt.plot(range(len(y1)),y2,'b-*', label = "LRU")
    plt.plot(range(len(y1)),y3,'b--', label = "FIFO")
    plt.plot(range(len(y1)),y4,'r--', label = "SVD")
    plt.plot(range(len(y1)),y5,'r-o', label = "UserCF")
    plt.plot(range(len(y1)),y6,'r-*', label = "ItemCF")
    plt.plot(range(len(y1)),y7,'g-*', label = "AttRec")
    plt.xlabel("Cache Size", fontsize=14)
    plt.ylabel("Network Load Rate", fontsize=14)
    plt.xticks(range(len(y1)), x, fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=14)
    plt.show()
    
    #Cache Replace Rate图像
    x=[100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    y1=[0.9798584905660384, 0.9895283018867934, 0.9930031446540895, 0.9945636792452842, 0.9955188679245277, 0.9961713836477999, 0.9967654986522917, 0.996981132075473, 0.997321802935012, 0.9974339622641506]
    y2=[0.9591981132075468, 0.9750235849056619, 0.9803616352201263, 0.9836438679245294, 0.9867547169811312, 0.9885613207547186, 0.9896428571428569, 0.9909257075471712, 0.9915932914046126, 0.9921509433962262]
    y3=[0.9522169811320749, 0.9703066037735861, 0.977594339622643, 0.9815683962264163, 0.9851132075471692, 0.9870676100628943, 0.988632075471698, 0.9897759433962275, 0.9907704402515735, 0.9915660377358488]
    
    y4=[0.9995754716981132, 0.9995990566037737, 0.9995754716981133, 0.998054245283019, 0.9972358490566036, 0.9972484276729562, 0.9972506738544474, 0.996774764150944, 0.9965566037735853, 0.9963349056603773]
    y5=[0.9464622641509429, 0.9671933962264161, 0.9765880503144668, 0.9813797169811327, 0.9850094339622638, 0.9873270440251583, 0.9889353099730457, 0.9901650943396234, 0.9911058700209652, 0.9919622641509434]
    y6=[0.952028301886792, 0.9702358490566049, 0.9778459119496873, 0.9820990566037745, 0.9853113207547163, 0.9874764150943403, 0.9888948787061996, 0.9901120283018875, 0.991006289308177, 0.9917594339622642]
    y7=[0.9790291262135922, 0.9861650485436904, 0.989336569579289, 0.9912621359223313, 0.9925145631067953, 0.9932766990291271, 0.9938418862690706, 0.9942718446601955, 0.9945523193096018, 0.9944466019417475]
    plt.figure(figsize=(10,8))
    plt.plot(range(len(y1)),y1,"rv-", label = "Caser")
    plt.plot(range(len(y1)),y2,"b^--", label = "AttRec")
    plt.plot(range(len(y1)),y1,'b-o', label = "LFU")
    plt.plot(range(len(y1)),y2,'b-*', label = "LRU")
    plt.plot(range(len(y1)),y3,'b--', label = "FIFO")
    plt.plot(range(len(y1)),y4,'r--', label = "SVD")
    plt.plot(range(len(y1)),y5,'r-o', label = "UserCF")
    plt.plot(range(len(y1)),y6,'r-*', label = "ItemCF")
    plt.plot(range(len(y1)),y7,'g-*', label = "AttRec")
    plt.xlabel("Cache Size", fontsize=14)
    plt.ylabel("Cache Replace Rate", fontsize=14)
    plt.xticks(range(len(y1)), x, fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=14)
    plt.show()

if __name__ == "__main__":
    our_dataset()