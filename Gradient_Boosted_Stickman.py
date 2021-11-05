#Logo_test
from imageio import imread         
from PIL import Image
import os 
import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd 
import h2o 
from h2o.estimators.gbm import H2OGradientBoostingEstimator
from h2o.grid.grid_search import H2OGridSearch

def logo_maker():
    im = Image.open('Stickman.png')       
    width, height = im.size       
    if width/height != 250/250: 
        if width > height:
            width = height 
        else:
            height = width    
    left = 0
    top = 0
    right = width 
    bottom = height  
    
    im1 = im.crop((left, top, right, bottom))   
    im1.save("my_image_square.png", "PNG")
    im1 = imread("my_image_square.png")  
    os.remove("my_image_square.png")
    
    xs_in,xs_out = [],[] 
    ys_in,ys_out = [],[]  
    xsize = 1 
    ysize = 1 
    n = 5
    for i in range(int(len(im1[0])/n)):
        for j in range(int(len(im1[0])/n)):
            if im1[int(j*n)][int(i*n)][0] < 255:
                xs_in.append(i*(xsize/len(im1[0]))-(xsize/2))
                ys_in.append(-j*(ysize/len(im1[0]))+(ysize/2)) 
            else:
                xs_out.append(i*(xsize/len(im1[0]))-(xsize/2))
                ys_out.append(-j*(ysize/len(im1[0]))+(ysize/2)) 
                
    np.random.seed(1234)  
    
    x_ran = []#np.random.rand(int(len(xs)/20))
    y_ran = []#np.random.rand(int(len(ys)/20))             
    
    x_ran_out = []#np.random.rand(int(len(xs)/20))
    y_ran_out = []
    
    in_or_out = [] 
    D = 25
    n = int(len(xs_in)/D)
    n2 = int(len(xs_out)/D)
    
    Train_data = [] 
    for i in range(n): 
        a = np.random.randint(len(xs_in))
        x_ran.append(xs_in[a])
        y_ran.append(ys_in[a])
        in_or_out.append(0)
        Train_data.append([xs_in[a],ys_in[a],0])
    for j in range(n2):
        b = np.random.randint(len(xs_out))
        x_ran_out.append(xs_out[b])
        y_ran_out.append(ys_out[b])
        in_or_out.append(1)
        Train_data.append([xs_out[b],ys_out[b],1])
        
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_aspect('equal')       

    df = pd.DataFrame(Train_data,columns=['x','y','in_or_out'])
    return df,x_ran,y_ran,x_ran_out,y_ran_out


def h2o_GBM(filename):
    h2o.init()
    year1 = h2o.import_file(filename) 
    year1['in_or_out'] = year1['in_or_out'].asfactor()
    year1_split = year1.split_frame(ratios = [0.8], seed = 1234)
    
    year1_train = year1_split[0] # using 80% for training
    ##year1_test = year1_split[1]  # using the rest 20% for out-of-bag evaluation  
    
    search_criteria = {'strategy': "RandomDiscrete", 
                       'max_models': 5,
                       'seed': 1234} 
    

    hyper_params = {'sample_rate': [0.7, 0.8, 0.9],
                    'col_sample_rate': [0.7, 0.8, 0.9],
                    'max_depth': [3, 5, 7]} 
    
    
    gbm_rand_grid = H2OGridSearch(H2OGradientBoostingEstimator(
                            model_id = 'gbm_rand_grid', 
                            seed = 1066,
                            ntrees = 10000,   
                            nfolds = 5,
                            fold_assignment = "Modulo",               # needed for stacked ensembles
                            keep_cross_validation_predictions = True, # needed for stacked ensembles
                            stopping_metric = 'mse', 
                            stopping_rounds = 15,     
                            score_tree_interval = 1),
                        search_criteria = search_criteria, # full grid search
                        hyper_params = hyper_params) 
                        
    gbm_rand_grid.train(x = ['x','y'], 
                        y = 'in_or_out', 
                        training_frame = year1_train)                     
    return gbm_rand_grid,year1_train
  
df,x_ran,y_ran,x_ran_out,y_ran_out = logo_maker()

df.to_csv('inman.csv',index = False)

gbm_rand_grid,year1_train = h2o_GBM('inman.csv')  

n = 1000
Test_data = [] 
xsize = 2 
ysize = 2 
for x in range(n):
    for y in range(n):
        Test_data.append([x*(xsize/(n))-(xsize/2),y*(ysize/(n))-(ysize/2)]) 

df2 = pd.DataFrame(Test_data,columns=['x','y'])
h2o_frame = h2o.H2OFrame(df2) 
predict = gbm_rand_grid[0].predict(h2o_frame) 
data_as_list = h2o.as_list(predict, use_pandas=False) 
test_df = pd.DataFrame(data_as_list,columns=['in-or_out','p0','p1']) 
tesr_class = test_df['in-or_out'][1:len(test_df)] 
Val_xy_df = pd.concat([tesr_class, df2], axis=1, join="inner") 


mlcircx = [] 
mlcircy = [] 
for i in range(2,len(Val_xy_df)):
    if int(Val_xy_df['in-or_out'][i-1]) + int(Val_xy_df['in-or_out'][i]) == 1:
        mlcircx.append(Val_xy_df['x'][i])
        mlcircy.append(Val_xy_df['y'][i])
pospos = []
posneg = []
negneg = []
negpos = []
for i in range(len(mlcircx)):
    if mlcircx[i] > 0 and mlcircy[i] > 0:
        pospos.append([mlcircx[i],mlcircy[i]]) 
    elif mlcircx[i] > 0 and mlcircy[i] < 0:
        posneg.append([mlcircx[i],mlcircy[i]])
    elif mlcircx[i] < 0 and mlcircy[i] < 0:
        negneg.append([mlcircx[i],mlcircy[i]])
    else:
        negpos.append([mlcircx[i],mlcircy[i]])

pospos_df = pd.DataFrame(pospos,columns=['x','y'])
posneg_df = pd.DataFrame(posneg,columns=['x','y'])[::-1]
negneg_df = pd.DataFrame(negneg,columns=['x','y'])[::-1]
negpos_df = pd.DataFrame(negpos,columns=['x','y'])
negpos_df = negpos_df[0:len(negpos_df)-2]
All = pd.concat([pospos_df,posneg_df,negneg_df,negpos_df])

plt.plot(x_ran,y_ran,'r.', label = 'Inside Stickman')
plt.plot(x_ran_out,y_ran_out,'b.',label = 'Outside Stickman')
plt.plot(All['x'],All['y'],'k.',label='GBM Generated Border') 
plt.title('Stickman')
plt.xlabel('x') 
plt.ylabel('y') 


model = gbm_rand_grid[0]


plt.rcdefaults()
fig, ax = plt.subplots()
variables = gbm_rand_grid[0]._model_json['output']['variable_importances']['variable']
y_pos = np.arange(len(variables))
scaled_importance = gbm_rand_grid[0]._model_json['output']['variable_importances']['scaled_importance']
ax.barh(y_pos, scaled_importance, align='center', color='green', ecolor='black')
ax.set_yticks(y_pos)      
ax.set_yticklabels(variables,fontsize=10)
ax.invert_yaxis()
ax.set_xlabel('Scaled Importance')
ax.set_title('Variable Importance ')
plt.show()
plt.savefig('The_Variable_importance_top_2.png')
plt.tight_layout()

cols = ['x','y'] 

pdp = model.partial_plot(data = year1_train, cols = cols, plot=False,nbins = 100) 
