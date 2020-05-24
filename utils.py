import os
from torchvision.utils import make_grid
import numpy as np
import matplotlib.pyplot as plt

def visualize(image):
    image = make_grid(image,nrow=10)
    npimg = image.detach().numpy()
    npimg = (np.transpose(npimg, (1,2,0)) + 1)/2
    plt.imshow(npimg, interpolation='nearest')

# Create necessary directories    
def create_dirs(job_id):
    if not os.path.exists('logs'):
        os.mkdir('logs')
        
    if not os.path.exists('logs/'+str(job_id)):
        os.mkdir('logs/'+str(job_id))
        
    if not os.path.exists('imgs'):
        os.mkdir('imgs')
        
    if not os.path.exists('imgs/'+str(job_id)):
        os.mkdir('imgs/'+str(job_id))
        
    if not os.path.exists('models'):
        os.mkdir('models')
        
    if not os.path.exists('models/'+str(job_id)):
        os.mkdir('models/'+str(job_id))

# Log the hyperparameters of the model
def param_log(params):
    f = open('logs/'+str(params['job_id'])+'/params.txt','w')
    for k, v in params.items():
        f.write(str(k) + ':'+ str(v) + '\n')
    f.close()

