import numpy as np
import matplotlib.pyplot as plt
import torch
from model import DDPM, Conditional_Denoised_Unet
import numpy as np
import matplotlib.animation as animation

torch.manual_seed(100)
time_step = 400
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

ddpm = DDPM(Conditional_Denoised_Unet(), time_step, DEVICE, drop_prob=0)
ddpm.load_state_dict(torch.load("ddpm_model/model_75.pt"))
ddpm.to(DEVICE)
ddpm.eval()

with torch.no_grad():
    img = ddpm.sample(40, DEVICE, cls_free_guide_w=2.0)
    
    fig = plt.figure() # make figure
    print(img.shape)
    img = np.transpose(img, axes=(0, 2, 3, 1))
    
    # make axesimage object
    # the vmin and vmax here are very important to get the color map correct
    im = plt.imshow(img[0], cmap=plt.get_cmap('jet'), vmin=img.min(), vmax=img.max())

    # function to update figure
    def updatefig(j):
        # set the data in the axesimage object
        im.set_array(img[j])
        # return the artists set
        return [im]
    # kick off the animation
    ani = animation.FuncAnimation(fig, updatefig, frames=range(100), interval=100, blit=True, repeat=False)
    ani.save(filename="example.gif", writer="pillow")
    plt.show()
    