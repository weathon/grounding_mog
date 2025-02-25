import numpy as np

class MedianBGS():
    def __init__(self, history=30):
        self.history = history
        self.frames = []
    
    def apply(self, img):
        self.frames.append(img)
        if len(self.frames) > self.history:
            self.frames.pop(0)
        
    def getBackgroundImage(self):
        return np.median(np.stack(self.frames), axis=0)